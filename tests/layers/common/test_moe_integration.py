# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for which MoE program an expert-parallel call gets: the token
threshold, the fused kernel's refusals and the fallback they take, and what
the two kernels compute for the same batch."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh

from tpu_inference import envs
from tpu_inference.kernels.fused_ep_moe.fused_ep_moe_v2 import \
    WIRE_RELATIVE_DELTA_BOUND
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName,
                                                  ShardingAxisNameBase)

# moe.py imports the routed-expert layer from vllm. A tree whose installed
# vllm predates it cannot import moe.py; skip rather than fail collection.
try:
    from tpu_inference.layers.common import moe as moe_module
    from tpu_inference.layers.common import moe_fused_ep
    from tpu_inference.layers.common.moe import MoEBackend, moe_apply
    from tpu_inference.layers.common.process_weights.moe_weights import \
        FusedMoEWeights
except ImportError as error:  # pragma: no cover - environment dependent
    pytest.skip(f"the common MoE layer is not importable here: {error}",
                allow_module_level=True)

# Every test below builds an eight-device serving mesh. A machine that
# offers fewer devices cannot run any of them, so skip the file by name.
if jax.local_device_count() < 8:
    pytest.skip(
        f"this suite builds an eight-device serving mesh and jax offers "
        f"{jax.local_device_count()} device(s) here. Run it with no "
        "accelerator visible (JAX_PLATFORMS=cpu), where the suite conftest "
        "supplies eight CPU devices, or on a host with eight chips.",
        allow_module_level=True)

# The served generation. The route asks the kernel how much VMEM it has,
# which a CPU cannot answer; nothing numeric is decided by it.
SERVED_CHIP = pltpu.ChipVersion.TPU_7X

# A served-shaped expert layer, small enough to build on CPU: one local
# expert per device, hidden a whole number of the kernel's lane blocks.
NUM_EXPERTS, HIDDEN, INTER, TOP_K = 8, 512, 512, 8

# Token counts either side of the default threshold. All are acceptable to
# the fused kernel, so only the threshold separates them.
BELOW_THRESHOLD, AT_THRESHOLD, ABOVE_THRESHOLD = 1016, 1024, 2048

# The EP path carries expert output rows over the wire as fp8 e4m3 where the
# general path uses bfloat16; a zero delta would mean that stopped happening.
WIRE_DELTA_FLOOR = 1e-4
WIRE_DELTA_CEILING = WIRE_RELATIVE_DELTA_BOUND


class FakeMoELayer:
    """The attributes moe_apply and the fused-EP gate read off a layer."""

    def __init__(self, **attributes):
        self.use_ep = True
        self.top_k = TOP_K
        self.renormalize = True
        self.scoring_func = "softmax"
        self.activation = "silu"
        self.__dict__.update(attributes)

    def _get_name(self):
        return "FakeMoELayer"


def quantize_channelwise(weight):
    """Per-output-channel fp8 e4m3, the layout the fused kernels consume."""
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    amax = jnp.max(jnp.abs(weight), axis=1, keepdims=True)
    scale = amax / fp8_max
    quantized = jnp.clip(weight / jnp.where(scale == 0, 1, scale), -fp8_max,
                         fp8_max).astype(jnp.float8_e4m3fn)
    return quantized, scale


def eight_device_mesh(devices):
    """A mesh the fused-EP gate accepts: pure data-parallel attention."""
    mesh_shape = tuple(8 if axis == "attn_dp" else 1
                       for axis in MESH_AXIS_NAMES)
    return Mesh(np.array(devices).reshape(mesh_shape), MESH_AXIS_NAMES)


class MoEWeightsMixin:
    """A served-shaped quantized expert layer and a batch to send through."""

    def build_weights(self, key, with_bias=False):
        w13_key, w2_key = jax.random.split(key)
        w13 = jax.random.normal(w13_key, (NUM_EXPERTS, HIDDEN, 2 * INTER),
                                jnp.float32) / 10
        w2 = jax.random.normal(w2_key,
                               (NUM_EXPERTS, INTER, HIDDEN), jnp.float32) / 10
        w13_q, w13_scale = quantize_channelwise(w13)
        w2_q, w2_scale = quantize_channelwise(w2)
        return FusedMoEWeights(
            w13_weight=w13_q,
            w13_weight_scale=w13_scale.reshape(NUM_EXPERTS, 1, 1, 2 * INTER),
            w13_bias=(jnp.zeros(
                (NUM_EXPERTS, 2 * INTER), jnp.float32) if with_bias else None),
            w2_weight=w2_q,
            w2_weight_scale=w2_scale.reshape(NUM_EXPERTS, 1, 1, HIDDEN),
            w2_bias=(jnp.zeros(
                (NUM_EXPERTS, HIDDEN), jnp.float32) if with_bias else None),
        )

    def build_activations(self, key, num_tokens):
        x_key, gate_key = jax.random.split(key)
        x = (jax.random.normal(x_key, (num_tokens, HIDDEN), jnp.float32) /
             10).astype(jnp.bfloat16)
        gating = jax.random.normal(gate_key, (num_tokens, NUM_EXPERTS),
                                   jnp.float32)
        return x, gating


class MoERouteTestBase(MoEWeightsMixin, jtu.JaxTestCase):
    """An eight-device CPU mesh, and the hardware answers the route needs."""

    def setUp(self):
        super().setUp()
        self.cpu_devices = jax.devices("cpu")[:8]
        self.assertLen(self.cpu_devices, 8)
        # The gate takes a mesh whose attention is data-parallel everywhere.
        ShardingAxisName.override(ATTN_DATA=ShardingAxisNameBase.ATTN_DATA)
        self.addCleanup(ShardingAxisName.reset)
        self.mesh = eight_device_mesh(self.cpu_devices)

        info = pltpu.get_tpu_info_for_chip(SERVED_CHIP, 1)
        original = pltpu.get_tpu_info
        pltpu.get_tpu_info = lambda: info
        self.addCleanup(setattr, pltpu, "get_tpu_info", original)

    def route_taken(self,
                    num_tokens,
                    *,
                    weights=None,
                    extra=None,
                    backend=MoEBackend.GMM_EP):
        taken = []

        def fake_fused_ep_apply(**kwargs):
            taken.append("fused_ep")
            return jnp.zeros((kwargs["x"].shape[0], HIDDEN), jnp.bfloat16)

        def fake_fused_moe_func(**kwargs):
            taken.append("fused_moe_func")
            return jnp.zeros((kwargs["hidden_states"].shape[0], HIDDEN),
                             jnp.bfloat16)

        with jax.default_device(self.cpu_devices[0]):
            if weights is None:
                weights = self.build_weights(jax.random.key(0))
            x, gating = self.build_activations(jax.random.key(1), num_tokens)
            patched_ep = self.enter_context(
                _patched(moe_fused_ep, "moe_fused_ep_apply",
                         fake_fused_ep_apply))
            patched_general = self.enter_context(
                _patched(moe_module, "fused_moe_func", fake_fused_moe_func))
            del patched_ep, patched_general
            moe_apply(
                layer=FakeMoELayer(),
                x=x,
                gating_output=gating,
                weights=weights,
                moe_backend=backend,
                mesh=self.mesh,
                extra_backend_kwargs={
                    "scatter_results": True,
                    **(extra or {})
                },
            )
        self.assertLen(taken, 1)
        return taken[0]


class RouteSelectionTest(MoERouteTestBase):
    """Which kernel a call is handed, at and around the threshold. Both
    branches are stubbed, so only the choice is under test."""

    def test_the_default_threshold_is_the_served_one(self):
        self.assertEqual(envs.MOE_FUSED_EP_MIN_TOKENS, 1024)

    def test_below_the_threshold_runs_the_general_path(self):
        self.assertEqual(self.route_taken(BELOW_THRESHOLD), "fused_moe_func")

    def test_at_the_threshold_runs_the_fused_kernel(self):
        """The comparison is at-or-above, so the boundary count is in."""
        self.assertEqual(self.route_taken(AT_THRESHOLD), "fused_ep")

    def test_above_the_threshold_runs_the_fused_kernel(self):
        self.assertEqual(self.route_taken(ABOVE_THRESHOLD), "fused_ep")

    def test_the_below_threshold_shape_is_otherwise_acceptable(self):
        """The threshold is the only thing keeping the small batch out."""
        with jax.default_device(self.cpu_devices[0]):
            weights = self.build_weights(jax.random.key(0))
            x, gating = self.build_activations(jax.random.key(1),
                                               BELOW_THRESHOLD)
            reason = moe_fused_ep.unsupported_reason(
                layer=FakeMoELayer(),
                x=x,
                gating_output=gating,
                weights=weights,
                mesh=self.mesh,
                activation="silu",
                scatter_results=True,
                extra_backend_kwargs={},
            )
        self.assertIsNone(reason)

    def test_raising_the_threshold_moves_the_boundary(self):
        with _patched(envs, "MOE_FUSED_EP_MIN_TOKENS", ABOVE_THRESHOLD):
            self.assertEqual(self.route_taken(AT_THRESHOLD), "fused_moe_func")
            self.assertEqual(self.route_taken(ABOVE_THRESHOLD), "fused_ep")

    def test_the_tensor_parallel_backend_never_takes_the_fused_kernel(self):
        """The threshold only ever selects for the expert-parallel backend."""
        self.assertEqual(
            self.route_taken(ABOVE_THRESHOLD, backend=MoEBackend.GMM_TP),
            "fused_moe_func")


class RefusalFallbackTest(MoERouteTestBase):
    """Configurations the fused kernel has no operand for; each changes what
    the layer computes, so each has to fall back rather than be taken."""

    def refusal_for(self, *, weights=None, extra=None, layer=None):
        with jax.default_device(self.cpu_devices[0]):
            if weights is None:
                weights = self.build_weights(jax.random.key(0))
            x, gating = self.build_activations(jax.random.key(1),
                                               ABOVE_THRESHOLD)
            return moe_fused_ep.unsupported_reason(
                layer=layer or FakeMoELayer(),
                x=x,
                gating_output=gating,
                weights=weights,
                mesh=self.mesh,
                activation="silu",
                scatter_results=True,
                extra_backend_kwargs=extra or {},
            )

    def test_a_biased_layer_is_refused_by_name(self):
        with jax.default_device(self.cpu_devices[0]):
            weights = self.build_weights(jax.random.key(0), with_bias=True)
        reason = self.refusal_for(weights=weights)
        self.assertIsNotNone(reason)
        self.assertIn("no MoE bias operands", reason)

    def test_a_biased_layer_routes_to_the_general_path(self):
        with jax.default_device(self.cpu_devices[0]):
            weights = self.build_weights(jax.random.key(0), with_bias=True)
        route = self.route_taken(ABOVE_THRESHOLD, weights=weights)
        self.assertEqual(route, "fused_moe_func")

    @parameterized.named_parameters(
        ("hash_based_topk_indices", "hash_based_topk_indices"),
        ("e_score_correction_bias", "e_score_correction_bias"),
        ("num_valid_tokens", "num_valid_tokens"),
    )
    def test_a_routing_modifier_is_refused_by_name(self, name):
        reason = self.refusal_for(extra={name: jnp.zeros((4, ), jnp.int32)})
        self.assertIsNotNone(reason)
        self.assertIn(name, reason)

    def test_a_routing_modifier_routes_to_the_general_path(self):
        route = self.route_taken(
            ABOVE_THRESHOLD,
            extra={"num_valid_tokens": jnp.zeros((4, ), jnp.int32)})
        self.assertEqual(route, "fused_moe_func")

    def test_approximate_top_k_is_refused_by_name(self):
        with _patched(envs, "MOE_APPROX_TOPK", True):
            reason = self.refusal_for()
        self.assertIsNotNone(reason)
        self.assertIn("MOE_APPROX_TOPK", reason)

    def test_non_softmax_routing_is_refused_by_name(self):
        reason = self.refusal_for(layer=FakeMoELayer(scoring_func="sigmoid"))
        self.assertIsNotNone(reason)
        self.assertIn("softmax", reason)

    def test_the_all_reduced_form_is_refused_by_name(self):
        """scatter_results is part of what the kernel layer returns."""
        with jax.default_device(self.cpu_devices[0]):
            weights = self.build_weights(jax.random.key(0))
            x, gating = self.build_activations(jax.random.key(1),
                                               ABOVE_THRESHOLD)
            reason = moe_fused_ep.unsupported_reason(
                layer=FakeMoELayer(),
                x=x,
                gating_output=gating,
                weights=weights,
                mesh=self.mesh,
                activation="silu",
                scatter_results=False,
                extra_backend_kwargs={},
            )
        self.assertIsNotNone(reason)
        self.assertIn("scatter_results", reason)


class CrossThresholdNumericsTest(MoEWeightsMixin, jtu.JaxTestCase):
    """What the two kernels compute for the same batch. These run the
    kernels, so they need a TPU mesh and no substituted hardware answers."""

    def setUp(self):
        super().setUp()
        if not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+")
        devices = jax.devices()[:8]
        if len(devices) < 8:
            self.skipTest("Expect eight devices")
        ShardingAxisName.override(ATTN_DATA=ShardingAxisNameBase.ATTN_DATA)
        self.addCleanup(ShardingAxisName.reset)
        self.mesh = eight_device_mesh(devices)

    def run_both_routes(self, num_tokens, weights):
        """The same call routed each way, by moving the threshold."""
        x, gating = self.build_activations(jax.random.key(3), num_tokens)
        call = dict(layer=FakeMoELayer(),
                    x=x,
                    gating_output=gating,
                    weights=weights,
                    moe_backend=MoEBackend.GMM_EP,
                    mesh=self.mesh,
                    extra_backend_kwargs={"scatter_results": True})
        with _patched(envs, "MOE_FUSED_EP_MIN_TOKENS", num_tokens):
            fused = moe_apply(**call)
        with _patched(envs, "MOE_FUSED_EP_MIN_TOKENS", num_tokens + 1):
            general = moe_apply(**call)
        return fused, general

    def test_the_two_kernels_agree_within_the_wire_quantization(self):
        weights = self.build_weights(jax.random.key(2))
        fused, general = self.run_both_routes(AT_THRESHOLD, weights)

        fused = np.asarray(fused, np.float32)
        general = np.asarray(general, np.float32)
        delta = float(
            np.linalg.norm(fused - general) / np.linalg.norm(general))
        self.assertGreater(delta, WIRE_DELTA_FLOOR)
        self.assertLess(delta, WIRE_DELTA_CEILING)

    def test_a_refused_layer_gets_exactly_the_general_path(self):
        """The fallback is the general path itself, bit for bit, not an
        imitation of it."""
        weights = self.build_weights(jax.random.key(2), with_bias=True)
        x, gating = self.build_activations(jax.random.key(3), ABOVE_THRESHOLD)
        layer = FakeMoELayer()

        through_moe_apply = moe_apply(
            layer=layer,
            x=x,
            gating_output=gating,
            weights=weights,
            moe_backend=MoEBackend.GMM_EP,
            mesh=self.mesh,
            extra_backend_kwargs={"scatter_results": True},
        )
        direct = moe_module.fused_moe_func(
            hidden_states=x,
            w1=weights.w13_weight,
            w2=weights.w2_weight,
            w1_scale=weights.w13_weight_scale,
            w2_scale=weights.w2_weight_scale,
            w1_bias=weights.w13_bias,
            w2_bias=weights.w2_bias,
            gating_output=gating,
            topk=layer.top_k,
            renormalize=layer.renormalize,
            mesh=self.mesh,
            use_ep=layer.use_ep,
            activation="silu",
            scoring_fn=layer.scoring_func,
            all_gather_fp8=False,
            enable_rs_kernel=envs.ENABLE_RS_KERNEL,
            onehot_moe_permute_threshold=envs.ONEHOT_MOE_PERMUTE_THRESHOLD,
            scatter_results=True,
            defer_all_reduce=False,
            hash_based_topk_indices=None,
            expert_score_correction_bias=None,
            moe_chunk_size=0,
            num_valid_tokens=None,
        )
        self.assertArraysEqual(through_moe_apply, direct)


class _patched:
    """Set an attribute for the duration of a block, then put it back."""

    def __init__(self, target, name, value):
        self.target, self.name, self.value = target, name, value

    def __enter__(self):
        self.original = getattr(self.target, self.name)
        setattr(self.target, self.name, self.value)
        return self.value

    def __exit__(self, *exc):
        setattr(self.target, self.name, self.original)
        return False


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
