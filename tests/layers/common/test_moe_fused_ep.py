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
"""Tests for the fused expert-parallel MoE serving adapter: the acceptance
envelope in unsupported_reason and moe_fused_ep_apply, and the layer's
numerics against a plain jax reference."""

import contextlib
import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import test_util as jtu
from jax.sharding import Mesh

from tests.kernels.gmm_fused_test import pinned_tpu
from tpu_inference.kernels.fused_ep_moe.fused_ep_moe_v2 import (
    VMEM_FRACTION, WIRE_RELATIVE_DELTA_BOUND, WIRE_TOKEN_MAX_DELTA_BOUND,
    vmem_limit)
from tpu_inference.layers.common import moe_fused_ep
from tpu_inference.layers.common.moe_fused_ep import (moe_fused_ep_apply,
                                                      unsupported_reason)
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName,
                                                  ShardingAxisNameBase)

# Every test below builds an eight-device serving mesh. A machine that
# offers fewer devices cannot run any of them, so skip the file by name.
if jax.local_device_count() < 8:
    pytest.skip(
        f"this suite builds an eight-device serving mesh and jax offers "
        f"{jax.local_device_count()} device(s) here. Run it with no "
        "accelerator visible (JAX_PLATFORMS=cpu), where the suite conftest "
        "supplies eight CPU devices, or on a host with eight chips.",
        allow_module_level=True)

# The served MoE layer.
EXPERTS = 512
HIDDEN = 4096
INTER = 1024
TOPK = 10
TOKENS = 8192
EP = 8
FP4_QB = 512

# One core's VMEM times the fraction the kernel may use.
V7_VMEM_BUDGET = int(64 * 1024 * 1024 * VMEM_FRACTION)

# A hidden width the kernel's row staging cannot hold (DeepSeek-class).
WIDE_HIDDEN = 7168


@dataclasses.dataclass
class FakeLayer:
    """The attributes the gate reads off a serving MoE layer."""
    top_k: int = TOPK
    renormalize: bool = True
    use_ep: bool = True
    scoring_func: str = "softmax"


@dataclasses.dataclass
class FakeWeights:
    """The attributes the gate reads off a serving MoE weight bundle."""
    w13_weight: Any
    w2_weight: Any
    w13_weight_scale: Any
    w2_weight_scale: Any
    w13_bias: Any = None
    w2_bias: Any = None


# Registered so the bundle traces as one argument; absent biases are None.
jax.tree_util.register_dataclass(
    FakeWeights,
    data_fields=[f.name for f in dataclasses.fields(FakeWeights)],
    meta_fields=[])


def serving_mesh(shape=(1, 8, 1, 1, 1, 1, 1), names=MESH_AXIS_NAMES):
    """A mesh in the serving axis set, over however many devices it needs."""
    count = int(np.prod(shape))
    devices = jax.devices()
    assert len(devices) >= count, (
        f"this suite builds a {count}-device mesh and jax offers "
        f"{len(devices)}. On a host with no accelerator the CPU backend "
        "supplies them, which the suite conftest arranges by adding "
        "--xla_force_host_platform_device_count=8 to XLA_FLAGS before jax "
        "starts. Something removed that or started a backend first.")
    return Mesh(np.array(devices[:count]).reshape(shape), names)


@pytest.fixture(autouse=True)
def base_sharding_axes():
    """Pin the base axis set; the 2D set's ATTN_DATA is a single axis, which
    is not the shape the adapter reconciles."""
    ShardingAxisName.override(ATTN_DATA=ShardingAxisNameBase.ATTN_DATA)
    yield
    ShardingAxisName.reset()


def abstract_weights(experts=EXPERTS,
                     hidden=HIDDEN,
                     inter=INTER,
                     dtype=jnp.float8_e4m3fn,
                     qb=None,
                     w_hidden=None,
                     biased=False):
    """The served weight bundle as shapes and dtypes. qb None is the fp8
    per-channel scale layout, an integer the fp4 block layout at that size."""
    w_hidden = hidden if w_hidden is None else w_hidden
    struct = jax.ShapeDtypeStruct
    blocks1 = 1 if qb is None else w_hidden // qb
    blocks2 = 1 if qb is None else inter // qb
    bias1 = struct((experts, 1, 2 * inter), jnp.float32) if biased else None
    bias2 = struct((experts, 1, hidden), jnp.float32) if biased else None
    return FakeWeights(
        w13_weight=struct((experts, w_hidden, 2 * inter), dtype),
        w2_weight=struct((experts, inter, hidden), dtype),
        w13_weight_scale=struct((experts, blocks1, 1, 2 * inter), jnp.float32),
        w2_weight_scale=struct((experts, blocks2, 1, hidden), jnp.float32),
        w13_bias=bias1,
        w2_bias=bias2,
    )


def gate(mesh,
         weights=None,
         layer=None,
         tokens=TOKENS,
         hidden=HIDDEN,
         activation="silu",
         scatter_results=True,
         extra_backend_kwargs=None,
         pin=True):
    """unsupported_reason on abstract inputs at the served shapes; pin names
    the served chip, pin=False asks on a host that names none."""
    weights = abstract_weights() if weights is None else weights
    layer = FakeLayer() if layer is None else layer
    leaves, treedef = jax.tree.flatten(
        (jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
         jax.ShapeDtypeStruct((tokens, EXPERTS), jnp.float32), weights))
    reason = {}

    def probe(*flat):
        x, gating, w = jax.tree.unflatten(treedef, flat)
        reason["value"] = unsupported_reason(
            layer=layer,
            x=x,
            gating_output=gating,
            weights=w,
            mesh=mesh,
            activation=activation,
            scatter_results=scatter_results,
            extra_backend_kwargs=extra_backend_kwargs)
        return jnp.zeros((1, ), jnp.float32)

    with pinned_tpu() if pin else contextlib.nullcontext():
        jax.eval_shape(probe, *leaves)
    return reason["value"]


# Acceptance envelope


def test_the_served_fp8_configuration_is_accepted():
    assert gate(serving_mesh()) is None


def test_the_served_fp4_block_configuration_is_accepted():
    weights = abstract_weights(dtype=jnp.float4_e2m1fn, qb=FP4_QB)
    assert gate(serving_mesh(), weights=weights) is None


def test_a_hidden_the_row_staging_cannot_hold_is_refused():
    """The staging ceiling: 32 blocks of 128 lanes, so 4096 and no wider."""
    weights = abstract_weights(hidden=WIDE_HIDDEN, w_hidden=WIDE_HIDDEN)
    reason = gate(serving_mesh(), weights=weights, hidden=WIDE_HIDDEN)
    assert reason is not None
    assert "7168" in reason and "row staging holds" in reason


def test_a_hidden_that_is_not_whole_lane_blocks_is_refused():
    odd = 4096 - 64
    weights = abstract_weights(hidden=odd, w_hidden=odd)
    reason = gate(serving_mesh(), weights=weights, hidden=odd)
    assert reason is not None
    assert "128-lane blocks" in reason


def test_sigmoid_scoring_is_refused():
    reason = gate(serving_mesh(), layer=FakeLayer(scoring_func="sigmoid"))
    assert reason is not None
    assert "softmax + top_k" in reason and "sigmoid" in reason


def test_a_non_silu_activation_is_refused():
    reason = gate(serving_mesh(), activation="gelu")
    assert reason is not None
    assert "silu(gate) * up" in reason and "gelu" in reason


def test_the_all_reduced_output_form_is_refused():
    reason = gate(serving_mesh(), scatter_results=False)
    assert reason is not None
    assert "scatter_results" in reason


def test_a_tensor_parallel_layer_is_refused():
    reason = gate(serving_mesh(), layer=FakeLayer(use_ep=False))
    assert reason is not None
    assert "expert-parallel" in reason


@pytest.mark.parametrize("modifier", [
    "hash_based_topk_indices",
    "e_score_correction_bias",
    "num_valid_tokens",
])
def test_a_routing_modifier_the_kernel_cannot_honour_is_refused(modifier):
    """Each of these changes which experts a token goes to."""
    reason = gate(serving_mesh(),
                  extra_backend_kwargs={modifier: jnp.zeros((4, ))})
    assert reason is not None
    assert modifier in reason


def test_routing_modifiers_passed_as_none_do_not_refuse():
    """The caller forwards the whole kwarg set; only a value refuses."""
    assert gate(serving_mesh(),
                extra_backend_kwargs={
                    "hash_based_topk_indices": None,
                    "e_score_correction_bias": None,
                    "num_valid_tokens": None,
                }) is None


def test_approximate_top_k_is_refused(monkeypatch):
    monkeypatch.setenv("MOE_APPROX_TOPK", "1")
    reason = gate(serving_mesh())
    assert reason is not None
    assert "MOE_APPROX_TOPK" in reason


def test_bias_carrying_weights_are_refused():
    reason = gate(serving_mesh(), weights=abstract_weights(biased=True))
    assert reason is not None
    assert "no MoE bias operands" in reason


def test_an_expert_count_the_shards_do_not_divide_is_refused():
    # 516 experts over 8 shards: four experts have nowhere to live.
    weights = abstract_weights(experts=EXPERTS + 4)
    reason = gate(serving_mesh(), weights=weights)
    assert reason is not None
    assert "expert-parallel width" in reason


def test_a_token_count_the_shards_do_not_divide_is_refused():
    reason = gate(serving_mesh(), tokens=TOKENS + 4)
    assert reason is not None
    assert "not divisible by the expert-parallel width" in reason


def test_an_unquantized_weight_dtype_is_refused():
    weights = abstract_weights(dtype=jnp.bfloat16)
    reason = gate(serving_mesh(), weights=weights)
    assert reason is not None
    assert "fp8 e4m3 or fp4 e2m1" in reason


def test_mixed_weight_dtypes_are_refused():
    weights = abstract_weights()
    weights.w2_weight = jax.ShapeDtypeStruct(weights.w2_weight.shape,
                                             jnp.bfloat16)
    reason = gate(serving_mesh(), weights=weights)
    assert reason is not None
    assert "both expert weights must carry one dtype" in reason


def test_an_fp4_block_size_mismatch_between_the_two_matmuls_is_refused():
    weights = abstract_weights(dtype=jnp.float4_e2m1fn, qb=FP4_QB)
    # w2 keeps a different block count, so the two matmuls would need two
    # block sizes.
    weights.w2_weight_scale = jax.ShapeDtypeStruct(
        (EXPERTS, INTER // 256, 1, HIDDEN), jnp.float32)
    reason = gate(serving_mesh(), weights=weights)
    assert reason is not None
    assert "same block size" in reason


def test_a_layer_whose_buffers_do_not_fit_vmem_is_refused():
    """The weight slabs dominate and do not shrink with the batch, so the
    gate must refuse before the build assert fires inside the traced layer."""
    wide = INTER * 4
    weights = abstract_weights(inter=wide)
    reason = gate(serving_mesh(), weights=weights)
    assert reason is not None
    assert "MiB of VMEM" in reason and "budget" in reason


def test_the_gate_answers_with_no_chip_to_read_rather_than_raising():
    """With no chip to read a capacity from, the gate refuses by name."""
    if jtu.test_device_matches(["tpu"]):
        pytest.skip("a chip is attached, so its capacity is readable")
    reason = gate(serving_mesh(), pin=False)
    assert reason is not None
    assert "VMEM budget cannot be read" in reason


def test_a_gating_output_that_is_not_one_array_is_refused():
    reason = unsupported_reason(layer=FakeLayer(),
                                x=jax.ShapeDtypeStruct((TOKENS, HIDDEN),
                                                       jnp.bfloat16),
                                gating_output=(None, None),
                                weights=abstract_weights(),
                                mesh=serving_mesh(),
                                activation="silu",
                                scatter_results=True)
    assert reason is not None
    assert "one logits array" in reason


# Mesh reconciliation


def test_data_parallel_replicas_are_refused():
    """data > 1 permutes shards under the single-axis re-wrap."""
    mesh = serving_mesh(shape=(2, 4, 1, 1, 1, 1, 1))
    reason = gate(mesh)
    assert reason is not None
    assert "mesh data axis size 2 != 1" in reason


def test_a_mesh_whose_axes_are_out_of_order_is_refused():
    names = ("attn_dp", "data", "attn_dp_expert", "expert", "model", "dcp",
             "pcp")
    mesh = serving_mesh(shape=(8, 1, 1, 1, 1, 1, 1), names=names)
    reason = gate(mesh)
    assert reason is not None
    assert "do not contain" in reason


def test_attention_that_is_not_pure_data_parallel_is_refused():
    mesh = serving_mesh(shape=(1, 1, 1, 8, 1, 1, 1))
    reason = gate(mesh)
    assert reason is not None
    assert "attention is not pure DP over all devices" in reason


def test_a_non_degenerate_axis_outside_the_proven_set_is_refused():
    names = MESH_AXIS_NAMES + ("stage", )
    mesh = serving_mesh(shape=(1, 4, 1, 1, 1, 1, 1, 2), names=names)
    reason = gate(mesh)
    assert reason is not None
    assert "'stage'" in reason and "degenerate" in reason


def test_the_single_axis_rewrap_raises_on_a_refused_mesh():
    mesh = serving_mesh(shape=(2, 4, 1, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="fused EP MoE: mesh data axis size"):
        moe_fused_ep._single_axis_mesh(mesh)


def test_the_single_axis_rewrap_keeps_the_device_order():
    mesh = serving_mesh()
    rewrapped = moe_fused_ep._single_axis_mesh(mesh)
    assert rewrapped.axis_names == ("d", )
    assert list(rewrapped.devices.reshape(-1)) == list(
        mesh.devices.reshape(-1))


# The apply entry point re-asks the gate


def test_apply_raises_rather_than_running_a_refused_configuration():
    """A refused configuration must not reach the kernel by another door."""
    mesh = serving_mesh()
    layer = FakeLayer(scoring_func="sigmoid")
    leaves, treedef = jax.tree.flatten((jax.ShapeDtypeStruct(
        (TOKENS, HIDDEN),
        jnp.bfloat16), jax.ShapeDtypeStruct((TOKENS, EXPERTS),
                                            jnp.float32), abstract_weights()))

    def probe(*flat):
        x, gating, weights = jax.tree.unflatten(treedef, flat)
        return moe_fused_ep_apply(layer=layer,
                                  x=x,
                                  gating_output=gating,
                                  weights=weights,
                                  mesh=mesh,
                                  activation="silu",
                                  scatter_results=True)

    with pytest.raises(ValueError,
                       match="fused EP MoE: kernel routing is softmax"):
        jax.eval_shape(probe, *leaves)


# Numerics against a plain jax reference (device)

# A reduced but structurally identical layer: expert count divisible by the
# shard count, hidden whole 128-lane blocks, fp4 block dividing both axes.
DEV_EXPERTS = 16
DEV_HIDDEN = 1024
DEV_INTER = 512
DEV_TOPK = 4
DEV_TOKENS = 1024


def requires_tpu(test):
    """Skip unless a served-generation chip is attached. The device list is
    read inside the test; at module scope it starts the backend too early."""

    @functools.wraps(test)
    def guarded(*args, **kwargs):
        if not jtu.is_device_tpu_at_least(version=7):
            pytest.skip("Expect TPUv7+ (the kernel reads its VMEM budget "
                        "off the device)")
        return test(*args, **kwargs)

    return guarded


def quantize_weight(w, dtype, block=None):
    """Quantize along the contraction axis the way serving does: block None
    is one scale per output channel, an integer one per contraction block."""
    experts, contract, out = w.shape
    span = contract if block is None else block
    peak = float(jnp.finfo(dtype).max)
    blocks = contract // span
    reshaped = w.reshape(experts, blocks, span, out)
    amax = jnp.max(jnp.abs(reshaped), axis=2, keepdims=True)
    scale = jnp.where(amax == 0, 1.0, amax / peak).astype(jnp.float32)
    quantized = (reshaped / scale).astype(dtype).reshape(
        experts, contract, out)
    return quantized, scale.reshape(experts, blocks, 1, out)


def dequantize_weight(q, scale):
    """The f32 weight the kernel's matmuls are meant to be computing with."""
    experts, contract, out = q.shape
    blocks = scale.shape[1]
    span = contract // blocks
    return (q.astype(jnp.float32).reshape(experts, blocks, span, out) *
            scale.reshape(experts, blocks, 1, out)).reshape(
                experts, contract, out)


def ref_moe(x, w13, w13_scale, w2, w2_scale, gating, topk, renormalize):
    """A plain f32 jax MoE over the same dequantized weights, with no fp8
    transport anywhere: the comparison the wire tolerance is stated against."""
    experts, _, inter = w13.shape[0], w13.shape[1], w2.shape[1]
    weights13 = dequantize_weight(w13, w13_scale)
    weights2 = dequantize_weight(w2, w2_scale)
    scores = jax.nn.softmax(gating.astype(jnp.float32), axis=-1)
    tw, ti = jax.lax.top_k(scores, topk)
    if renormalize:
        tw = tw / tw.sum(axis=-1, keepdims=True)
    x32 = x.astype(jnp.float32)
    out = jnp.zeros_like(x32)
    for e in range(experts):
        # A token's gate for this expert is the sum of the slots that
        # chose it, which is zero for a token that did not.
        gate_w = jnp.sum(jnp.where(ti == e, tw, 0.0), axis=-1)[:, None]
        acc1 = x32 @ weights13[e]
        mid = jax.nn.silu(acc1[:, :inter]) * acc1[:, inter:]
        out = out + gate_w * (mid @ weights2[e])
    return out


def device_layer_inputs(seed, dtype, qb=None):
    key = jax.random.key(seed)
    kx, k1, k2, kg = jax.random.split(key, 4)
    x = (jax.random.normal(kx, (DEV_TOKENS, DEV_HIDDEN), jnp.float32) /
         10).astype(jnp.bfloat16)
    w13 = jax.random.normal(k1, (DEV_EXPERTS, DEV_HIDDEN, 2 * DEV_INTER),
                            jnp.float32) / 10
    w2 = jax.random.normal(k2, (DEV_EXPERTS, DEV_INTER, DEV_HIDDEN),
                           jnp.float32) / 10
    q13, s13 = quantize_weight(w13, dtype, qb)
    q2, s2 = quantize_weight(w2, dtype, qb)
    gating = jax.random.normal(kg, (DEV_TOKENS, DEV_EXPERTS), jnp.float32)
    weights = FakeWeights(w13_weight=q13,
                          w2_weight=q2,
                          w13_weight_scale=s13,
                          w2_weight_scale=s2)
    return x, weights, gating


def relative_l2(actual, want):
    a = np.asarray(actual, np.float64)
    w = np.asarray(want, np.float64)
    return float(np.linalg.norm(a - w) / np.linalg.norm(w))


def worst_token_relative_l2(actual, want):
    """The largest per-token relative error. Routing failures are per token,
    and a batch-wide norm divides one bad row by every other row's size."""
    a = np.asarray(actual, np.float64)
    w = np.asarray(want, np.float64)
    per_token = np.linalg.norm(a - w, axis=-1)
    scale = np.linalg.norm(w, axis=-1)
    return float(np.max(per_token / np.where(scale == 0, 1.0, scale)))


@requires_tpu
@pytest.mark.parametrize("dtype,qb,label", [
    (jnp.float8_e4m3fn, None, "fp8"),
    (jnp.float4_e2m1fn, FP4_QB, "fp4_qb512"),
])
def test_layer_output_tracks_a_plain_jax_reference(dtype, qb, label):
    """How far the layer may sit from an unquantized reference. The bound is
    the fp8 wire tolerance; the rotated-expert control makes it meaningful."""
    mesh = serving_mesh()
    layer = FakeLayer(top_k=DEV_TOPK)
    x, weights, gating = device_layer_inputs(0, dtype, qb)
    out = moe_fused_ep_apply(layer=layer,
                             x=x,
                             gating_output=gating,
                             weights=weights,
                             mesh=mesh,
                             activation="silu",
                             scatter_results=True)
    want = ref_moe(x, weights.w13_weight, weights.w13_weight_scale,
                   weights.w2_weight, weights.w2_weight_scale, gating,
                   DEV_TOPK, layer.renormalize)
    error = relative_l2(out, want)
    assert error < WIRE_RELATIVE_DELTA_BOUND, (
        f"{label}: relative L2 {error:.4f} past the wire band "
        f"{WIRE_RELATIVE_DELTA_BOUND}")
    worst = worst_token_relative_l2(out, want)
    assert worst < WIRE_TOKEN_MAX_DELTA_BOUND, (
        f"{label}: worst token's relative error {worst:.4f} past the "
        f"per-token band {WIRE_TOKEN_MAX_DELTA_BOUND}; the batch norm was "
        f"{error:.4f}, so this is a few tokens rather than the whole batch")
    rotated = ref_moe(x, jnp.roll(weights.w13_weight, 1, axis=0),
                      jnp.roll(weights.w13_weight_scale, 1, axis=0),
                      jnp.roll(weights.w2_weight, 1, axis=0),
                      jnp.roll(weights.w2_weight_scale, 1, axis=0), gating,
                      DEV_TOPK, layer.renormalize)
    assert relative_l2(out, rotated) > 10 * error


@requires_tpu
def test_one_corrupted_token_fails_the_per_token_bound():
    """The control for the per-token bound. The quarter-way mix is sized to
    sit in the gap: replacing the token outright the batch norm also sees."""
    mesh = serving_mesh()
    layer = FakeLayer(top_k=DEV_TOPK)
    x, weights, gating = device_layer_inputs(2, jnp.float8_e4m3fn)
    out = moe_fused_ep_apply(layer=layer,
                             x=x,
                             gating_output=gating,
                             weights=weights,
                             mesh=mesh,
                             activation="silu",
                             scatter_results=True)
    want = np.asarray(
        ref_moe(x, weights.w13_weight, weights.w13_weight_scale,
                weights.w2_weight, weights.w2_weight_scale, gating, DEV_TOPK,
                layer.renormalize), np.float64)

    corrupted = np.array(want)
    bad_token = 41
    mix = 0.25
    corrupted[bad_token] = ((1.0 - mix) * want[bad_token] +
                            mix * want[(bad_token + 1) % DEV_TOKENS])

    clean = relative_l2(out, want)
    batch = relative_l2(out, corrupted)
    assert batch < WIRE_RELATIVE_DELTA_BOUND, (
        f"one corrupted token is meant to slip past the batch-wide norm: "
        f"it took a clean {clean:.4f} to {batch:.4f}, and the band is "
        f"{WIRE_RELATIVE_DELTA_BOUND}. If it does not slip past, this "
        "control no longer says anything about the per-token measure")
    worst = worst_token_relative_l2(out, corrupted)
    assert worst > WIRE_RELATIVE_DELTA_BOUND, (
        f"the per-token measure read {worst:.4f} on a token corrupted by "
        f"a quarter, which is inside the band {WIRE_RELATIVE_DELTA_BOUND}; "
        "it is not seeing what the batch norm cannot")


@requires_tpu
def test_a_nan_row_leaves_every_other_token_bitwise_unchanged():
    """Token locality: one inf logit gives the router a NaN row, and every
    token but that one must come back with identical bits."""
    mesh = serving_mesh()
    layer = FakeLayer(top_k=DEV_TOPK)
    x, weights, gating = device_layer_inputs(1, jnp.float8_e4m3fn)
    nan_token = 137
    poisoned = gating.at[nan_token, 3].set(jnp.inf)
    assert bool(jnp.isnan(jax.nn.softmax(poisoned, axis=-1)[nan_token]).all())

    def run(g):
        return moe_fused_ep_apply(layer=layer,
                                  x=x,
                                  gating_output=g,
                                  weights=weights,
                                  mesh=mesh,
                                  activation="silu",
                                  scatter_results=True)

    clean = np.asarray(run(gating).astype(jnp.float32))
    dirty = np.asarray(run(poisoned).astype(jnp.float32))
    keep = [t for t in range(DEV_TOKENS) if t != nan_token]
    np.testing.assert_array_equal(dirty[keep], clean[keep])
    assert np.isfinite(dirty[nan_token]).all()


@requires_tpu
def test_the_pinned_vmem_budget_matches_the_device():
    """The envelope answers from V7_VMEM_BUDGET; hold it to the truth."""
    assert vmem_limit() == V7_VMEM_BUDGET
