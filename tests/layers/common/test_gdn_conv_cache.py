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
"""The layout a rank-3 GDN conv-state cache is allocated in."""

import re
import unittest

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental.layout import Layout

from tpu_inference.kernels.gdn.v3 import wrapper

# The allocator needs a recent vllm; the wrapper boundary below does not.
try:
    from tpu_inference.runner.kv_cache_manager import _conv_state_layout
except ImportError as error:  # pragma: no cover - environment dependent
    _conv_state_layout = None
    ALLOCATOR_IMPORT_ERROR = error
else:
    ALLOCATOR_IMPORT_ERROR = None

ALLOCATOR_LOGGER = "vllm.tpu_inference.runner.kv_cache_manager"

CACHE_SHAPE = (17, 3, 512)


@unittest.skipIf(
    _conv_state_layout is None, f"kv_cache_manager is not importable here: "
    f"{ALLOCATOR_IMPORT_ERROR}")
@jtu.with_config(jax_numpy_dtype_promotion="standard")
class ConvStateLayoutTest(jtu.JaxTestCase):

    def test_bfloat16_cache_gets_the_served_layout(self):
        layout = _conv_state_layout(CACHE_SHAPE, jnp.bfloat16)
        self.assertEqual(layout.major_to_minor, (0, 1, 2))
        self.assertEqual(layout.tiling, ((4, 128), (2, 1)))

    @parameterized.named_parameters(
        ("bfloat16", jnp.bfloat16, ((4, 128), (2, 1))),
        ("float16", jnp.float16, ((4, 128), (2, 1))),
        ("float32", jnp.float32, ((8, 128), )),
        ("int8", jnp.int8, ((2, 128), (4, 1))),
        ("float8_e4m3fn", jnp.float8_e4m3fn, ((2, 128), (4, 1))),
    )
    def test_tiling_follows_the_element_width(self, dtype, tiling):
        layout = _conv_state_layout(CACHE_SHAPE, dtype)
        self.assertIsInstance(layout, Layout)
        self.assertEqual(layout.major_to_minor, (0, 1, 2))
        self.assertEqual(layout.tiling, tiling)

    def test_derivation_is_the_same_answer_every_time(self):
        first = _conv_state_layout(CACHE_SHAPE, jnp.bfloat16)
        second = _conv_state_layout(CACHE_SHAPE, jnp.bfloat16)
        self.assertEqual(first, second)

    @parameterized.named_parameters(
        ("rank_two", (17, 512)),
        ("rank_four", (17, 3, 512, 1)),
        ("rank_one", (512, )),
    )
    def test_a_cache_that_is_not_rank_three_falls_back_by_name(self, shape):
        """The pin is an optimization, so an odd cache still allocates."""
        with self.assertLogs(ALLOCATOR_LOGGER, level="WARNING") as logs:
            self.assertIsNone(_conv_state_layout(shape, jnp.bfloat16))
        self.assertIn("is not the rank-3", "\n".join(logs.output))

    def test_the_fallback_names_the_shape_it_was_given(self):
        with self.assertLogs(ALLOCATOR_LOGGER, level="WARNING") as logs:
            self.assertIsNone(_conv_state_layout((17, 512), jnp.bfloat16))
        self.assertIn("(17, 512)", "\n".join(logs.output))

    @parameterized.named_parameters(
        ("float64", jnp.float64),
        ("complex64", jnp.complex64),
    )
    def test_an_element_width_with_no_defined_layout_falls_back_by_name(
            self, dtype):
        with self.assertLogs(ALLOCATOR_LOGGER, level="WARNING") as logs:
            self.assertIsNone(_conv_state_layout(CACHE_SHAPE, dtype))
        self.assertIn("bytes per element", "\n".join(logs.output))

    def test_the_channel_count_does_not_change_the_layout(self):
        """Holds for this minor-dimension order only; a transposed conv
        state needs its own descriptor."""
        for shape in ((17, 3, 512), (129, 3, 4096), (5, 1, 128)):
            layout = _conv_state_layout(shape, jnp.bfloat16)
            self.assertEqual(layout.tiling, ((4, 128), (2, 1)))


# Two key/query projections and one value projection make up DIM.
N_KQ = N_V = 1
D_K = D_V = 128
KERNEL_SIZE = 4
DIM = N_KQ * D_K * 2 + N_V * D_V
BATCH = 8
NUM_SEQS = 4
NUM_SLOTS = NUM_SEQS + 1
NATIVE_CACHE_SHAPE = (NUM_SLOTS, KERNEL_SIZE - 1, DIM)


def trace_native_wrapper(conv_state_shape):
    sds = jax.ShapeDtypeStruct
    operands = dict(
        qkv=sds((BATCH, DIM), jnp.bfloat16),
        b=sds((BATCH, N_V), jnp.bfloat16),
        a=sds((BATCH, N_V), jnp.bfloat16),
        conv_state=sds(conv_state_shape, jnp.bfloat16),
        recurrent_state=sds((NUM_SLOTS, N_V, D_K, D_V), jnp.float32),
        conv_weight=sds((DIM, 1, KERNEL_SIZE), jnp.bfloat16),
        conv_bias=sds((DIM, ), jnp.bfloat16),
        a_log=sds((N_V, ), jnp.float32),
        dt_bias=sds((N_V, ), jnp.float32),
        query_start_loc=sds((NUM_SEQS + 1, ), jnp.int32),
        state_indices=sds((NUM_SEQS, ), jnp.int32),
        distribution=sds((3, ), jnp.int32),
        seq_lens=sds((NUM_SEQS, ), jnp.int32),
    )
    return jax.eval_shape(
        lambda **kw: wrapper.fused_conv1d_gdn(**kw,
                                              n_kq=N_KQ,
                                              n_v=N_V,
                                              d_k=D_K,
                                              d_v=D_V,
                                              kernel_size=KERNEL_SIZE,
                                              conv_cache_native=True),
        **operands)


class NativeConvStateBoundaryTest(jtu.JaxTestCase):
    """What the wrapper does with a cache the kernel cannot read."""

    @parameterized.named_parameters(
        ("rank_two", (NUM_SLOTS, DIM)),
        ("rank_four", (NUM_SLOTS, KERNEL_SIZE - 1, DIM, 1)),
        ("whole_window_instead_of_the_carry_over",
         (NUM_SLOTS, KERNEL_SIZE, DIM)),
        ("another_models_channel_count",
         (NUM_SLOTS, KERNEL_SIZE - 1, 2 * DIM)),
    )
    def test_a_cache_the_kernel_cannot_read_is_refused_by_name(self, shape):
        with self.assertRaisesRegex(ValueError, "is not the cache-native"):
            trace_native_wrapper(shape)

    def test_the_refusal_names_the_shape_it_was_given(self):
        shape = (NUM_SLOTS, KERNEL_SIZE, DIM)
        with self.assertRaisesRegex(ValueError, re.escape(str(shape))):
            trace_native_wrapper(shape)

    def test_the_refusal_names_the_shape_it_wanted(self):
        with self.assertRaisesRegex(ValueError,
                                    rf"\[slots, {KERNEL_SIZE - 1}, {DIM}\]"):
            trace_native_wrapper((NUM_SLOTS, DIM))

    def test_the_served_cache_shape_gets_past_the_boundary(self):
        try:
            trace_native_wrapper(NATIVE_CACHE_SHAPE)
        except ValueError as error:
            self.assertNotIn("cache-native", str(error))


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
