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
"""Contracts between the quantized weights on disk and the kernels reading
them: weight-scale block layouts and bfloat16 accumulator drift. Shape and
bit arithmetic runs on CPU with the served generation's hardware answers;
cases needing a real matmul skip off TPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental import pallas as pl

from tpu_inference.kernels.megablox.gmm_fused import gmm_fused
from tpu_inference.kernels.megablox.gmm_v2_fused_support import \
    LHS_QUANT_BLOCK_SIZE

jax.config.parse_flags_with_absl()


def postscale_matmul(lhs_q, lhs_scale, rhs_q, rhs_scale, acc_dtype):
    """One quantized matmul the way the fused kernel's k loop runs it: steps
    by LHS_QUANT_BLOCK_SIZE, reads rhs scale block start_k // rhs_block."""
    size_m, size_k = lhs_q.shape
    size_n = rhs_q.shape[1]
    num_rhs_blocks = rhs_scale.shape[0]
    rhs_block = size_k // num_rhs_blocks

    def body(lhs_ref, lhs_scale_ref, rhs_ref, rhs_scale_ref, out_ref):
        lhs_f32 = lhs_ref[...].astype(jnp.float32)
        rhs_f32 = rhs_ref[...].astype(jnp.float32)
        acc = jnp.zeros((size_m, size_n), acc_dtype)
        for start_k in range(0, size_k, LHS_QUANT_BLOCK_SIZE):
            window = slice(start_k, start_k + LHS_QUANT_BLOCK_SIZE)
            product = jax.lax.dot_general(lhs_f32[:, window],
                                          rhs_f32[window, :],
                                          (((1, ), (0, )), ((), ())),
                                          preferred_element_type=jnp.float32)
            lhs_block_id = start_k // LHS_QUANT_BLOCK_SIZE
            product = product * lhs_scale_ref[:, lhs_block_id][:, None]
            product = product * rhs_scale_ref[start_k // rhs_block][None, :]
            acc = (acc.astype(jnp.float32) + product).astype(acc_dtype)
        out_ref[...] = acc

    out_shape = jax.ShapeDtypeStruct((size_m, size_n), acc_dtype)
    return pl.pallas_call(body, out_shape=out_shape,
                          interpret=True)(lhs_q, lhs_scale, rhs_q, rhs_scale)


def quantize_rows(x, num_blocks):
    """Per-row, per-block fp8 quantization, the lhs form the kernel uses."""
    size_m, size_k = x.shape
    blocked = x.astype(jnp.float32).reshape(size_m, num_blocks,
                                            size_k // num_blocks)
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    amax = jnp.max(jnp.abs(blocked), axis=-1, keepdims=True)
    scale = amax / fp8_max
    inverse = jnp.where(scale == 0.0, 0.0,
                        1.0 / jnp.where(scale == 0.0, 1.0, scale))
    quantized = jnp.clip(blocked * inverse, -fp8_max,
                         fp8_max).astype(jnp.float8_e4m3fn)
    return (quantized.reshape(size_m, size_k),
            scale.reshape(size_m, num_blocks).astype(jnp.float32))


def quantize_columns(w, num_blocks):
    """Per-column fp8 quantization over num_blocks contraction blocks."""
    size_k, size_n = w.shape
    blocked = w.astype(jnp.float32).reshape(num_blocks, size_k // num_blocks,
                                            size_n)
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    amax = jnp.max(jnp.abs(blocked), axis=1, keepdims=True)
    scale = amax / fp8_max
    inverse = jnp.where(scale == 0.0, 0.0,
                        1.0 / jnp.where(scale == 0.0, 1.0, scale))
    quantized = jnp.clip(blocked * inverse, -fp8_max,
                         fp8_max).astype(jnp.float8_e4m3fn)
    return (quantized.reshape(size_k, size_n),
            scale.reshape(num_blocks, size_n).astype(jnp.float32))


class ScaleLayoutContractTest(jtu.JaxTestCase):
    """That both rhs scale layouts reconstruct the unquantized product."""

    @parameterized.named_parameters(("channelwise", 1), ("block_512", 4))
    def test_postscale_dequantization_matches_a_full_precision_reference(
            self, rhs_blocks):
        """Both scale layouts reconstruct the unquantized product."""
        size_m, size_k, size_n = 8, 2048, 128
        lhs_key, rhs_key = jax.random.split(jax.random.key(3))
        lhs = jax.random.normal(lhs_key, (size_m, size_k), jnp.float32) / 8
        rhs = jax.random.normal(rhs_key, (size_k, size_n), jnp.float32) / 8

        lhs_q, lhs_scale = quantize_rows(lhs, size_k // LHS_QUANT_BLOCK_SIZE)
        rhs_q, rhs_scale = quantize_columns(rhs, rhs_blocks)

        actual = postscale_matmul(lhs_q, lhs_scale, rhs_q, rhs_scale,
                                  jnp.float32)

        lhs_deq = (lhs_q.astype(jnp.float32).reshape(
            size_m, size_k // LHS_QUANT_BLOCK_SIZE, LHS_QUANT_BLOCK_SIZE) *
                   lhs_scale[:, :, None]).reshape(size_m, size_k)
        rhs_deq = (rhs_q.astype(jnp.float32).reshape(
            rhs_blocks, size_k // rhs_blocks, size_n) *
                   rhs_scale[:, None, :]).reshape(size_k, size_n)
        reference = lhs_deq @ rhs_deq

        self.assertAllClose(actual, reference, atol=2e-3, rtol=2e-3)


class AccumulationDivergenceTest(jtu.JaxTestCase):
    """How far the quantized path's bfloat16 accumulator drifts."""

    # The emulation lands between 0.00383 and 0.00390 relative Frobenius norm
    # at hidden 4096 across seeds; this bracket is set just outside that.
    MIN_DIVERGENCE = 3e-3
    MAX_DIVERGENCE = 5e-3

    # The kernel's own drift on the device is a different measurement and
    # is not characterized across seeds, so it keeps a wide bracket.
    MIN_DEVICE_DIVERGENCE = 1e-4
    MAX_DEVICE_DIVERGENCE = 2e-2

    def divergence(self, actual, reference):
        actual = np.asarray(actual, np.float32)
        reference = np.asarray(reference, np.float32)
        return float(
            np.linalg.norm(actual - reference) / np.linalg.norm(reference))

    def test_bfloat16_accumulation_diverges_and_stays_bounded(self):
        """The accumulator dtype, isolated from everything else."""
        size_m, size_k, size_n = 64, 4096, 256
        lhs_key, rhs_key = jax.random.split(jax.random.key(4))
        lhs = jax.random.normal(lhs_key, (size_m, size_k), jnp.float32) / 10
        rhs = jax.random.normal(rhs_key, (size_k, size_n), jnp.float32) / 10

        lhs_q, lhs_scale = quantize_rows(lhs, size_k // LHS_QUANT_BLOCK_SIZE)
        rhs_q, rhs_scale = quantize_columns(rhs, 1)

        in_bf16 = postscale_matmul(lhs_q, lhs_scale, rhs_q, rhs_scale,
                                   jnp.bfloat16)
        in_f32 = postscale_matmul(lhs_q, lhs_scale, rhs_q, rhs_scale,
                                  jnp.float32)

        drift = self.divergence(in_bf16.astype(jnp.float32), in_f32)
        self.assertGreater(drift, self.MIN_DIVERGENCE)
        self.assertLess(drift, self.MAX_DIVERGENCE)

    def test_fused_kernel_carries_the_same_divergence(self):
        """The same measurement on the kernel itself, which needs a TPU."""
        if not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+")

        groups, rows, hidden, inter = 2, 64, 4096, 512
        lhs_key, w1_key, w2_key = jax.random.split(jax.random.key(5), 3)
        lhs = (jax.random.normal(lhs_key, (rows, hidden), jnp.float32) /
               10).astype(jnp.bfloat16)
        w1 = jax.random.normal(w1_key,
                               (groups, hidden, 2 * inter), jnp.float32) / 10
        w2 = jax.random.normal(w2_key,
                               (groups, inter, hidden), jnp.float32) / 10

        fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)

        def channelwise(w):
            amax = jnp.max(jnp.abs(w), axis=1, keepdims=True)
            scale = amax / fp8_max
            quantized = jnp.clip(w / jnp.where(scale == 0, 1, scale), -fp8_max,
                                 fp8_max).astype(jnp.float8_e4m3fn)
            return quantized, scale

        w1_q, w1_scale = channelwise(w1)
        w2_q, w2_scale = channelwise(w2)
        w1_scale = w1_scale.reshape(groups, 1, 1, 2 * inter)
        w2_scale = w2_scale.reshape(groups, 1, 1, hidden)
        group_sizes = jnp.array([rows // 2, rows - rows // 2], jnp.int32)

        served = gmm_fused(lhs, w1_q, w2_q, group_sizes, w1_scale, w2_scale)
        in_f32 = gmm_fused(lhs,
                           w1_q,
                           w2_q,
                           group_sizes,
                           w1_scale,
                           w2_scale,
                           acc_dtype=jnp.float32)

        drift = self.divergence(served.astype(jnp.float32),
                                in_f32.astype(jnp.float32))
        self.assertGreater(drift, self.MIN_DEVICE_DIVERGENCE)
        self.assertLess(drift, self.MAX_DEVICE_DIVERGENCE)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
