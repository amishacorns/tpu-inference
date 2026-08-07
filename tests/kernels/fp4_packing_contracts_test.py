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
"""Contract between the four-bit weights on disk and the kernel reading
them: the packing order. Shape and bit arithmetic runs on CPU with the
served generation's hardware answers; cases needing a real unpack on
device skip off TPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.fused_moe.v2 import PACK4
from tpu_inference.layers.common.quantization import quantize_tensor

jax.config.parse_flags_with_absl()

# e2m1 code i is the value at index i; the top eight are the bottom eight
# negated.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5,
               -2.0, -3.0, -4.0, -6.0)


def fp4_from_codes(codes: np.ndarray) -> jax.Array:
    table = np.asarray(E2M1_VALUES, dtype=np.float32)
    return jnp.asarray(table[codes]).astype(jnp.float4_e2m1fn)


def pack_fp4_to_u32(values: jax.Array) -> jax.Array:
    """Pack fp4 [K, N] into uint32 [K // PACK4, N]; offset j along K
    lands in bits [4j, 4j+3], the order the kernel's bitcast undoes."""
    num_k, num_n = values.shape
    if num_k % PACK4:
        raise ValueError(f"{num_k} rows is not a whole number of {PACK4}")
    # [K, N] -> [K / 8, 8, N] -> [K / 8, N, 8], so the eight values that
    # share a word are the minor axis the bitcast folds into one uint32.
    grouped = values.reshape(num_k // PACK4, PACK4, num_n)
    grouped = jnp.swapaxes(grouped, 1, 2)
    return jax.lax.bitcast_convert_type(grouped, jnp.uint32)


def kernel_unpack_fp4(packed: jax.Array, num_k: int) -> jax.Array:
    """The kernel's widening bitcast, uint32 [K / 8, N] to fp4 [K, N]."""

    def body(packed_ref, out_ref):
        out_ref[...] = pltpu.bitcast(packed_ref[...], jnp.float4_e2m1fn)

    out_shape = jax.ShapeDtypeStruct((num_k, packed.shape[1]),
                                     jnp.float4_e2m1fn)
    return pl.pallas_call(body, out_shape=out_shape, interpret=True)(packed)


def kernel_unpack_fp4_on_device(values: jax.Array, num_k: int) -> jax.Array:
    """The same widening bitcast on the backend, off the kernel's own HBM
    ref view, so the order under test is the hardware's not the packer's."""
    num_n = values.shape[1]

    def body(weight_hbm, out_ref, packed_vm, sem):
        packed_hbm = weight_hbm.bitcast(jnp.uint32)
        copy = pltpu.make_async_copy(packed_hbm, packed_vm, sem)
        copy.start()
        copy.wait()
        out_ref[...] = pltpu.bitcast(packed_vm[...],
                                     jnp.float4_e2m1fn).astype(jnp.float32)

    return pl.pallas_call(
        body,
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
        out_shape=jax.ShapeDtypeStruct((num_k, num_n), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((num_k // PACK4, num_n), jnp.uint32),
            pltpu.SemaphoreType.DMA,
        ],
    )(values)


def kernel_unpack_fp4_on_device(values: jax.Array, num_k: int) -> jax.Array:
    """The same widening bitcast on the backend, off the kernel's own HBM
    ref view, so the order under test is the hardware's not the packer's."""
    num_n = values.shape[1]

    def body(weight_hbm, out_ref, packed_vm, sem):
        packed_hbm = weight_hbm.bitcast(jnp.uint32)
        copy = pltpu.make_async_copy(packed_hbm, packed_vm, sem)
        copy.start()
        copy.wait()
        out_ref[...] = pltpu.bitcast(packed_vm[...],
                                     jnp.float4_e2m1fn).astype(jnp.float32)

    return pl.pallas_call(
        body,
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
        out_shape=jax.ShapeDtypeStruct((num_k, num_n), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((num_k // PACK4, num_n), jnp.uint32),
            pltpu.SemaphoreType.DMA,
        ],
    )(values)


class Fp4PackingContractTest(jtu.JaxTestCase):
    """The packing order the checkpoint and the kernel have to agree on."""

    def test_code_table_is_the_one_the_dtype_implements(self):
        codes = np.arange(16, dtype=np.int32).reshape(2, 8)
        values = fp4_from_codes(codes)
        packed = pack_fp4_to_u32(jnp.swapaxes(values, 0, 1))
        expected = np.array(
            [sum(c << (4 * j) for j, c in enumerate(row)) for row in codes],
            dtype=np.uint32)
        self.assertArraysEqual(np.asarray(packed).reshape(-1), expected)

    def test_packer_and_kernel_unpack_round_trip_known_values(self):
        num_k, num_n = 32, 128
        codes = np.random.default_rng(0).integers(0, 16, size=(num_k, num_n))
        values = fp4_from_codes(codes)

        packed = pack_fp4_to_u32(values)
        self.assertEqual(packed.shape, (num_k // PACK4, num_n))
        self.assertEqual(packed.dtype, jnp.uint32)

        recovered = kernel_unpack_fp4(packed, num_k)
        self.assertArraysEqual(recovered.astype(jnp.float32),
                               values.astype(jnp.float32))

    def test_a_disagreeing_order_does_not_round_trip(self):
        num_k, num_n = 32, 128
        codes = np.random.default_rng(1).integers(1, 16, size=(num_k, num_n))
        values = fp4_from_codes(codes)

        reversed_groups = values.reshape(num_k // PACK4, PACK4,
                                         num_n)[:, ::-1, :]
        mispacked = jax.lax.bitcast_convert_type(
            jnp.swapaxes(reversed_groups, 1, 2), jnp.uint32)

        recovered = kernel_unpack_fp4(mispacked, num_k)
        self.assertFalse(
            np.array_equal(np.asarray(recovered.astype(jnp.float32)),
                           np.asarray(values.astype(jnp.float32))))

    def test_the_backend_reads_the_same_nibble_order(self):
        """The nibble order the hardware reads, not the test packer's."""
        if not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+")

        num_k, num_n = 32, 128
        codes = np.random.default_rng(6).integers(0, 16, size=(num_k, num_n))
        values = fp4_from_codes(codes)

        recovered = kernel_unpack_fp4_on_device(values, num_k)
        self.assertArraysEqual(recovered, values.astype(jnp.float32))

    def test_quantizer_output_survives_the_round_trip(self):
        hidden, size_n, block = 64, 128, 32
        weight = jax.random.normal(jax.random.key(2),
                                   (hidden, size_n), jnp.float32) / 4
        quantized, scale = quantize_tensor(jnp.float4_e2m1fn, weight, 0, block)
        self.assertEqual(quantized.dtype, jnp.float4_e2m1fn)
        self.assertEqual(scale.shape, (hidden // block, size_n))

        recovered = kernel_unpack_fp4(pack_fp4_to_u32(quantized), hidden)
        self.assertArraysEqual(recovered.astype(jnp.float32),
                               quantized.astype(jnp.float32))

        dequantized = (recovered.astype(jnp.float32).reshape(
            hidden // block, block, size_n) * scale[:, None, :]).reshape(
                hidden, size_n)
        reference = (quantized.astype(jnp.float32).reshape(
            hidden // block, block, size_n) * scale[:, None, :]).reshape(
                hidden, size_n)
        self.assertArraysEqual(dequantized, reference)



if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
