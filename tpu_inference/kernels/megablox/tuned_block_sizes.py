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

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# Key:
#   - m: int, total number of tokens/rows
#   - k: int, input feature dimension
#   - n: int, output feature dimension per group
#   - num_total_groups: int, total experts in the model
#   - num_current_groups: int, experts assigned to this TPU shard
#   - lhs_dtype: str, data type name of the LHS matrix
#   - rhs_dtype: str, data type name of the RHS (weights) matrix
#   - quant_block_size: int, granularity of quantization scales
# Value:
#   - tm: int, m-dimension tile size
#   - tk: int, k-dimension tile size
#   - tn: int, n-dimension tile size
TUNED_BLOCK_SIZES = {
    (128, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (128, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 12,
    ),
    (128, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 24,
    ),
    (128, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 24,
    ),
    (256, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (256, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 5,
        256 * 24,
    ),
    (256, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 20,
    ),
    (256, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 24,
    ),
    (512, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 12,
    ),
    (512, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 12,
    ),
    (512, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 5,
    ),
    (512, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 6,
    ),
    (1024, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (1024, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 8,
    ),
    (1024, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 20,
    ),
    (1024, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 24,
    ),
    (2048, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (2048, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 12,
    ),
    (2048, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 20,
    ),
    (2048, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 8,
    ),
    (4096, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (4096, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 12,
    ),
    (4096, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 5,
    ),
    (4096, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 8,
    ),
    (8192, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (8192, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        256 * 1,
        256 * 10,
        256 * 12,
    ),
    (8192, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 5,
    ),
    (8192, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 8,
    ),
    (16384, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (16384, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        256 * 1,
        256 * 10,
        256 * 12,
    ),
    (16384, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        256 * 1,
        256 * 24,
        256 * 5,
    ),
    (16384, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        256 * 1,
        256 * 24,
        256 * 6,
    ),
    (32768, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (32768, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        256 * 1,
        256 * 10,
        256 * 12,
    ),
    (32768, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        256 * 1,
        256 * 24,
        256 * 5,
    ),
    (32768, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 8,
    ),
    (65536, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (65536, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        256 * 1,
        256 * 10,
        256 * 12,
    ),
    (65536, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        256 * 1,
        256 * 24,
        256 * 5,
    ),
    (65536, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        256 * 1,
        256 * 24,
        256 * 6,
    ),
    # ─── DeepSeek-R1 on TPU v7 (256 experts, EP=8 → 32 local, BF16) ───
    # GMM1: lhs=[M, 7168] @ rhs=[32, 7168, 4096]  (hidden → gate+up)
    # Swept with benchmark_gmm_tile_sweep.py on tpu7x-8, vmem=64MB.
    (512, 7168, 4096, 256, 32, 'bfloat16', 'bfloat16', 7168): (
        128, 7168, 2048,
    ),
    (1024, 7168, 4096, 256, 32, 'bfloat16', 'bfloat16', 7168): (
        128, 1792, 4096,
    ),
    (2048, 7168, 4096, 256, 32, 'bfloat16', 'bfloat16', 7168): (
        128, 1792, 4096,
    ),
    (4096, 7168, 4096, 256, 32, 'bfloat16', 'bfloat16', 7168): (
        256, 7168, 1024,
    ),
    (8192, 7168, 4096, 256, 32, 'bfloat16', 'bfloat16', 7168): (
        512, 7168, 1024,
    ),
    (16384, 7168, 4096, 256, 32, 'bfloat16', 'bfloat16', 7168): (
        512, 7168, 1024,
    ),
    (32768, 7168, 4096, 256, 32, 'bfloat16', 'bfloat16', 7168): (
        512, 7168, 1024,
    ),
    (65536, 7168, 4096, 256, 32, 'bfloat16', 'bfloat16', 7168): (
        512, 7168, 1024,
    ),
    (131072, 7168, 4096, 256, 32, 'bfloat16', 'bfloat16', 7168): (
        512, 7168, 1024,
    ),
    # GMM2: lhs=[M, 2048] @ rhs=[32, 2048, 7168]  (intermediate → hidden)
    (512, 2048, 7168, 256, 32, 'bfloat16', 'bfloat16', 2048): (
        128, 2048, 3584,
    ),
    (1024, 2048, 7168, 256, 32, 'bfloat16', 'bfloat16', 2048): (
        128, 2048, 3584,
    ),
    (2048, 2048, 7168, 256, 32, 'bfloat16', 'bfloat16', 2048): (
        128, 1024, 7168,
    ),
    (4096, 2048, 7168, 256, 32, 'bfloat16', 'bfloat16', 2048): (
        256, 2048, 3584,
    ),
    (8192, 2048, 7168, 256, 32, 'bfloat16', 'bfloat16', 2048): (
        512, 2048, 3584,
    ),
    (16384, 2048, 7168, 256, 32, 'bfloat16', 'bfloat16', 2048): (
        512, 2048, 3584,
    ),
    (32768, 2048, 7168, 256, 32, 'bfloat16', 'bfloat16', 2048): (
        512, 2048, 3584,
    ),
    (65536, 2048, 7168, 256, 32, 'bfloat16', 'bfloat16', 2048): (
        512, 2048, 3584,
    ),
    (131072, 2048, 7168, 256, 32, 'bfloat16', 'bfloat16', 2048): (
        512, 2048, 3584,
    ),
    # ─── DeepSeek-R1 on TPU v7 (256 experts, EP=8 → 32 local, FP8 weights) ───
    # GMM1: lhs=[M, 7168] bf16 @ rhs=[32, 7168, 4096] fp8  (hidden → gate+up)
    # Swept with benchmark_gmm_tile_sweep.py --rhs-dtype fp8, vmem=64MB.
    (512, 7168, 4096, 256, 32, 'bfloat16', 'float8_e4m3fn', 7168): (
        128, 7168, 4096,
    ),
    (1024, 7168, 4096, 256, 32, 'bfloat16', 'float8_e4m3fn', 7168): (
        128, 7168, 2048,
    ),
    (2048, 7168, 4096, 256, 32, 'bfloat16', 'float8_e4m3fn', 7168): (
        128, 7168, 2048,
    ),
    (4096, 7168, 4096, 256, 32, 'bfloat16', 'float8_e4m3fn', 7168): (
        256, 7168, 2048,
    ),
    (8192, 7168, 4096, 256, 32, 'bfloat16', 'float8_e4m3fn', 7168): (
        512, 7168, 1024,
    ),
    (16384, 7168, 4096, 256, 32, 'bfloat16', 'float8_e4m3fn', 7168): (
        512, 7168, 2048,
    ),
    (32768, 7168, 4096, 256, 32, 'bfloat16', 'float8_e4m3fn', 7168): (
        512, 7168, 2048,
    ),
    (65536, 7168, 4096, 256, 32, 'bfloat16', 'float8_e4m3fn', 7168): (
        512, 7168, 2048,
    ),
    (131072, 7168, 4096, 256, 32, 'bfloat16', 'float8_e4m3fn', 7168): (
        512, 7168, 2048,
    ),
    # GMM2: lhs=[M, 2048] bf16 @ rhs=[32, 2048, 7168] fp8  (intermediate → hidden)
    (512, 2048, 7168, 256, 32, 'bfloat16', 'float8_e4m3fn', 2048): (
        128, 2048, 7168,
    ),
    (1024, 2048, 7168, 256, 32, 'bfloat16', 'float8_e4m3fn', 2048): (
        128, 2048, 7168,
    ),
    (2048, 2048, 7168, 256, 32, 'bfloat16', 'float8_e4m3fn', 2048): (
        128, 2048, 7168,
    ),
    (4096, 2048, 7168, 256, 32, 'bfloat16', 'float8_e4m3fn', 2048): (
        256, 2048, 7168,
    ),
    (8192, 2048, 7168, 256, 32, 'bfloat16', 'float8_e4m3fn', 2048): (
        512, 2048, 3584,
    ),
    (16384, 2048, 7168, 256, 32, 'bfloat16', 'float8_e4m3fn', 2048): (
        512, 2048, 3584,
    ),
    (32768, 2048, 7168, 256, 32, 'bfloat16', 'float8_e4m3fn', 2048): (
        512, 2048, 7168,
    ),
    (65536, 2048, 7168, 256, 32, 'bfloat16', 'float8_e4m3fn', 2048): (
        512, 2048, 7168,
    ),
    (131072, 2048, 7168, 256, 32, 'bfloat16', 'float8_e4m3fn', 2048): (
        512, 2048, 7168,
    ),
    # ─── DeepSeek-R1 on TPU v7 (256 experts, EP=8 → 32 local, FP4 weights) ───
    # GMM1: lhs=[M, 7168] bf16 @ rhs=[32, 7168, 4096] fp4  (hidden → gate+up)
    # Swept with benchmark_gmm_tile_sweep.py --rhs-dtype fp4, vmem=64MB.
    (512, 7168, 4096, 256, 32, 'bfloat16', 'float4_e2m1fn', 7168): (
        128, 7168, 4096,
    ),
    (1024, 7168, 4096, 256, 32, 'bfloat16', 'float4_e2m1fn', 7168): (
        128, 7168, 4096,
    ),
    (2048, 7168, 4096, 256, 32, 'bfloat16', 'float4_e2m1fn', 7168): (
        128, 7168, 2048,
    ),
    (4096, 7168, 4096, 256, 32, 'bfloat16', 'float4_e2m1fn', 7168): (
        256, 7168, 4096,
    ),
    (8192, 7168, 4096, 256, 32, 'bfloat16', 'float4_e2m1fn', 7168): (
        512, 7168, 2048,
    ),
    (16384, 7168, 4096, 256, 32, 'bfloat16', 'float4_e2m1fn', 7168): (
        512, 7168, 4096,
    ),
    (32768, 7168, 4096, 256, 32, 'bfloat16', 'float4_e2m1fn', 7168): (
        512, 7168, 4096,
    ),
    (65536, 7168, 4096, 256, 32, 'bfloat16', 'float4_e2m1fn', 7168): (
        512, 7168, 4096,
    ),
    (131072, 7168, 4096, 256, 32, 'bfloat16', 'float4_e2m1fn', 7168): (
        512, 7168, 4096,
    ),
    # GMM2: lhs=[M, 2048] bf16 @ rhs=[32, 2048, 7168] fp4  (intermediate → hidden)
    (512, 2048, 7168, 256, 32, 'bfloat16', 'float4_e2m1fn', 2048): (
        128, 2048, 7168,
    ),
    (1024, 2048, 7168, 256, 32, 'bfloat16', 'float4_e2m1fn', 2048): (
        128, 2048, 7168,
    ),
    (2048, 2048, 7168, 256, 32, 'bfloat16', 'float4_e2m1fn', 2048): (
        128, 2048, 7168,
    ),
    (4096, 2048, 7168, 256, 32, 'bfloat16', 'float4_e2m1fn', 2048): (
        256, 2048, 7168,
    ),
    (8192, 2048, 7168, 256, 32, 'bfloat16', 'float4_e2m1fn', 2048): (
        512, 2048, 7168,
    ),
    (16384, 2048, 7168, 256, 32, 'bfloat16', 'float4_e2m1fn', 2048): (
        512, 2048, 7168,
    ),
    (32768, 2048, 7168, 256, 32, 'bfloat16', 'float4_e2m1fn', 2048): (
        512, 2048, 3584,
    ),
    (65536, 2048, 7168, 256, 32, 'bfloat16', 'float4_e2m1fn', 2048): (
        512, 2048, 7168,
    ),
    (131072, 2048, 7168, 256, 32, 'bfloat16', 'float4_e2m1fn', 2048): (
        512, 2048, 7168,
    ),
    # ─── DeepSeek-R1 on TPU v7 (256 experts, EP=8 → 32 local, FP8×FP8) ───
    # GMM1: lhs=[M, 7168] fp8 @ rhs=[32, 7168, 4096] fp8  (hidden → gate+up)
    # Swept with benchmark_gmm_tile_sweep.py --lhs-dtype fp8 --rhs-dtype fp8, vmem=64MB.
    # Note: (512, 7168, 4096) OOMs at M≥8192 with fp8 LHS — LHS tile still 512×7168=3.5MB.
    (512, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 7168): (
        128, 7168, 2048,
    ),
    (1024, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 7168): (
        128, 7168, 2048,
    ),
    (2048, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 7168): (
        128, 7168, 2048,
    ),
    (4096, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 7168): (
        256, 7168, 4096,
    ),
    (8192, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 7168): (
        512, 7168, 2048,
    ),
    (16384, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 7168): (
        512, 7168, 2048,
    ),
    (32768, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 7168): (
        512, 7168, 2048,
    ),
    (65536, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 7168): (
        512, 7168, 2048,
    ),
    (131072, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 7168): (
        512, 7168, 2048,
    ),
    # GMM2: lhs=[M, 2048] fp8 @ rhs=[32, 2048, 7168] fp8  (intermediate → hidden)
    (512, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 2048): (
        128, 2048, 7168,
    ),
    (1024, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 2048): (
        128, 2048, 7168,
    ),
    (2048, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 2048): (
        128, 2048, 7168,
    ),
    (4096, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 2048): (
        256, 2048, 7168,
    ),
    (8192, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 2048): (
        512, 2048, 7168,
    ),
    (16384, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 2048): (
        512, 2048, 7168,
    ),
    (32768, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 2048): (
        512, 2048, 7168,
    ),
    (65536, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 2048): (
        512, 2048, 7168,
    ),
    (131072, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float8_e4m3fn', 2048): (
        512, 2048, 7168,
    ),
    # ─── DeepSeek-R1 on TPU v7 (256 experts, EP=8 → 32 local, FP8×FP4) ───
    # GMM1: lhs=[M, 7168] fp8 @ rhs=[32, 7168, 4096] fp4  (hidden → gate+up)
    # Swept with benchmark_gmm_tile_sweep.py --lhs-dtype fp8 --rhs-dtype fp4, vmem=64MB.
    (512, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 7168): (
        128, 7168, 4096,
    ),
    (1024, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 7168): (
        128, 7168, 4096,
    ),
    (2048, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 7168): (
        128, 7168, 4096,
    ),
    (4096, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 7168): (
        256, 7168, 4096,
    ),
    (8192, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 7168): (
        512, 7168, 4096,
    ),
    (16384, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 7168): (
        512, 7168, 4096,
    ),
    (32768, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 7168): (
        512, 7168, 4096,
    ),
    (65536, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 7168): (
        512, 7168, 4096,
    ),
    (131072, 7168, 4096, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 7168): (
        512, 7168, 4096,
    ),
    # GMM2: lhs=[M, 2048] fp8 @ rhs=[32, 2048, 7168] fp4  (intermediate → hidden)
    (512, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 2048): (
        128, 2048, 7168,
    ),
    (1024, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 2048): (
        128, 2048, 7168,
    ),
    (2048, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 2048): (
        128, 2048, 7168,
    ),
    (4096, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 2048): (
        256, 2048, 7168,
    ),
    (8192, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 2048): (
        512, 2048, 7168,
    ),
    (16384, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 2048): (
        512, 2048, 7168,
    ),
    (32768, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 2048): (
        512, 2048, 7168,
    ),
    (65536, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 2048): (
        512, 2048, 7168,
    ),
    (131072, 2048, 7168, 256, 32, 'float8_e4m3fn', 'float4_e2m1fn', 2048): (
        512, 2048, 7168,
    ),
}


def get_default_gmm_block_sizes(m: int, k: int, n: int):
    """
    Heuristic-based defaults for GMM tiling. 
    """
    # TODO (Qiliang Cui): when update to v2, use the v2 default tiling.
    del k, n  # Currently not using input dimensions for heuristics
    return (min(m, 128), 128, 128)


def get_tuned_block_sizes(
    m: int,
    k: int,
    n: int,
    num_total_groups: int,
    num_current_groups: int,
    lhs_dtype: str,
    rhs_dtype: str,
    quant_block_size: int,
):
    """
    Retrieves optimized (TM, TK, TN) tiling parameters for the GMM kernel.
    """
    # GMM inputs must align to tile sizes; however, tile sizes themselves
    # are often powers of 2 or mxu multiples.
    key = (
        m,
        k,
        n,
        num_total_groups,
        num_current_groups,
        str(lhs_dtype),
        str(rhs_dtype),
        quant_block_size,
    )

    if key not in TUNED_BLOCK_SIZES:
        default_val = get_default_gmm_block_sizes(m, k, n)
        logger.warning_once(
            f'[GMM kernel] using default block sizes for key: {key}: {default_val}'
        )
        return default_val

    return TUNED_BLOCK_SIZES.get(key)
