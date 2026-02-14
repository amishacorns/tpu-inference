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
"""Auto-tuned block sizes for Fused MoE kernel."""

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


# TODO(jevinjiang): create sharable util
def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


# Key:
#   - hidden_size: int, padded hidden dimension size
#   - intermediate_size: int, padded intermediate dimension size
#   - num_experts: int, total number of experts
#   - top_k: int, selected number of top experts
#   - t_packing: int, token packing
#   - w_packing: int, weight packing
#   - num_tokens: int, total number of tokens
#   - ep_size: int, expert parallelism size
#   - subc_quant_w1_sz: int, subchannel quantization size for w1 (0 = no subchannel)
# Value:
#   - bt: int, block size of local_num_tokens
#   - bf: int, block size of intermediate_size
#   - bd1: int, block size of hidden_size in w1
#   - bd2: int, block size of hidden_size in w2
#   - btc: int, compute size of block tokens for active expert
#   - bfc: int, compute size of block intermediate_size
#   - bd1c: int, compute size of block hidden_size in w1
#   - bd2c: int, compute size of block hidden_size in w2
TUNED_BLOCK_SIZES = {
    (3072, 3072, 128, 4, 2, 2, 128, 4, 0): (
        32,
        256 * 6,
        256 * 12,
        256 * 12,
        16,
        256 * 6,
        256 * 12,
        256 * 12,
    ),
    (3072, 3072, 128, 4, 2, 2, 128, 8, 0): (
        16,
        256 * 6,
        256 * 6,
        256 * 12,
        16,
        256 * 6,
        256 * 6,
        256 * 6,
    ),
    (3072, 3072, 128, 4, 2, 2, 256, 4, 0): (
        64,
        256 * 6,
        256 * 6,
        256 * 12,
        16,
        256 * 6,
        256 * 6,
        256 * 12,
    ),
    (3072, 3072, 128, 4, 2, 2, 256, 8, 0): (
        32,
        256 * 6,
        256 * 6,
        256 * 12,
        32,
        256 * 6,
        256 * 6,
        256 * 6,
    ),
    (3072, 3072, 128, 4, 2, 2, 512, 4, 0): (
        128,
        256 * 6,
        256 * 6,
        256 * 6,
        64,
        256 * 6,
        256 * 6,
        256 * 6,
    ),
    (3072, 3072, 128, 4, 2, 2, 512, 8, 0): (
        64,
        256 * 6,
        256 * 6,
        256 * 12,
        64,
        256 * 6,
        256 * 6,
        256 * 6,
    ),
    (3072, 3072, 128, 4, 2, 2, 1024, 4, 0): (
        128,
        256 * 3,
        256 * 6,
        256 * 6,
        64,
        256 * 3,
        256 * 6,
        256 * 6,
    ),
    (3072, 3072, 128, 4, 2, 2, 1024, 8, 0): (
        64,
        256 * 6,
        256 * 6,
        256 * 12,
        64,
        256 * 6,
        256 * 6,
        256 * 6,
    ),
    (3072, 3072, 128, 4, 2, 2, 2048, 4, 0): (
        128,
        256 * 3,
        256 * 6,
        256 * 6,
        64,
        256 * 3,
        256 * 6,
        256 * 6,
    ),
    (3072, 3072, 128, 4, 2, 2, 2048, 8, 0): (
        64,
        256 * 6,
        256 * 6,
        256 * 12,
        64,
        256 * 6,
        256 * 6,
        256 * 6,
    ),
    (6144, 2560, 160, 8, 2, 4, 16, 4, 0): (
        256 * 1,
        256 * 5,
        256 * 12,
        256 * 24,
        16,
        256 * 5,
        256 * 12,
        256 * 24,
    ),
    (6144, 2560, 160, 8, 2, 4, 16, 8, 0): (
        16,
        256 * 10,
        256 * 8,
        256 * 12,
        16,
        256 * 2,
        256 * 8,
        256 * 4,
    ),
    (6144, 2560, 160, 8, 2, 4, 32, 4, 0): (
        128,
        256 * 5,
        256 * 12,
        256 * 24,
        32,
        256 * 5,
        256 * 12,
        256 * 24,
    ),
    (6144, 2560, 160, 8, 2, 4, 32, 8, 0): (
        64,
        256 * 10,
        256 * 4,
        256 * 6,
        32,
        256 * 10,
        256 * 4,
        256 * 2,
    ),
    (6144, 2560, 160, 8, 2, 4, 64, 4, 0): (
        64,
        256 * 10,
        256 * 12,
        256 * 12,
        32,
        256 * 10,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 64, 8, 0): (
        32,
        256 * 10,
        256 * 6,
        256 * 12,
        32,
        256 * 5,
        256 * 6,
        256 * 3,
    ),
    (6144, 2560, 160, 8, 2, 4, 128, 4, 0): (
        64,
        256 * 5,
        256 * 24,
        256 * 24,
        16,
        256 * 5,
        256 * 24,
        256 * 24,
    ),
    (6144, 2560, 160, 8, 2, 4, 128, 8, 0): (
        16,
        256 * 5,
        256 * 12,
        256 * 24,
        16,
        256 * 1,
        256 * 12,
        256 * 8,
    ),
    (6144, 2560, 160, 8, 2, 4, 256, 4, 0): (
        256 * 1,
        256 * 10,
        256 * 8,
        256 * 8,
        32,
        256 * 10,
        256 * 8,
        256 * 8,
    ),
    (6144, 2560, 160, 8, 2, 4, 256, 8, 0): (
        32,
        256 * 10,
        256 * 8,
        256 * 12,
        32,
        256 * 10,
        256 * 8,
        256 * 4,
    ),
    (6144, 2560, 160, 8, 2, 4, 512, 4, 0): (
        256 * 1,
        256 * 2,
        256 * 12,
        256 * 12,
        64,
        256 * 2,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 512, 8, 0): (
        64,
        256 * 2,
        256 * 24,
        256 * 24,
        32,
        256 * 1,
        256 * 24,
        256 * 8,
    ),
    (6144, 2560, 160, 8, 2, 4, 1024, 4, 0): (
        128,
        256 * 2,
        256 * 12,
        256 * 12,
        64,
        256 * 2,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 1024, 8, 0): (
        64,
        256 * 5,
        256 * 8,
        256 * 12,
        32,
        256 * 5,
        256 * 8,
        256 * 4,
    ),
    (6144, 2560, 160, 8, 2, 4, 2048, 4, 0): (
        128,
        256 * 2,
        256 * 12,
        256 * 12,
        64,
        256 * 2,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 2048, 8, 0): (
        64,
        256 * 5,
        256 * 8,
        256 * 12,
        32,
        256 * 5,
        256 * 8,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 4096, 4, 0): (
        128,
        256 * 2,
        256 * 12,
        256 * 12,
        64,
        256 * 2,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 4096, 8, 0): (
        64,
        256 * 5,
        256 * 8,
        256 * 12,
        32,
        256 * 5,
        256 * 8,
        256 * 6,
    ),
    (6144, 2560, 160, 8, 2, 4, 8192, 4, 0): (
        128,
        256 * 2,
        256 * 12,
        256 * 12,
        32,
        256 * 2,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 8192, 8, 0): (
        64,
        256 * 5,
        256 * 8,
        256 * 12,
        32,
        256 * 5,
        256 * 8,
        256 * 6,
    ),
    # DeepSeek-R1: hidden=7168, intermediate=2048, 256 experts, topk=8, EP=8
    # FP8 weights (w_packing=4), BF16 tokens (t_packing=2), TPU v7
    (7168, 2048, 256, 8, 2, 4, 16, 8, 0): (
        2,
        256 * 8,
        256 * 7,
        256 * 14,
        2,
        256 * 8,
        256 * 7,
        256 * 14,
    ),
    (7168, 2048, 256, 8, 2, 4, 32, 8, 0): (
        4,
        256 * 8,
        256 * 14,
        256 * 28,
        4,
        256 * 8,
        256 * 14,
        256 * 28,
    ),
    (7168, 2048, 256, 8, 2, 4, 64, 8, 0): (
        8,
        256 * 8,
        256 * 7,
        256 * 28,
        8,
        256 * 8,
        256 * 7,
        256 * 28,
    ),
    (7168, 2048, 256, 8, 2, 4, 128, 8, 0): (
        4,
        256 * 8,
        256 * 7,
        256 * 28,
        4,
        256 * 8,
        256 * 7,
        256 * 28,
    ),
    (7168, 2048, 256, 8, 2, 4, 256, 8, 0): (
        32,
        256 * 8,
        256 * 4,
        256 * 28,
        32,
        256 * 8,
        256 * 4,
        256 * 28,
    ),
    (7168, 2048, 256, 8, 2, 4, 512, 8, 0): (
        32,
        256 * 8,
        256 * 4,
        256 * 28,
        32,
        256 * 8,
        256 * 4,
        256 * 28,
    ),
    (7168, 2048, 256, 8, 2, 4, 1024, 8, 0): (
        32,
        256 * 8,
        256 * 4,
        256 * 28,
        32,
        256 * 8,
        256 * 4,
        256 * 28,
    ),
    (7168, 2048, 256, 8, 2, 4, 2048, 8, 0): (
        32,
        256 * 8,
        256 * 4,
        256 * 28,
        32,
        256 * 8,
        256 * 4,
        256 * 28,
    ),
    (7168, 2048, 256, 8, 2, 4, 4096, 8, 0): (
        32,
        256 * 8,
        256 * 4,
        256 * 28,
        32,
        256 * 8,
        256 * 4,
        256 * 28,
    ),
    (7168, 2048, 256, 8, 2, 4, 8192, 8, 0): (
        32,
        256 * 8,
        256 * 4,
        256 * 28,
        32,
        256 * 8,
        256 * 4,
        256 * 28,
    ),
    (7168, 2048, 256, 8, 2, 4, 16384, 8, 0): (
        32,
        256 * 8,
        256 * 4,
        256 * 28,
        32,
        256 * 8,
        256 * 4,
        256 * 28,
    ),
    # DeepSeek-R1: hidden=7168, intermediate=2048, 256 experts, topk=8, EP=8
    # FP4 weights (w_packing=8), BF16 tokens (t_packing=2), TPU v7
    (7168, 2048, 256, 8, 2, 8, 16, 8, 0): (
        2,
        256 * 8,
        256 * 14,
        256 * 14,
        2,
        256 * 8,
        256 * 14,
        256 * 14,
    ),
    (7168, 2048, 256, 8, 2, 8, 32, 8, 0): (
        2,
        256 * 8,
        256 * 14,
        256 * 28,
        2,
        256 * 8,
        256 * 14,
        256 * 28,
    ),
    (7168, 2048, 256, 8, 2, 8, 64, 8, 0): (
        4,
        256 * 8,
        256 * 14,
        256 * 28,
        4,
        256 * 8,
        256 * 14,
        256 * 28,
    ),
    (7168, 2048, 256, 8, 2, 8, 128, 8, 0): (
        16,
        256 * 8,
        256 * 14,
        256 * 14,
        16,
        256 * 8,
        256 * 14,
        256 * 14,
    ),
    (7168, 2048, 256, 8, 2, 8, 256, 8, 0): (
        32,
        256 * 8,
        256 * 7,
        256 * 14,
        32,
        256 * 8,
        256 * 7,
        256 * 14,
    ),
    (7168, 2048, 256, 8, 2, 8, 512, 8, 0): (
        32,
        256 * 8,
        256 * 7,
        256 * 14,
        32,
        256 * 8,
        256 * 7,
        256 * 14,
    ),
    (7168, 2048, 256, 8, 2, 8, 1024, 8, 0): (
        32,
        256 * 8,
        256 * 14,
        256 * 7,
        32,
        256 * 8,
        256 * 14,
        256 * 7,
    ),
    (7168, 2048, 256, 8, 2, 8, 2048, 8, 0): (
        32,
        256 * 8,
        256 * 7,
        256 * 14,
        32,
        256 * 8,
        256 * 7,
        256 * 14,
    ),
    (7168, 2048, 256, 8, 2, 8, 4096, 8, 0): (
        32,
        256 * 8,
        256 * 7,
        256 * 14,
        32,
        256 * 8,
        256 * 7,
        256 * 14,
    ),
    (7168, 2048, 256, 8, 2, 8, 8192, 8, 0): (
        32,
        256 * 8,
        256 * 7,
        256 * 14,
        32,
        256 * 8,
        256 * 7,
        256 * 14,
    ),
    (7168, 2048, 256, 8, 2, 8, 16384, 8, 0): (
        32,
        256 * 4,
        256 * 14,
        256 * 28,
        32,
        256 * 4,
        256 * 14,
        256 * 28,
    ),
}


def get_default_block_sizes(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    t_packing: int,
    w_packing: int,
    num_tokens: int,
    ep_size: int,
    subc_quant_w1_sz: int = 0,
):
    assert num_experts % ep_size == 0
    assert num_tokens % ep_size == 0
    assert 0 < top_k <= num_experts
    assert 0 < t_packing <= 8
    assert 0 < w_packing <= 8
    if hidden_size % 256:
        hidden_size = align_to(hidden_size, 256)
    if intermediate_size % 256:
        intermediate_size = align_to(intermediate_size, 256)
    # TODO(jevinjiang): the formula is only applied to tpu-v7 and we need to add
    # other formulas for other generations.
    local_num_tokens = num_tokens // ep_size

    d = hidden_size // 256
    f = intermediate_size // 256

    # Find largest valid bd that divides hidden_size and is a multiple of 256.
    def _largest_divisor(dim, cap):
        best = 256
        for m in range(1, dim // 256 + 1):
            v = 256 * m
            if dim % v == 0 and v <= cap:
                best = v
        return best

    bf = _largest_divisor(intermediate_size, min(256 * f // 2, 1024))
    bd1 = _largest_divisor(hidden_size, min(256 * d // 2, 1024))
    bd2 = _largest_divisor(hidden_size, min(256 * d // 2, 2048))

    bt = min(local_num_tokens, 128)
    btc = min(local_num_tokens // 2, 64)

    return (bt, bf, bd1, bd2, btc, bf, bd1, bd2)


def get_tuned_block_sizes(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    t_packing: int,
    w_packing: int,
    num_tokens: int,
    ep_size: int,
    subc_quant_w1_sz: int = 0,
):
    assert num_experts % ep_size == 0
    assert num_tokens % ep_size == 0
    assert 0 < top_k <= num_experts
    assert 0 < t_packing <= 8
    assert 0 < w_packing <= 8
    # TODO(b/467431118): currently we just use manual tuned block sizes for some
    # specific models.
    hidden_size = align_to(hidden_size, 256)
    intermediate_size = align_to(intermediate_size, 256)

    key = (
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        t_packing,
        w_packing,
        num_tokens,
        ep_size,
        subc_quant_w1_sz,
    )

    if key not in TUNED_BLOCK_SIZES:
        logger.warning_once(
            f'[Fused MOE kernel] using default block sizes for key: {key}: {get_default_block_sizes(*key)}'
        )
    return TUNED_BLOCK_SIZES.get(key, get_default_block_sizes(*key))
