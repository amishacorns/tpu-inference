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

"""TPU MLA (Multi-Head Latent Attention) backend for TorchAX/vLLM path.

Provides:
  - PallasMlaAttentionBackend: registered for FLASH_ATTN_MLA, is_mla()=True
  - PallasMlaAttentionImpl: stub impl that accepts MLA kwargs during __init__
  - patch_mla_for_tpu(model): patches MLAAttention.forward() to use MLA Pallas
    kernel, bypassing CUDA ops (concat_and_cache_mla, etc.)

MLA weight absorption:
  q_absorbed = q_nope @ W_K  -> (T, N, kv_lora_rank)
  output = attn_out_latent @ W_V^T -> (T, N, v_head_dim)
  KV cache stores compressed latent + RoPE per token (no per-head dim)
"""

import functools
import os
import types
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, PartitionSpec as P
from torchax.interop import jax_view, torch_view
from vllm.v1.attention.backend import (AttentionBackend, AttentionLayer,
                                       AttentionType, MLAAttentionImpl)
from vllm.v1.attention.backends.registry import (AttentionBackendEnum,
                                                 register_backend)

from tpu_inference.kernels.mla.v1.kernel import (
    get_kv_cache_shape as mla_get_kv_cache_shape,
    mla_ragged_paged_attention,
)
from tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes import \
    get_tuned_block_sizes
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Backend + Impl registration
# ---------------------------------------------------------------------------

@register_backend(AttentionBackendEnum.FLASH_ATTN_MLA)
class PallasMlaAttentionBackend(AttentionBackend):
    """TPU MLA attention backend using Pallas kernel."""

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA"

    @staticmethod
    def get_impl_cls() -> type["PallasMlaAttentionImpl"]:
        return PallasMlaAttentionImpl

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # For MLA: num_kv_heads=1, head_size = kv_lora_rank + qk_rope_head_dim
        return mla_get_kv_cache_shape(
            num_blocks, block_size, head_size, jnp.bfloat16)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks not used for TPU MLA backend.")


class PallasMlaAttentionImpl(MLAAttentionImpl):
    """TPU MLA attention implementation (stub).

    Accepts MLA-specific kwargs during __init__ so MLAAttention.__init__()
    succeeds. forward_mha/forward_mqa are NOT used because
    MLAAttention.forward() is patched by patch_mla_for_tpu() to call the
    MLA Pallas kernel directly (bypassing forward_impl and its CUDA ops).
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        # MLA-specific args
        q_lora_rank: int | None = None,
        kv_lora_rank: int = 0,
        qk_nope_head_dim: int = 0,
        qk_rope_head_dim: int = 0,
        qk_head_dim: int = 0,
        v_head_dim: int = 0,
        kv_b_proj=None,
        indexer=None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj
        self.kv_cache_dtype = kv_cache_dtype
        self.dcp_world_size = 1
        self.supports_quant_query_input = False

    def forward_mha(self, q, kv_c_normed, k_pe, kv_c_and_k_pe_cache,
                    attn_metadata, k_scale, output):
        raise NotImplementedError(
            "TPU MLA uses patched forward(). Call patch_mla_for_tpu(model).")

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        raise NotImplementedError(
            "TPU MLA uses patched forward(). Call patch_mla_for_tpu(model).")


# ---------------------------------------------------------------------------
# Forward patching
# ---------------------------------------------------------------------------


def patch_mla_for_tpu(model: torch.nn.Module) -> None:
    """Patch all MLAAttention layers to use the TPU MLA Pallas kernel.

    Call this AFTER model creation and weight sharding to TPU.
    Extracts W_K and W_V from kv_b_proj for weight absorption,
    then overrides forward() to call the MLA Pallas kernel.
    """
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, MLAAttention):
            logger.info("Patching MLAAttention '%s' for TPU MLA", name)
            _setup_tpu_mla(module)
            count += 1

    if count > 0:
        logger.info("Patched %d MLAAttention layers for TPU", count)
    else:
        logger.warning("No MLAAttention layers found to patch")


def _setup_tpu_mla(mla_layer) -> None:
    """Set up weight absorption matrices and override forward."""
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    num_heads = mla_layer.num_heads
    qk_nope_head_dim = mla_layer.qk_nope_head_dim
    qk_rope_head_dim = mla_layer.qk_rope_head_dim
    v_head_dim = mla_layer.v_head_dim
    kv_lora_rank = mla_layer.kv_lora_rank

    # Extract W_K and W_V from kv_b_proj for weight absorption.
    # kv_b_proj.weight: (N * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
    W = mla_layer.kv_b_proj.weight.data
    W = W.reshape(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
    # W_K: (N, P, L) - for q absorption: q_nope @ W_K -> q_absorbed
    mla_layer.register_buffer(
        '_tpu_W_K', W[:, :qk_nope_head_dim, :].contiguous())
    # W_V_T: (N, L, V) - for v up-projection: attn_out @ W_V_T -> output
    mla_layer.register_buffer(
        '_tpu_W_V_T',
        W[:, qk_nope_head_dim:, :].transpose(1, 2).contiguous())

    # Override forward to use TPU MLA kernel
    def _tpu_forward(self, q, kv_c_normed, k_pe, output_shape=None):
        return _tpu_mla_forward(self, q, kv_c_normed, k_pe, output_shape)

    mla_layer.forward = types.MethodType(_tpu_forward, mla_layer)


def _tpu_mla_forward(
    self,                       # MLAAttention instance
    q: torch.Tensor,            # (T, N, qk_head_dim) with RoPE already applied
    kv_c_normed: torch.Tensor,  # (T, kv_lora_rank)
    k_pe: torch.Tensor,         # (T, 1, qk_rope_head_dim) with RoPE applied
    output_shape=None,
) -> torch.Tensor:
    """TPU MLA forward: absorption -> Pallas kernel -> v_up_proj."""
    vllm_ctx = get_vllm_model_wrapper_context()
    mesh = vllm_ctx.mesh

    # Get KV cache from JAX-side wrapper context
    kv_cache_index = vllm_ctx.layer_name_to_kvcache_index[self.layer_name]
    kv_cache = vllm_ctx.kv_caches[kv_cache_index]

    T = q.shape[0]
    P = self.qk_nope_head_dim

    # 1. Weight absorption: q_nope @ W_K -> q_absorbed (T, N, L)
    q_nope = q[:, :, :P]    # (T, N, P)
    q_rope = q[:, :, P:]    # (T, N, rope_dim)
    # bmm: (N, T, P) @ (N, P, L) -> (N, T, L) -> (T, N, L)
    q_absorbed = torch.bmm(
        q_nope.transpose(0, 1), self._tpu_W_K
    ).transpose(0, 1)

    # 2. Convert to JAX and call MLA Pallas kernel
    q_absorbed_jax = jax_view(q_absorbed)
    q_rope_jax = jax_view(q_rope)
    kv_c_jax = jax_view(kv_c_normed)             # (T, L)
    k_pe_jax = jax_view(k_pe.squeeze(1))          # (T, rope_dim)

    # Get attention metadata from forward context (not wrapper context)
    from vllm.forward_context import get_forward_context
    attn_metadata = get_forward_context().attn_metadata

    new_kv_cache, attn_out_latent = _jax_mla_attn_func(
        kv_cache,
        q_absorbed_jax,
        q_rope_jax,
        kv_c_jax,
        k_pe_jax,
        attn_metadata,
        mesh,
        self.scale,
        self.num_heads,
        self.kv_lora_rank,
        self.qk_rope_head_dim,
    )

    # Update KV cache
    vllm_ctx.kv_caches[kv_cache_index] = new_kv_cache

    # 3. V up-projection: attn_out @ W_V_T -> (T, N, V)
    attn_out_torch = torch_view(attn_out_latent)  # (T, N, L)
    # bmm: (N, T, L) @ (N, L, V) -> (N, T, V) -> (T, N, V)
    output = torch.bmm(
        attn_out_torch.transpose(0, 1), self._tpu_W_V_T
    ).transpose(0, 1)

    # Reshape to (T, N*V) for o_proj
    return output.reshape(T, self.num_heads * self.v_head_dim)


@functools.partial(
    jax.jit,
    static_argnames=(
        "mesh",
        "scale",
        "num_heads",
        "kv_lora_rank",
        "qk_rope_head_dim",
    ),
    donate_argnames=("kv_cache",),
)
def _jax_mla_attn_func(
    kv_cache: jax.Array,
    q_absorbed: jax.Array,   # (T, N, kv_lora_rank) - weight-absorbed Q
    q_rope: jax.Array,       # (T, N, qk_rope_head_dim)
    new_kv_c: jax.Array,     # (S, kv_lora_rank) - new compressed KV latent
    new_k_pe: jax.Array,     # (S, qk_rope_head_dim) - new K RoPE
    attn_metadata: AttentionMetadata,
    mesh: Mesh,
    scale: float,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
):
    """JIT-compiled wrapper around the MLA Pallas kernel using shard_map."""
    max_num_tokens = q_absorbed.shape[0]
    max_num_seqs = attn_metadata.seq_lens.shape[0]
    pages_per_seq = attn_metadata.block_tables.shape[0] // max_num_seqs

    # Get tuned block sizes
    bkv_p, bq_sz = get_tuned_block_sizes(
        q_absorbed.dtype, kv_cache.dtype,
        num_heads, 1,  # num_kv_heads=1 for MLA
        kv_lora_rank,
        kv_cache.shape[1],  # page_size
        max_num_tokens, pages_per_seq)
    # get_tuned_block_sizes already clamps to pages_per_seq / max_num_tokens.
    # Additionally, bq_sz must divide num_q_heads_per_shard so the decode
    # branch (static_q_len=1) in the Pallas kernel compiles cleanly.
    # The kernel assertion is: (actual_bq_sz * num_q_heads_per_q_packing) % bq_sz == 0
    # For decode: actual_bq_sz=1, so need num_q_heads_per_q_packing % bq_sz == 0.
    head_axis = ShardingAxisName.MLP_TENSOR
    tp_size = mesh.shape[head_axis]
    num_q_heads_per_shard = num_heads // tp_size  # e.g. 128 / 8 = 16
    # Find largest value <= bq_sz that divides num_q_heads_per_shard
    bq_clamped = bq_sz
    while bq_clamped > 1 and num_q_heads_per_shard % bq_clamped != 0:
        bq_clamped //= 2
    # Also clamp bkvp to avoid VMEM / DMA issues with very large tiles
    # MLA has kv_dim=576 so KV VMEM = 2 * bkvp * page_size * 576 * 2bytes
    # Keeping bkvp reasonable (max 32 pages = 512 tokens per block)
    max_bkvp = int(os.environ.get('MAX_BKVP', '32'))
    num_kv_pages_per_block = min(bkv_p, max_bkvp)
    num_queries_per_block = bq_clamped

    # shard_map specs matching the JAX DeepSeek model:
    # Q tensors: head-sharded on model axis (N dim)
    # KV data: replicated
    # KV cache: sharded on model axis (page dim)
    # Metadata: replicated (ATTN_DATA on data axis, which is size 1)
    head_axis = ShardingAxisName.MLP_TENSOR  # 'model' in 2D mesh
    data_axis = ShardingAxisName.ATTN_DATA   # 'data' in 2D mesh

    in_specs = (
        P(None, head_axis, None),  # q_absorbed: (T, N, L)
        P(None, head_axis, None),  # q_rope: (T, N, R)
        P(None, None),              # new_kv_c: (S, L)
        P(None, None),              # new_k_pe: (S, R)
        P(head_axis),               # kv_cache: (pages, ...)
        P(data_axis),               # seq_lens
        P(data_axis),               # block_tables
        P(data_axis),               # query_start_loc
        P(data_axis),               # request_distribution
    )
    out_specs = (
        P(head_axis),               # new_cache: (pages, ...)
        P(None, head_axis, None),   # attn_out: (T, N, L)
    )

    def _mla_kernel(q, q_pe, kv_c, k_pe, cache, *args):
        out, new_cache = mla_ragged_paged_attention(
            q, q_pe, kv_c, k_pe, cache, *args,
            sm_scale=scale,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
        )
        return new_cache, out

    return jax.shard_map(
        _mla_kernel,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(
        q_absorbed, q_rope, new_kv_c, new_k_pe, kv_cache,
        attn_metadata.seq_lens,
        attn_metadata.block_tables,
        attn_metadata.query_start_loc,
        attn_metadata.request_distribution,
    )
