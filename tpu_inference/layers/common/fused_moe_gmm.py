# Copyright 2025 Google LLC
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

import functools
from typing import Literal

import jax
from jax import numpy as jnp
from jax._src import dtypes as jax_dtypes
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.megablox.common import tpu_generation
from tpu_inference.kernels.megablox.gmm import GroupMetadata, gmm, make_group_metadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.utils import get_mesh_shape_product


def apply_scoring_fn(scoring_fn: str, x: jax.Array) -> jax.Array:
    match scoring_fn:
        case "softmax":
            return jax.nn.softmax(x, axis=-1)
        case "sigmoid":
            return jax.nn.sigmoid(x)
        case _:
            raise NotImplementedError(
                f"FusedMoE does not support {scoring_fn} scoring function")


def apply_act_fn(activation: str, x1: jax.Array, x2: jax.Array) -> jax.Array:
    match activation:
        case "silu":
            return jax.nn.silu(x1) * x2
        case "swigluoai":
            return _swigluoai(x1, x2)
        case _:
            raise NotImplementedError(
                f"FusedMoE does not support {activation} activation function")


def _swigluoai(x1: jax.Array,
               x2: jax.Array,
               alpha=1.702,
               limit=7.0) -> jax.Array:
    x1 = jnp.clip(x1, a_max=limit)
    x2 = jnp.clip(x2, a_min=-limit, a_max=limit)

    gated_activation = x1 * jax.nn.sigmoid(alpha * x1)

    return gated_activation * (x2 + 1)


# =============================================================================
# VMEM-budget-aware tiling for the GMM kernel
# =============================================================================

# VMEM capacity per TPU generation (bytes). These are the total VMEM available
# to a single kernel on one chip. Source: quantized_matmul/tuned_block_sizes.py
_DEVICE_VMEM_BYTES = {
    6: 96 * 1024 * 1024,  # TPU v6: 96 MB
    7: 48 * 1024 * 1024,  # TPU v7: 48 MB
}

# Fallback for unknown TPU generations — use the largest known VMEM (v6).
_DEFAULT_VMEM_BYTES = 96 * 1024 * 1024  # 96 MB

# Safety margin: our VMEM model (_estimate_gmm_vmem) tracks all major tile
# buffers (lhs, rhs, output, scratch, scale, bias) but omits small compiler
# overhead: store mask temporaries, scalar prefetch buffers, vreg spills,
# and instruction buffers. 0.85 leaves 15% headroom (7.2 MB on v7).
_VMEM_SAFETY_FACTOR = 0.85

def _dtype_bytes(dtype: jnp.dtype) -> float:
    """Bytes per element, correctly handling sub-byte dtypes like FP4."""
    if hasattr(jax_dtypes, "bit_width"):
        return jax_dtypes.bit_width(dtype) / 8
    return jax_dtypes.itemsize_bits(dtype) / 8


def _estimate_gmm_vmem(
    tm: int, tk: int, tn: int,
    lhs_dtype: jnp.dtype,
    rhs_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    scale_dtype: jnp.dtype | None = None,
    quant_block_size: int = 0,
    bias_dtype: jnp.dtype | None = None,
) -> int:
    """Estimate GMM kernel VMEM usage in bytes.

    The GMM Pallas kernel allocates these buffers in VMEM:
      - LHS tile:      (tm, tk) in lhs_dtype        — double-buffered (2x)
      - RHS tile:       (tk, tn) in rhs_dtype        — double-buffered (2x)
      - Output tile:    (tm, tn) in out_dtype        — double-buffered (2x)
      - Scratch accum:  (tm, tn) in f32              — NOT double-buffered (1x)
      - Scale tile:     (num_qblocks_per_tk, 1, tn)  — double-buffered (2x)
      - Bias tile:      (1, tn)                      — double-buffered (2x)

    Double-buffering is applied by the Mosaic compiler for tiles that change
    across the sequential ("arbitrary") grid dimensions, allowing overlapping
    of compute and memory transfers.
    """
    lhs_bytes = tm * tk * _dtype_bytes(lhs_dtype)
    rhs_bytes = tk * tn * _dtype_bytes(rhs_dtype)
    out_bytes = tm * tn * _dtype_bytes(out_dtype)
    scratch_bytes = tm * tn * 4  # f32 accumulator, always 4 bytes/elem

    scale_bytes = 0.0
    if scale_dtype is not None and quant_block_size > 0:
        num_qblocks_per_tk = -(-tk // quant_block_size)  # ceil
        scale_bytes = num_qblocks_per_tk * 1 * tn * _dtype_bytes(scale_dtype)

    bias_bytes = 0.0
    if bias_dtype is not None:
        bias_bytes = 1 * tn * _dtype_bytes(bias_dtype)

    # I/O tiles are double-buffered (2x), scratch is not
    return int(2 * (lhs_bytes + rhs_bytes + out_bytes + scale_bytes + bias_bytes)
               + scratch_bytes)


def _get_vmem_budget() -> int:
    """Get the usable VMEM budget for GMM tiling on the current TPU."""
    try:
        gen = tpu_generation()
        device_vmem = _DEVICE_VMEM_BYTES.get(gen, _DEFAULT_VMEM_BYTES)
    except Exception:
        device_vmem = _DEFAULT_VMEM_BYTES
    return int(device_vmem * _VMEM_SAFETY_FACTOR)


def _divisors_at_least(n: int, min_val: int = 128) -> list[int]:
    """All divisors of n that are >= min_val, sorted descending."""
    divs = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            if i >= min_val:
                divs.add(i)
            if n // i >= min_val:
                divs.add(n // i)
    return sorted(divs, reverse=True)


# TODO (jacobplatin): make this more generic
def round_up_to_multiple_of_128_within_limit(x: int, limit: int) -> int:
    """
    Rounds the given integer `x` up to the nearest multiple of 128, without
    exceeding the specified `limit`.

    If `x` is less than or equal to 128, returns 128.
    If `x` is less than `limit`, returns the smallest multiple of 128 greater
    than or equal to `x`.
    If `x` is greater than or equal to `limit`, searches for the largest
    multiple of 128 less than or equal to `limit` (down to 512) that divides `x`
    evenly, and returns it.
    If no such candidate is found, returns `limit`.

    Args:
        x (int): The integer to round up.
        limit (int): The upper bound (must be a multiple of 128).

    Returns:
        int: The rounded value according to the rules above.

    Raises:
        AssertionError: If `limit` is less than 128 or not a multiple of 128.
    """
    assert limit >= 128 and limit % 128 == 0
    if x <= 128:
        return 128
    if x < limit:
        return (x + 127) // 128 * 128
    for candidate in range(limit, 511, -128):
        if x % candidate == 0:
            return candidate
    return limit


@functools.lru_cache(maxsize=256)
def _get_tiling_size_for_gmm_kernel(
    m: int, k: int, n: int, g: int,
    lhs_dtype: jnp.dtype = jnp.bfloat16,
    rhs_dtype: jnp.dtype = jnp.bfloat16,
    out_dtype: jnp.dtype = jnp.bfloat16,
    scale_dtype: jnp.dtype | None = None,
    quant_block_size: int = 0,
    bias_dtype: jnp.dtype | None = None,
) -> tuple[int, int, int, int]:
    """Calculate optimal GMM tiling within the device's VMEM budget.

    Finds (tm, tk, tn) that minimizes total grid iterations while fitting
    within VMEM. This replaces the old approach of hardcoded tile limits
    that were dtype-unaware and left most of VMEM unused.

    Strategy:
      - tm is determined by the MoE workload shape (tokens per expert).
      - tk candidates are 128-aligned divisors of k (required for both
        quantization block alignment and the Pallas block shape rule that
        the last LHS dimension must be divisible by 128 or equal to k).
      - tn candidates are multiples of 128 (Pallas block shape rule for
        the last RHS/output dimension) plus n itself (always valid since
        it equals the full array dimension).
      - The (tk, tn) pair that minimizes an effective tile score while
        fitting the VMEM budget is selected. The score is the raw tile
        count (ceil(k/tk) * ceil(n/tn)) inflated by MXU waste for
        inexact tn values, so 4 exact tiles beats 3 inexact tiles with
        >9% waste in the last tile.
        Tiebreakers prefer exact divisors of n, then larger tk.

    Args:
        m: Total number of expanded tokens (num_tokens * topk).
        k: Input feature dimension.
        n: Output feature dimension.
        g: Number of experts.
        lhs_dtype: Dtype of activations (LHS).
        rhs_dtype: Dtype of weights (RHS). Sub-byte types like FP4 allow
            larger tiles in the same VMEM.
        out_dtype: Dtype of the output tensor.
        scale_dtype: Dtype of rhs_scale (e.g. bf16 for FP8), or None.
        quant_block_size: Number of k elements per quantization block, or 0.
        bias_dtype: Dtype of rhs_bias, or None.

    Returns:
        (tm, tk, tn, vmem_limit_bytes) where vmem_limit_bytes should be
        passed to the kernel's CompilerParams.
    """
    vmem_budget = _get_vmem_budget()

    # --- tm: driven by tokens-per-expert workload shape ---
    tm = _compute_tm(m, g)

    # --- tk candidates: divisors of k >= 128, divisible by 128 ---
    # When rhs_scale is None, quant_block_size = k and the GMM kernel
    # requires k % tk == 0 (otherwise it overrides tk = k). Using exact
    # divisors satisfies this for all cases.
    # Pallas requires the last dimension of the LHS block (tk) to be
    # divisible by 128 (or equal to k). LHS is 2D (M, K), so K is the
    # last dimension.
    # Additionally, the GMM kernel enforces quant block alignment:
    #   if tk % quant_block_size != 0 and quant_block_size % tk != 0:
    #       tk = quant_block_size   (silent override!)
    # We filter out any tk that would be overridden to avoid a mismatch
    # between our VMEM estimate and the kernel's actual tile size.
    def _quant_aligned(tk_val: int) -> bool:
        if quant_block_size == 0:
            return True
        return tk_val % quant_block_size == 0 or quant_block_size % tk_val == 0

    tk_cands = [d for d in _divisors_at_least(k, 128)
                if d % 128 == 0 and _quant_aligned(d)]
    if not tk_cands:
        tk_cands = [k]  # k itself always satisfies k % tk == 0

    # --- tn candidates: multiples of 128 (+ n itself) ---
    # Pallas requires the last block dimension to be divisible by 128
    # OR equal to the full array dimension n. Multiples of 128 satisfy
    # the first condition; we always include n for the second.
    tn_set = set(range(128, n + 1, 128))
    tn_set.add(n)  # always valid: equals array dimension (Pallas rule)
    tn_cands = sorted(tn_set, reverse=True)

    # --- Search for best (tk, tn) within VMEM budget ---
    # Minimize total_tiles = ceil(k/tk) * ceil(n/tn).
    # Tiebreakers (in order):
    #   1. Prefer tn that divides n exactly (no wasted MXU ops in last tile)
    #   2. Prefer larger tk (fewer sequential k-loop iterations)
    #   3. Prefer smaller tn (less VMEM pressure, more compiler headroom)
    lb = _dtype_bytes(lhs_dtype)
    rb = _dtype_bytes(rhs_dtype)
    ob = _dtype_bytes(out_dtype)
    sb = _dtype_bytes(scale_dtype) if scale_dtype is not None else 0.0
    bb = _dtype_bytes(bias_dtype) if bias_dtype is not None else 0.0

    best_key = (float('inf'), True, 0, float('inf'))
    best_tk = tk_cands[-1]  # smallest fallback
    best_tn = 128

    for tk_c in tk_cands:
        tiles_k = -(-k // tk_c)  # ceil(k / tk_c)

        # Analytical upper bound on tn for this tk (avoids scanning all tn)
        # vmem = 2*(lhs + rhs + out + scale + bias) + scratch
        # where lhs = tm*tk*lb, rhs = tk*tn*rb, out = tm*tn*ob,
        #       scale = nqb*tn*sb, bias = tn*bb, scratch = tm*tn*4
        # Grouping: fixed = 2*tm*tk*lb, per_tn = 2*(tk*rb + tm*ob + nqb*sb + bb) + tm*4
        nqb = -(-tk_c // quant_block_size) if quant_block_size > 0 else 0
        lhs_cost = 2 * tm * tk_c * lb
        remaining = vmem_budget - lhs_cost
        if remaining <= 0:
            continue
        tn_cost_per_unit = (2 * (tk_c * rb + tm * ob + nqb * sb + bb)
                           + tm * 4)
        max_tn = remaining / tn_cost_per_unit if tn_cost_per_unit > 0 else n

        # For this tk, find the best tn. We need to check multiple tn values
        # because the first fitting tn (largest) may not divide n exactly,
        # while a smaller tn with the same tile count might.
        best_tiles_n_for_tk = float('inf')

        for tn_c in tn_cands:
            if tn_c > max_tn:
                continue  # won't fit, try next smaller tn

            tiles_n = -(-n // tn_c)

            # Allow up to +1 tile beyond best seen: an exact tiling with
            # one extra tile often beats an inexact tiling (no MXU waste).
            # Beyond +1 the waste penalty can't compensate for 2+ extra tiles.
            if tiles_n > best_tiles_n_for_tk + 1:
                break
            if tiles_n < best_tiles_n_for_tk:
                best_tiles_n_for_tk = tiles_n

            # Verify with full VMEM estimate
            vmem = _estimate_gmm_vmem(tm, tk_c, tn_c, lhs_dtype,
                                      rhs_dtype, out_dtype,
                                      scale_dtype, quant_block_size,
                                      bias_dtype)
            if vmem > vmem_budget:
                continue

            is_exact = (n % tn_c == 0)
            # Penalise inexact tilings: when tn doesn't divide n, the
            # last n-tile wastes MXU on padding columns AND the compiler
            # generates less efficient code for the non-uniform tile.
            # Empirically, 3 inexact tiles are ~12% slower than 4 exact
            # tiles (DeepSeek M=65536).  Treat N inexact tiles as
            # equivalent to N+1 tiles so exact tilings with 1 more tile
            # win via the is_exact tiebreaker.
            if is_exact:
                score = float(tiles_k * tiles_n)
            else:
                score = float(tiles_k * (tiles_n + 1))
            key = (score, not is_exact, -tk_c, tn_c)

            if key < best_key:
                best_key = key
                best_tk = tk_c
                best_tn = tn_c

        # Early exit: can't do better than 1 total tile
        if best_key[0] == 1:
            break

    # Return the full VMEM budget (not the tight estimate) as the limit
    # passed to the compiler. The compiler may need slightly more VMEM
    # than our model predicts (alignment, instruction buffers, etc.),
    # so passing the tight estimate can cause OOM by just a few KB.
    return tm, best_tk, best_tn, int(vmem_budget)


def _compute_tm(m: int, g: int) -> int:
    """Compute the m-dimension tile size (depends only on m and g).

    This is factored out so that moe_gmm_local can compute group metadata
    once and share it between GMM1 and GMM2 (which have the same m and g
    but different k and n).
    """
    tm = round_up_to_multiple_of_128_within_limit(2 * m // g, 512)
    return min(tm, m)  # kernel requires m % tm == 0


def gmm_wrapper(lhs, rhs, rhs_scale, rhs_bias, group_sizes, group_offset,
                group_metadata=None, num_active_tiles=None):
    m, g, k, n = lhs.shape[0], *rhs.shape

    # Extract scale/bias metadata for VMEM estimation (all hashable for cache)
    scale_dtype = rhs_scale.dtype if rhs_scale is not None else None
    quant_block_size = (k // rhs_scale.shape[1]) if rhs_scale is not None else 0
    bias_dtype = rhs_bias.dtype if rhs_bias is not None else None

    tm, tk, tn, vmem_bytes = _get_tiling_size_for_gmm_kernel(
        m, k, n, g,
        lhs_dtype=lhs.dtype,
        rhs_dtype=rhs.dtype,
        out_dtype=lhs.dtype,
        scale_dtype=scale_dtype,
        quant_block_size=quant_block_size,
        bias_dtype=bias_dtype,
    )

    gmm_res = gmm(
        lhs=lhs,
        rhs=rhs,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_sizes=group_sizes,
        preferred_element_type=lhs.dtype,
        tiling=None,
        group_offset=group_offset[0],
        vmem_limit_bytes=vmem_bytes,
        group_metadata=group_metadata,
        num_active_tiles=num_active_tiles,
    )

    return gmm_res


def moe_gmm_local(
    x: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    topk_argsort_revert_indices: jax.Array,
    topk_weights: jax.Array,
    *,
    activation: str,
    topk: int,
    parallelism: Literal["tp", "ep"],
) -> jax.Array:
    """ Main MoE logic on a local shard can run in TP or EP mode.
    
    Set parallelism for "tp" or "ep"
    """

    assert parallelism in ["tp", "ep"]

    # Pre-compute group metadata once and share between GMM1 and GMM2.
    # Both calls have the same (m, g, group_sizes, group_offset) and the
    # same tm (which depends only on m and g, not k/n). 
    m = x.shape[0]
    g = w1.shape[0]  # num (local) experts — same for w1 and w2
    tm = _compute_tm(m, g)
    group_metadata, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes,
        m=m,
        tm=tm,
        start_group=group_offset[0],
        num_nonzero_groups=g,
        visit_empty_groups=False,
    )

    # GMM1 computes x @ (W_up | W_gate) tegether and then split out to apply activation
    # to the gate result
    gmm1_res_gate_up = gmm_wrapper(x, w1, w1_scale, w1_bias, group_sizes,
                                   group_offset, group_metadata,
                                   num_active_tiles)
    gmm1_res_gate, gmm1_res_up = jnp.split(gmm1_res_gate_up, 2, -1)
    gmm1_res = apply_act_fn(activation, gmm1_res_gate, gmm1_res_up)

    # When the parallelism is TP since w2_bias is not sharded, we should only apply bias
    # once, not applying to every shard. So we set w2_bias to 0 to all shards other than
    # shard 0. For EP, it is not needed since bias is sharded on leading expert axis.
    if parallelism == "tp" and w2_bias is not None:
        shard_id = jax.lax.axis_index(ShardingAxisName.MLP_TENSOR).sum()
        w2_bias = jnp.where(shard_id == 0, w2_bias, 0)

    gmm2_res = gmm_wrapper(gmm1_res, w2, w2_scale, w2_bias, group_sizes,
                           group_offset, group_metadata, num_active_tiles)

    # First run local reduction on topk experts owned by the rank for all tokens
    token_topk_hidden = gmm2_res[topk_argsort_revert_indices].reshape(
        (-1, topk, gmm2_res.shape[-1]))
    token_topk_hidden = token_topk_hidden * jnp.expand_dims(topk_weights,
                                                            axis=-1)
    token_hidden = token_topk_hidden.sum(axis=-2)

    reduction_axis = ShardingAxisName.MLP_TENSOR if parallelism == "tp" else ShardingAxisName.EXPERT
    # Then global reduction on all ranks for all tokens and all experts
    return jax.lax.psum(token_hidden, axis_name=reduction_axis)


def tensor_parallel_gmm(
    x: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    group_sizes: jax.Array,
    topk_argsort_revert_indices: jax.Array,
    topk_weights: jax.Array,
    *,
    activation: str,
    topk: int,
    mesh: Mesh,
) -> jax.Array:
    data_p_spec = P(ShardingAxisName.MLP_DATA)
    group_offset = jnp.array([0])

    w1_spec = P(None, None, ShardingAxisName.MLP_TENSOR)
    w2_spec = P(None, ShardingAxisName.MLP_TENSOR, None)

    w1_scale_spec = None if w1_scale is None else P(
        None, None, None, ShardingAxisName.MLP_TENSOR)
    w1_bias_spec = None if w1_bias is None else P(None, None,
                                                  ShardingAxisName.MLP_TENSOR)

    num_blocks = 1 if w2_scale is None else w2_scale.shape[1]
    w2_scale_spec = None if num_blocks == 1 else P(
        None, ShardingAxisName.MLP_TENSOR, None, None)
    w2_bias_spec = None if w2_bias is None else P(None, None, None)

    return jax.shard_map(
        functools.partial(moe_gmm_local,
                          activation=activation,
                          topk=topk,
                          parallelism="tp"),
        mesh=mesh,
        in_specs=(data_p_spec, w1_spec, w1_scale_spec, w1_bias_spec, w2_spec,
                  w2_scale_spec, w2_bias_spec, data_p_spec, data_p_spec,
                  data_p_spec, data_p_spec),
        out_specs=(data_p_spec),
        check_vma=False,
    )(x, w1, w1_scale, w1_bias, w2, w2_scale, w2_bias, group_sizes,
      group_offset, topk_argsort_revert_indices, topk_weights)


def expert_parallel_gmm(
    x: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    group_sizes: jax.Array,
    topk_argsort_revert_indices: jax.Array,
    topk_weights: jax.Array,
    *,
    activation: str,
    topk: int,
    mesh: Mesh,
) -> jax.Array:
    ep_size = get_mesh_shape_product(mesh, ShardingAxisName.EXPERT)
    ep_p_spec = P(ShardingAxisName.EXPERT)
    data_p_spec = P(ShardingAxisName.MLP_DATA)
    num_experts = w1.shape[0]
    num_experts_per_shard = num_experts // ep_size
    group_offset = jnp.arange(0, num_experts, num_experts_per_shard)

    w1_scale_spec = None if w1_scale is None else ep_p_spec
    w1_bias_spec = None if w1_bias is None else ep_p_spec
    w2_scale_spec = None if w2_scale is None else ep_p_spec
    w2_bias_spec = None if w2_bias is None else ep_p_spec

    return jax.shard_map(
        functools.partial(moe_gmm_local,
                          activation=activation,
                          topk=topk,
                          parallelism="ep"),
        mesh=mesh,
        in_specs=(data_p_spec, ep_p_spec, w1_scale_spec, w1_bias_spec,
                  ep_p_spec, w2_scale_spec, w2_bias_spec, data_p_spec,
                  ep_p_spec, data_p_spec, data_p_spec),
        out_specs=(data_p_spec),
        check_vma=False,
    )(x, w1, w1_scale, w1_bias, w2, w2_scale, w2_bias, group_sizes,
      group_offset, topk_argsort_revert_indices, topk_weights)


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "renormalize",
        "mesh",
        "use_ep",
        "activation",
        "scoring_fn",
    ),
)
def fused_moe_func(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    gating_output: jax.Array,
    topk: int,
    renormalize: bool,
    mesh: Mesh,
    use_ep: bool,
    activation: str,
    scoring_fn: str,
) -> jax.Array:
    """Route tokens in hidden_states into each experts based on routing.

    Args:
        hidden_states: [num_tokens, hidden_size]
        w1: first moe weights [num_experts, intermediate_size * 2, hidden_size]
        w2: second moe weights [num_experts, hidden_size, intermediate_size]
        w1_scale: w1 scale [num_experts, num_blocks, 1, intermediate_size * 2]
        w2_scale: w2 scale [num_experts, num_blocks, 1, hidden_size]
        w1_bias: optional bias of w1 [num_experts, 1, intermediate_size * 2]
        w2_bias: optional bias of w2 [num_experts, 1, hidden_size]
        gating_output: routing information of tokens [num_tokens, num_experts]
        topk: number of experts to choose per token.
        renormalize: normalize gating_output.
        mesh: mesh to perform moe.
        use_ep: use expert parallelism.
        activation: activation function to perform on the output of w1.
        scoring_fn: scoring function to apply on gating_output.

    Returns:
        Output of moe operation [num_tokens, hidden_size]
    """
    num_tokens, hidden_size = hidden_states.shape
    global_num_experts, padded_hidden_size, _ = w1.shape
    dtype = hidden_states.dtype

    assert (num_tokens * topk) % 16 == 0, (
        "The kernel requires num_tokens * topk to be a multiple of "
        f"16 but got {num_tokens}*{topk}={num_tokens*topk}")

    assert gating_output.shape == (num_tokens, global_num_experts)

    topk_weights = apply_scoring_fn(scoring_fn, gating_output)
    # All-gather topk weights for attention dp
    topk_weights = jax.lax.with_sharding_constraint(
        topk_weights, NamedSharding(mesh, P(ShardingAxisName.MLP_DATA, None)))
    topk_weights, topk_indices = jax.lax.top_k(topk_weights, k=topk)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
    topk_weights = topk_weights.astype(dtype)

    def _process_tokens_locally(hidden_states_local, topk_indices_local):
        num_tokens_local = hidden_states_local.shape[0]
        topk_indices_flat = topk_indices_local.flatten()
        topk_argsort_indices = jnp.argsort(topk_indices_flat)
        # Inverse permutation via scatter instead of a second O(n log n) argsort.
        topk_argsort_revert_indices = jnp.empty_like(topk_argsort_indices).at[
            topk_argsort_indices].set(jnp.arange(topk_argsort_indices.shape[0],
                                                  dtype=jnp.int32))
        token_indices = jnp.arange(num_tokens_local,
                                   dtype=jnp.int32).repeat(topk)
        token_indices_sorted = token_indices[topk_argsort_indices]
        group_sizes_local = jnp.bincount(topk_indices_flat,
                                         length=global_num_experts)

        x = hidden_states_local[token_indices_sorted]

        return x, group_sizes_local, topk_argsort_revert_indices

    x, group_sizes, topk_argsort_revert_indices = jax.shard_map(
        _process_tokens_locally,
        mesh=mesh,
        in_specs=(P(ShardingAxisName.MLP_DATA,
                    None), P(ShardingAxisName.MLP_DATA, None)),
        out_specs=(P(ShardingAxisName.MLP_DATA,
                     None), P(ShardingAxisName.MLP_DATA),
                   P(ShardingAxisName.MLP_DATA)))(hidden_states, topk_indices)

    x = jnp.pad(x, ((0, 0), (0, padded_hidden_size - hidden_size)))

    if use_ep:
        x = expert_parallel_gmm(x,
                                w1,
                                w1_scale,
                                w1_bias,
                                w2,
                                w2_scale,
                                w2_bias,
                                group_sizes,
                                topk_argsort_revert_indices,
                                topk_weights,
                                activation=activation,
                                topk=topk,
                                mesh=mesh)
    else:
        x = tensor_parallel_gmm(x,
                                w1,
                                w1_scale,
                                w1_bias,
                                w2,
                                w2_scale,
                                w2_bias,
                                group_sizes,
                                topk_argsort_revert_indices,
                                topk_weights,
                                activation=activation,
                                topk=topk,
                                mesh=mesh)

    return x[:num_tokens, :hidden_size]
