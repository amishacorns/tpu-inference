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
"""GMM kernel with fused gather: loads non-contiguous rows from HBM via DMA.

Instead of operating on a pre-permuted LHS (which costs ~1GB HBM write),
this kernel takes the un-permuted hidden_states in HBM and a token_indices
array to gather rows on the fly inside the Pallas kernel.

Based on gmm.py — fork kept minimal to reduce maintenance burden.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.megablox.gmm import (
    GroupMetadata,
    LutFn,
    _calculate_irregular_num_tiles,
    _get_store_mask,
    _zero_uninitialized_memory,
    make_group_metadata,
)
from tpu_inference.kernels.megablox.tuned_block_sizes import \
    get_tuned_block_sizes

partial = functools.partial


@functools.partial(
    jax.jit,
    static_argnames=[
        "preferred_element_type",
        "tiling",
        "interpret",
        "vmem_limit_bytes",
    ],
)
def gmm_gather(
    lhs: jnp.ndarray,
    token_indices: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    rhs_scale: jnp.ndarray | None = None,
    rhs_bias: jnp.ndarray | None = None,
    tiling: tuple[int, int, int] | LutFn | None = None,
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    interpret: bool = False,
    vmem_limit_bytes: int | None = None,
    group_metadata: GroupMetadata | None = None,
    num_active_tiles: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Like gmm() but with fused gather: LHS rows are loaded by index from HBM.

    Instead of:
        x = hidden_states[token_indices_sorted]   # materializes permuted LHS
        out = gmm(x, rhs, ...)

    This does:
        out = gmm_gather(hidden_states, token_indices_sorted, rhs, ...)

    The kernel keeps hidden_states in HBM and gathers rows on the fly.

    Args:
        lhs: The UN-PERMUTED activations [num_tokens, k]. Stays in HBM.
        token_indices: Which row of lhs to use for each row of the output [m].
            m = num_tokens * topk (sorted by expert).
        rhs: Expert weights [num_groups, k, n].
        group_sizes: Per-expert token counts [num_groups].
        ... (rest same as gmm())

    Returns:
        A 2d array [m, n].
    """
    # m is the number of output rows = len(token_indices), NOT lhs.shape[0]
    m = token_indices.shape[0]
    k = lhs.shape[1]

    if existing_out is not None:
        assert isinstance(existing_out, jax.Array)
        if existing_out.dtype != preferred_element_type:
            raise ValueError(
                "Existing output dtype must match preferred_element_type.")

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        if group_offset.shape:
            raise ValueError(
                f"group_offset must be a ()-shaped array. Got: {group_offset.shape}.")
        group_offset = group_offset[None]

    num_current_groups = rhs.shape[0]
    num_total_groups = group_sizes.shape[0]
    n = rhs.shape[-1]

    # Validate basic shapes.
    if lhs.ndim != 2:
        raise ValueError(f"Expected 2-tensor for 'lhs' but got {lhs.ndim=}.")
    if rhs.ndim != 3:
        raise ValueError(f"Expected 3-tensor for 'rhs' but got {rhs.ndim=}.")
    if lhs.shape[1] != rhs.shape[1]:
        raise ValueError(
            f"lhs k-dim ({lhs.shape[1]}) != rhs k-dim ({rhs.shape[1]})")
    if token_indices.ndim != 1:
        raise ValueError(
            f"Expected 1-d token_indices but got {token_indices.ndim=}.")

    # Look up tiling.
    if callable(tiling):
        tiling = tiling(m, k, n)
    elif tiling is None:
        tiling = get_tuned_block_sizes(
            m=m, k=k, n=n,
            num_total_groups=num_total_groups,
            num_current_groups=rhs.shape[0],
            lhs_dtype=str(lhs.dtype),
            rhs_dtype=str(rhs.dtype),
            quant_block_size=k,
        )
    if tiling is None:
        raise ValueError(
            f"No tuned tiling found for (m, k, n) = ({m}, {k}, {n})")

    tm, tk, tn = tiling

    if rhs_scale is not None:
        assert isinstance(rhs_scale, jax.Array)
        assert rhs_scale.shape[0] == num_current_groups
        num_quant_blocks = rhs_scale.shape[1]
    else:
        num_quant_blocks = 1
    quant_block_size = k // num_quant_blocks

    if tk % quant_block_size != 0 and quant_block_size % tk != 0:
        tk = quant_block_size

    tiles_k, k_rem = _calculate_irregular_num_tiles(k, tk)
    tiles_n, n_rem = _calculate_irregular_num_tiles(n, tn)
    del n_rem
    num_quant_blocks_per_tk = pl.cdiv(tk, quant_block_size)

    # Create group metadata (or reuse caller-provided).
    if group_metadata is None:
        group_metadata, num_active_tiles = make_group_metadata(
            group_sizes=group_sizes,
            m=m,
            tm=tm,
            start_group=group_offset[0],
            num_nonzero_groups=rhs.shape[0],
            visit_empty_groups=False,
        )

    # Mosaic requires DMA slices along dim-0 of VMEM refs to be aligned
    # to the sublane tiling (8 for bf16/fp8).  We reshape the hidden_states
    # to 3D: (num_tokens, packing, k // packing) where packing >= 8.
    # This puts the token dim on axis-0 where individual token DMAs are
    # fine (each token is packing × k//packing, satisfying sublane alignment
    # on the inner dims).
    #
    # Actually, the simpler approach used by the v1 fused MoE kernel:
    # keep dim-0 as tokens, pack into (t, t_packing, k // t_packing).
    # DMA of pl.ds(src_row, 1) on dim-0 works because the DMA transfers
    # the full inner dimensions which are sublane-aligned.
    SUBLANE_COUNT = 8  # TPU sublane count
    LANE_WIDTH = 128   # elements per lane for Mosaic VMEM tiling
    # We reshape lhs from 2D [num_tokens, k] to 3D [num_tokens, P, k//P]
    # so that per-token DMA (dim-0 slice of 1) is valid.
    # Mosaic VMEM tiling is (8, 128) on the last two dims, so we need:
    #   dim-1 (P) >= 8  AND  dim-2 (k//P) >= 128
    # We also need tk/P >= 128 since the DMA slices dim-2 to tk/P.
    #
    # If tk < P*128 (i.e. tk < 1024 for P=8), dim-2 slice would be
    # < 128.  In that case we reduce P until tk/P >= 128.
    # Special case: if tk < 128, this approach cannot work (but our real
    # workloads have tk >= 2048).
    MIN_TK = SUBLANE_COUNT * LANE_WIDTH  # 8 * 128 = 1024
    assert tk >= MIN_TK, (
        f"gmm_gather requires tk >= {MIN_TK}, got tk={tk}. "
        f"Use regular gmm() for small tile sizes.")
    assert k >= MIN_TK, (
        f"gmm_gather requires k >= {MIN_TK}, got k={k}.")
    # Choose largest P that satisfies both constraints.
    lhs_packing = min(SUBLANE_COUNT, tk // LANE_WIDTH)
    # Ensure P is a power of 2 for clean division.
    while lhs_packing > 1 and (tk % lhs_packing != 0 or k % lhs_packing != 0):
        lhs_packing //= 2
    assert lhs_packing >= 1
    assert tk % lhs_packing == 0
    assert k % lhs_packing == 0
    lhs_k_inner = k // lhs_packing    # full k dimension packed
    lhs_tk_inner = tk // lhs_packing   # tile k dimension packed
    assert lhs_tk_inner >= LANE_WIDTH, (
        f"lhs_tk_inner={lhs_tk_inner} must be >= {LANE_WIDTH}")

    # Reshape lhs before passing to pallas_call.
    lhs_3d = lhs.reshape(lhs.shape[0], lhs_packing, lhs_k_inner)

    # ── Kernel body ──────────────────────────────────────────────────────
    def kernel(
        group_metadata,
        group_offset,
        token_indices,   # scalar prefetch: [m] int32
        lhs_hbm,         # full un-permuted LHS in HBM [num_tokens, lhs_packing, lhs_k_inner]
        rhs,             # [num_groups, tk, tn] tile
        rhs_scale,
        rhs_bias,
        existing_out,
        out,
        acc_scratch,      # [tm, tn] f32
        lhs_scratch,      # [tm, lhs_packing, lhs_tk_inner] in lhs dtype
        dma_sem,          # DMA semaphore for LHS gather
    ):
        m_tile_ids = group_metadata.m_tile_ids
        del group_offset

        grid_id = pl.program_id(1)
        k_i = pl.program_id(2)

        # ── Gather LHS tile from HBM → VMEM scratch ──
        m_tile_start = m_tile_ids[grid_id] * tm
        k_inner_start = k_i * lhs_tk_inner

        # Issue tm individual token DMAs.
        # lhs_hbm is 3D: [num_tokens, lhs_packing, lhs_k_inner].
        # DMA one token at a time: slice dim-0 by 1, take full dim-1
        # (lhs_packing=8 which satisfies sublane alignment), slice dim-2
        # for the k-tile.
        for row_i in range(tm):
            src_row = token_indices[m_tile_start + row_i]
            pltpu.make_async_copy(
                src_ref=lhs_hbm.at[
                    pl.ds(src_row, 1),
                    :,
                    pl.ds(k_inner_start, lhs_tk_inner)],
                dst_ref=lhs_scratch.at[pl.ds(row_i, 1), :, :],
                sem=dma_sem,
            ).start()

        # Wait for all DMAs to complete.
        pltpu.make_async_copy(
            src_ref=lhs_scratch,
            dst_ref=lhs_scratch,
            sem=dma_sem,
        ).wait()

        # Reshape gathered LHS back to 2D [tm, tk] for the matmul.
        gathered_lhs = lhs_scratch[...].reshape(tm, tk)

        @pl.when(k_i == 0)
        def _zero_acc():
            acc_scratch[...] = jnp.zeros_like(acc_scratch)

            if existing_out is not None:
                prev_grid_id = jnp.where(grid_id > 0, grid_id - 1, 0)
                is_first_processed_group = grid_id == 0
                m_tile_changed = m_tile_ids[grid_id] != m_tile_ids[
                    prev_grid_id]
                first_time_seeing_out = jnp.logical_or(
                    is_first_processed_group, m_tile_changed)

                @pl.when(first_time_seeing_out)
                def _init_out():
                    out[...] = existing_out[...]

        def mask_k_rem(x, *, dim):
            if k_rem == 0:
                return x
            orig_dtype = x.dtype
            iota = lax.broadcasted_iota(jnp.int32, x.shape, dim)
            x = x.astype(jnp.float32)
            return jnp.where(iota < k_rem, x, 0).astype(orig_dtype)

        def _accum(is_last_k_tile):
            if is_last_k_tile:
                mask_k_rem_lhs = partial(mask_k_rem, dim=1)
                mask_k_rem_rhs = partial(mask_k_rem, dim=0)
            else:
                def _wrapper(x):
                    return x
                mask_k_rem_lhs = _wrapper
                mask_k_rem_rhs = _wrapper

            # Read gathered LHS from VMEM scratch instead of BlockSpec ref.
            loaded_lhs = mask_k_rem_lhs(gathered_lhs)
            loaded_rhs = mask_k_rem_rhs(rhs[...])

            acc = acc_scratch[...]
            for b_i in range(num_quant_blocks_per_tk):
                partial_result = jnp.dot(
                    loaded_lhs[..., b_i * quant_block_size:(b_i + 1) *
                               quant_block_size],
                    loaded_rhs[b_i * quant_block_size:(b_i + 1) *
                               quant_block_size, ...],
                    preferred_element_type=jnp.float32,
                )
                if rhs_scale is not None:
                    partial_result *= jnp.broadcast_to(rhs_scale[b_i],
                                                       partial_result.shape)
                acc = acc + partial_result

            if is_last_k_tile:
                loaded_out = out[...].astype(jnp.float32)
                if rhs_bias is not None:
                    acc += rhs_bias[...].astype(jnp.float32)

                mask = _get_store_mask(
                    grid_id=grid_id,
                    group_metadata=group_metadata,
                    tm=tm,
                    tn=tn,
                )
                out[...] = jax.lax.select(
                    mask[...], acc, loaded_out).astype(preferred_element_type)
            else:
                acc_scratch[...] = acc

        is_last_k_tile = k_i == (tiles_k - 1)
        lax.cond(
            is_last_k_tile,
            partial(_accum, True),
            partial(_accum, False),
        )

    # ── Transform indices for RHS / output (same as gmm.py) ─────────────

    def rhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset,
                              token_indices):
        group_ids = group_metadata.group_ids
        del token_indices
        return group_ids[grid_id] - group_offset[0], k_i, n_i

    def rhs_scale_transform_indices(n_i, grid_id, k_i, group_metadata,
                                    group_offset, token_indices):
        group_ids = group_metadata.group_ids
        del token_indices
        b_i = (k_i * tk) // quant_block_size
        b_tile_i = b_i // num_quant_blocks_per_tk
        return group_ids[grid_id] - group_offset[0], b_tile_i, 0, n_i

    def rhs_bias_transform_indices(n_i, grid_id, k_i, group_metadata,
                                   group_offset, token_indices):
        group_ids = group_metadata.group_ids
        del k_i, token_indices
        return group_ids[grid_id] - group_offset[0], 0, n_i

    def out_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset,
                              token_indices):
        m_tile_ids = group_metadata.m_tile_ids
        del k_i, group_offset, token_indices
        return m_tile_ids[grid_id], n_i

    out_block_spec = pl.BlockSpec((tm, tn), out_transform_indices)
    if existing_out is None:
        in_out_block_spec: Any = None
        input_output_aliases = {}
    else:
        in_out_block_spec = out_block_spec
        _preceding_args = (
            group_metadata, group_offset, token_indices,
            lhs_3d, rhs, rhs_scale, rhs_bias,
        )
        _existing_out_idx = sum(
            len(jax.tree_util.tree_leaves(a)) for a in _preceding_args
        )
        input_output_aliases = {_existing_out_idx: 0}

    # LHS stays in HBM — whole tensor, no tiling via BlockSpec.
    lhs_block_spec = pl.BlockSpec(memory_space=pltpu.ANY)

    rhs_block_spec = pl.BlockSpec((None, tk, tn), rhs_transform_indices)

    if rhs_scale is None:
        rhs_scale_block_spec = None
    else:
        rhs_scale_block_spec = pl.BlockSpec(
            (None, num_quant_blocks_per_tk, 1, tn),
            rhs_scale_transform_indices)

    if rhs_bias is None:
        rhs_bias_block_spec = None
    else:
        rhs_bias_block_spec = pl.BlockSpec((None, 1, tn),
                                           rhs_bias_transform_indices)

    # ── Cost estimate ────────────────────────────────────────────────────
    # LHS bytes: we read m*k bytes total (same as before, just scattered).
    lhs_bytes = m * k * lhs.itemsize
    rhs_bytes = (k * n) * rhs.itemsize
    if rhs_scale is not None:
        rhs_bytes += (num_quant_blocks * n) * rhs_scale.itemsize
    if rhs_bias is not None:
        rhs_bytes += n * rhs_bias.itemsize
    out_bytes = 2 * (m * n) * jnp.dtype(preferred_element_type).itemsize
    tiles_m = -(-m // tm)
    bytes_accessed = ((lhs_bytes * tiles_n) + (rhs_bytes * tiles_m) +
                      out_bytes)
    flops = 2 * m * k * n
    cost_estimate = pl.CostEstimate(flops=flops,
                                    bytes_accessed=bytes_accessed,
                                    transcendentals=0)

    # ── Scratch shapes ───────────────────────────────────────────────────
    # acc_scratch: accumulator (same as gmm.py)
    # lhs_scratch: VMEM staging buffer for gathered LHS rows
    # dma_sem: semaphore for async DMA
    scratch_shapes = [
        pltpu.VMEM((tm, tn), jnp.float32),                           # acc_scratch
        pltpu.VMEM((tm, lhs_packing, lhs_tk_inner), lhs.dtype),      # lhs_scratch (3D)
        pltpu.SemaphoreType.DMA,                                      # dma_sem
    ]

    call_gmm = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), preferred_element_type),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,  # group_metadata, group_offset, token_indices
            in_specs=[
                lhs_block_spec,        # lhs in HBM (whole tensor)
                rhs_block_spec,
                rhs_scale_block_spec,
                rhs_bias_block_spec,
                in_out_block_spec,
            ],
            out_specs=out_block_spec,
            grid=(tiles_n, num_active_tiles, tiles_k),
            scratch_shapes=scratch_shapes,
        ),
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(dimension_semantics=(
            "parallel",
            "arbitrary",
            "arbitrary",
        ), vmem_limit_bytes=vmem_limit_bytes),
        interpret=interpret,
        cost_estimate=cost_estimate,
        name=f"gmm_gather-m_{m}-k_{k}-n_{n}-tm_{tm}-tk_{tk}-tn_{tn}",
    )

    out = call_gmm(
        group_metadata,
        group_offset,
        token_indices,
        lhs_3d,
        rhs,
        rhs_scale,
        rhs_bias,
        existing_out,
    )
    if existing_out is None and num_current_groups < num_total_groups:
        out = _zero_uninitialized_memory(
            out,
            start_group=group_offset[0],
            num_nonzero_groups=rhs.shape[0],
            group_metadata=group_metadata,
        )
    return out
