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
"""Fused expert-parallel MoE kernel: one Pallas TPU kernel computes the
expert half of an MoE layer for one expert-parallel shard and pushes each
result row to the shard that owns its token, so the layer needs no dense
combine reduce-scatter."""
import functools
import threading

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP8_MAX = 448.0
FP8 = jnp.float8_e4m3fn
# Storage only: there is no four-bit MXU, so each block is widened to e4m3.
FP4 = jnp.float4_e2m1fn
# Default four-bit scale block along the contraction axis.
FP4_QB = 512
# e2m1 values packed per uint32 word along K.
FP4_PACK = 8
DN = (((1, ), (0, )), ((), ()))
# The single-axis mesh name the layer's shard_map uses.
AXIS = "d"
# Weight buffer slots. The refill for expert e+2 is issued at expert e's
# head, so the slots holding e, e+1 and e+2 must all be distinct.
NBUF = 3
# Column chunk of the intermediate requantization.
QCHUNK = 512
# Every transport moves whole 8-row blocks; dynamic DMA offsets are
# block-aligned.
ROWBLK = 8
# A token row is a whole number of 128-lane blocks, staged as (subq, 128).
HIDDEN_LANE_BLOCK = 128
HIDDEN_MAX_BLOCKS = 32
# Fraction of the generation's VMEM capacity the kernel may use.
VMEM_FRACTION = 0.98


def vmem_limit():
    """VMEM budget for the kernel, read from this generation's capacity."""
    return int(pltpu.get_tpu_info().vmem_capacity_bytes * VMEM_FRACTION)


def align_up(v, m):
    return -(-v // m) * m


# How far this kernel's output may sit from the same MoE block computed with
# a bfloat16 wire, as a relative difference. Callers and tests read this.
WIRE_RELATIVE_DELTA_BOUND = 0.05


def rowquant_fp8(x):
    """Per-row dynamic fp8 quant, reducing and applying in bf16."""
    amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True).astype(jnp.float32)
    scale = amax / FP8_MAX
    sinv = jnp.where(scale == 0, 0.0, 1.0 / scale)
    return (x * sinv.astype(x.dtype)).astype(FP8), scale


def _absmax_rows(x):
    return jnp.max(jnp.abs(x), axis=-1, keepdims=True)


def _quant_scale(amax):
    scale = amax.astype(jnp.float32) / FP8_MAX
    sinv = jnp.where(scale == 0, 0.0, 1.0 / scale)
    return scale, sinv


def _quant_apply(x, sinv_native):
    return (x * sinv_native).astype(FP8)


def _ffn_dots_pre(q, s, w1, w2, w1s):
    """GMM1 -> silu/postscale/mid-quant -> GMM2 from q fp8 [m, k], s f32 [m, 1]."""
    inter = w1.shape[-1] // 2
    acc1 = lax.dot_general(q, w1, DN, preferred_element_type=jnp.float32)
    mids, amax = [], None
    for c0 in range(0, inter, QCHUNK):
        c1 = min(c0 + QCHUNK, inter)
        g = acc1[:, c0:c1] * s * w1s[:, c0:c1]
        u = acc1[:, inter + c0:inter + c1] * s * w1s[:, inter + c0:inter + c1]
        mc = (jax.nn.silu(g) * u).astype(jnp.bfloat16)
        mids.append(mc)
        pm = _absmax_rows(mc)
        amax = pm if amax is None else jnp.maximum(amax, pm)
    s2, sinv2 = _quant_scale(amax)
    sb = sinv2.astype(jnp.bfloat16)
    q2 = jnp.concatenate([_quant_apply(mc, sb) for mc in mids], axis=-1)
    acc2 = lax.dot_general(q2, w2, DN, preferred_element_type=jnp.float32)
    return acc2, s2


def _ffn_dots_pre_fp4(q, s, w1_of, w2_of, w1s_b, w2s_b, *, qb=FP4_QB):
    """Block-scaled form of _ffn_dots_pre for four-bit weights."""
    # w1_of(b) / w2_of(b) return weight k-block b widened to fp8; the block
    # scales w1s_b [nb1, 2*inter] / w2s_b [nb2, hidden] apply post-matmul.
    nb1, nb2 = w1s_b.shape[0], w2s_b.shape[0]
    inter = w1s_b.shape[-1] // 2
    acc1 = None
    for b in range(nb1):
        blk = lax.dot_general(q[:, b * qb:(b + 1) * qb],
                              w1_of(b),
                              DN,
                              preferred_element_type=jnp.float32)
        blk = blk * w1s_b[b][None, :]
        acc1 = blk if acc1 is None else acc1 + blk
    mids, amax = [], None
    for c0 in range(0, inter, QCHUNK):
        c1 = min(c0 + QCHUNK, inter)
        g = acc1[:, c0:c1] * s
        u = acc1[:, inter + c0:inter + c1] * s
        mc = (jax.nn.silu(g) * u).astype(jnp.bfloat16)
        mids.append(mc)
        pm = _absmax_rows(mc)
        amax = pm if amax is None else jnp.maximum(amax, pm)
    s2, sinv2 = _quant_scale(amax)
    sb = sinv2.astype(jnp.bfloat16)
    q2 = jnp.concatenate([_quant_apply(mc, sb) for mc in mids], axis=-1)
    acc2 = None
    for b in range(nb2):
        blk = lax.dot_general(q2[:, b * qb:(b + 1) * qb],
                              w2_of(b),
                              DN,
                              preferred_element_type=jnp.float32)
        blk = blk * w2s_b[b][None, :]
        acc2 = blk if acc2 is None else acc2 + blk
    return acc2, s2


# Replicated all-to-all routing plan.
def plan_ragged_dispatch(topk_idx,
                         topk_w,
                         *,
                         e_total,
                         ep,
                         t_local,
                         block=256,
                         tile_m=256,
                         rows_alloc=None,
                         shard_stride=None):
    """Assign every routed (token, expert) pair the slab row it computes on."""
    # topk_idx [T, K] i32 expert ids, topk_w [T, K] f32 weights, all-gathered.
    # Rows order by expert then by owning shard, each run padded to ROWBLK.
    T, K = topk_idx.shape
    n = T * K
    g_local = e_total // ep
    egroup = 1  # the push is always per expert
    n_grp = g_local // egroup
    assert e_total % ep == 0
    # comb_tbl below packs the position and the alignment slot as
    # position * 64 + slot. The slot runs to (ROWBLK - 1) * (ep - 1), so a
    # wider ep overflows the field and the two silently corrupt each other.
    assert (ROWBLK - 1) * (ep - 1) < 64, (
        f"ep={ep} needs up to {(ROWBLK - 1) * (ep - 1)} alignment slots and "
        f"the routing plan packs them into 64; widths up to "
        f"{1 + 63 // (ROWBLK - 1)} are representable")
    assert n % block == 0, (n, block)
    assert (t_local * K) % block == 0, (t_local * K, block)
    assert tile_m % ROWBLK == 0
    if shard_stride is None and rows_alloc is None:
        # + tile_m: the last expert's tail tile reads a full static window
        # past its last row.
        rows_alloc = align_up(n + (ROWBLK - 1) * e_total * ep + tile_m, tile_m)
    if rows_alloc is not None:
        assert rows_alloc % tile_m == 0
    flat = topk_idx.reshape(-1).astype(jnp.int32)
    nb = n // block
    e_blk = flat.reshape(nb, block)
    bins = jnp.arange(e_total, dtype=jnp.int32)

    # Counting-sort core.
    blk_hist = jnp.sum((e_blk[:, :, None] == bins[None,
                                                  None, :]).astype(jnp.int32),
                       axis=1)  # [nb, E]
    block_off = jnp.cumsum(blk_hist, axis=0) - blk_hist  # excl over blocks
    base_per_slot = jnp.sum(jnp.where(e_blk[:, :, None] == bins[None, None, :],
                                      block_off[:, None, :], 0),
                            axis=2)  # [nb, blk]
    eq = e_blk[:, :, None] == e_blk[:, None, :]
    tri = jnp.tril(jnp.ones((block, block), dtype=jnp.bool_), k=-1)
    rank = jnp.sum((eq & tri[None]).astype(jnp.int32), axis=2)
    q = (base_per_slot + rank).reshape(-1)  # [n] slot rank

    blocks_per_d = (t_local * K) // block
    len_by_d = blk_hist.reshape(ep, blocks_per_d, e_total).sum(axis=1)
    len_raw = len_by_d.T  # [E, ep]
    run_start = jnp.cumsum(len_raw, axis=1) - len_raw  # excl over d

    la_full = -(-len_raw // ROWBLK) * ROWBLK
    run_start_a = jnp.cumsum(la_full, axis=1) - la_full  # aligned starts
    slot_tbl = run_start_a - run_start  # slot = q + tbl
    group_sizes = len_raw.sum(axis=1)  # [E] true rows

    # Ragged expert slabs.
    rows_a = la_full.sum(axis=1)  # [E] 8-aligned
    ntiles = -(-rows_a // tile_m)  # [E]
    total_rows = rows_a.sum()
    if shard_stride is None:
        expert_base = jnp.cumsum(rows_a) - rows_a  # [E] excl
        stride_over = jnp.int32(0)
    else:
        # Static per-shard slabs: shard s owns rows [s*stride, (s+1)*stride).
        assert rows_alloc is None, "shard_stride fixes the allocation"
        assert shard_stride % tile_m == 0
        ra_s = rows_a.reshape(ep, g_local)
        local_base = jnp.cumsum(ra_s, axis=1) - ra_s  # [s, G]
        expert_base = (local_base + (jnp.arange(ep, dtype=jnp.int32)[:, None] *
                                     shard_stride)).reshape(e_total)
        # Footprint plus one tile_m tail-read window must fit the stride; the
        # caller must keep stride_over at zero or a shard bleeds into the next.
        stride_over = jnp.maximum(ra_s.sum(axis=1) + tile_m - shard_stride,
                                  0).sum()
        rows_alloc = ep * shard_stride

    lc4 = la_full.reshape(ep, n_grp, egroup, ep)
    reg_len = lc4.sum(axis=2)  # [s, g, d]
    rl_d = reg_len.transpose(2, 0, 1).reshape(ep, ep * n_grp)  # [d, s*g]
    rb = (jnp.cumsum(rl_d, axis=1) - rl_d).reshape(ep, ep, n_grp)  # [d,s,g]
    recv_rows = rl_d.sum(axis=1)  # [d] incl. self
    elo = (jnp.cumsum(lc4, axis=2) - lc4).reshape(e_total, ep)  # [E, d]

    rb_sgd = rb.transpose(1, 2, 0)  # [s, g, d]
    rb_exp = jnp.broadcast_to(rb_sgd[:, :, None, :],
                              (ep, n_grp, egroup, ep)).reshape(e_total, ep)
    p_tbl = rb_exp + elo - run_start  # [E, ep] i32
    d_blk = (jnp.arange(nb, dtype=jnp.int32) * block) // (t_local * K)

    # Packed pass (p_tbl*64 + slot_tbl); the ragged expert base needs its own
    # select-sum because its value range is too wide for the packed word.
    comb_tbl = p_tbl * 64 + slot_tbl
    comb_b = jnp.take(comb_tbl.T, d_blk, axis=0)  # [nb, E]
    sel = jnp.sum(jnp.where(e_blk[:, :, None] == bins[None, None, :],
                            comb_b[:, None, :], 0),
                  axis=2).reshape(-1)
    ebase_j = jnp.sum(jnp.where(e_blk[:, :, None] == bins[None, None, :],
                                expert_base[None, None, :], 0),
                      axis=2).reshape(-1)
    pos = q + sel // 64
    slot = q + sel % 64
    valid = jnp.ones((n, ), jnp.bool_)  # nothing drops

    t_of_j = jnp.arange(n, dtype=jnp.int32) // K
    sink = rows_alloc
    didx = ebase_j + slot  # always < total
    token_gather = jnp.zeros(
        (rows_alloc + 1, ),
        jnp.int32).at[didx].add(t_of_j, mode="promise_in_bounds")[:-1]
    w_bits = lax.bitcast_convert_type(
        topk_w.reshape(-1).astype(jnp.float32), jnp.int32)
    w_row = lax.bitcast_convert_type(
        jnp.zeros((rows_alloc + 1, ),
                  jnp.int32).at[didx].add(w_bits,
                                          mode="promise_in_bounds")[:-1],
        jnp.float32)

    return dict(q=q,
                slot=slot,
                valid=valid,
                pos=pos.reshape(T, K),
                token_gather=token_gather,
                w_row=w_row,
                didx=didx,
                len_c=len_raw,
                len_a=la_full,
                start_c=run_start_a,
                reg_len=reg_len,
                rb=rb,
                elo=elo,
                recv_rows=recv_rows,
                group_sizes=group_sizes,
                overflow=jnp.int32(0),
                rows_a=rows_a,
                expert_base=expert_base,
                ntiles=ntiles,
                total_rows=total_rows,
                rows_alloc=rows_alloc,
                stride_over=stride_over,
                tile_m=tile_m,
                _sink=sink)


def shard_tables_ragged(plan, me, *, e_total, ep):
    """Per-shard ragged row tables: (rows, base) i32 [G], both in row units."""
    g_local = e_total // ep
    rows = lax.dynamic_slice(plan["rows_a"], (me * g_local, ), (g_local, ))
    base_g = lax.dynamic_slice(plan["expert_base"], (me * g_local, ),
                               (g_local, ))
    base = base_g - base_g[0]
    return rows.astype(jnp.int32), base.astype(jnp.int32)


def active_list(rows, g_local, occ_sort=None, tile_m=None):
    """Active-expert visit list and count from rows [g_local] i32."""
    # active[:n_active] = the local expert indices with rows > 0; the tail is
    # never visited. occ_sort orders them: None index, 'desc' most rows first,
    # 'asc' fewest, 'ilv' multi-tile and sub-tile alternating. Ties break by
    # ascending index. Every order gives bitwise-identical output.
    mask = rows > 0
    n_active = mask.sum().astype(jnp.int32).reshape(1)
    order = jnp.arange(g_local, dtype=jnp.int32)
    if occ_sort is None:
        active = jnp.sort(jnp.where(mask, order, jnp.int32(g_local)))[:g_local]
        active = jnp.minimum(active, jnp.int32(g_local - 1)).astype(jnp.int32)
        return active, n_active
    if occ_sort not in ("desc", "asc", "ilv"):
        raise ValueError(
            f"occ_sort={occ_sort!r} -- expected None, 'desc', 'asc', or 'ilv'")
    rows_i = rows.astype(jnp.int32)
    # lexsort keys are least-significant first: the last key is primary.
    inactive = (~mask).astype(jnp.int32)
    rowkey = rows_i if occ_sort == "asc" else (-rows_i)  # 'ilv' builds on desc
    perm = jnp.lexsort((order, rowkey, inactive)).astype(jnp.int32)
    if occ_sort in ("desc", "asc"):
        active = jnp.minimum(perm, jnp.int32(g_local - 1)).astype(jnp.int32)
        return active, n_active
    # occ_sort == 'ilv': split the desc-order active prefix at tile_m and
    # re-emit alternating, so the weight refill pipeline stays fed.
    if tile_m is None:
        raise ValueError(
            "occ_sort='ilv' requires tile_m (the plan tile height = the layer "
            "capacity in the ragged path) to classify multi-tile 'hot' vs "
            "sub-tile 'cold' experts")
    na = n_active[0]
    rows_perm = rows_i[perm]  # rows in desc (hot-first) order
    is_active = order < na  # desc positions [0, n_active)
    is_hot = (rows_perm > jnp.int32(tile_m)) & is_active  # multi-tile experts
    is_cold = is_active & (~is_hot)  # sub-tile actives (rows<=tile_m)
    hi = is_hot.astype(jnp.int32)
    ci = is_cold.astype(jnp.int32)
    n_hot = hi.sum().astype(jnp.int32)
    n_cold = na - n_hot
    # exclusive prefix rank within the hot / cold subsequences of the desc order
    hot_rank = jnp.cumsum(hi) - hi  # 0..n_hot-1 at the hot positions
    cold_rank = jnp.cumsum(ci) - ci  # 0..n_cold-1 at the cold positions
    # Interleave target: a bijection of the active desc positions onto
    # [0, n_active), with the exhausted class's leftovers at the tail.
    pos_hot = jnp.where(hot_rank < n_cold, 2 * hot_rank, n_cold + hot_rank)
    pos_cold = jnp.where(cold_rank < n_hot, 2 * cold_rank + 1,
                         n_hot + cold_rank)
    target = jnp.where(is_hot, pos_hot, pos_cold)
    # The inactive tail keeps its desc position, so the scatter is bijective.
    tgt = jnp.where(is_active, target, order).astype(jnp.int32)
    active = jnp.zeros(g_local, jnp.int32).at[tgt].set(perm)
    active = jnp.minimum(active, jnp.int32(g_local - 1)).astype(jnp.int32)
    return active, n_active


def shard_tables_diet(plan, me, *, e_total, ep):
    """Extra per-shard tables in row units, for true-length pushes."""
    # lc [G, ep] true rows per (e, d), pdstr [G, ep] recv row offset of that
    # run, tott [2] send and remote recv rows. Only the pushed lengths shrink:
    # the recv and contrib layouts stay aligned, so pos tables do not move.
    g_local = e_total // ep
    egroup = 1  # the push is always per expert
    n_grp = g_local // egroup
    lc_full = plan["len_c"]  # [E, ep]
    lc = lax.dynamic_slice(lc_full, (me * g_local, 0), (g_local, ep))
    elo = lax.dynamic_slice(plan["elo"], (me * g_local, 0), (g_local, ep))
    rb = plan["rb"]  # [d, s, g]
    rb_dst = lax.dynamic_slice(rb, (0, me, 0), (ep, 1, n_grp))[:, 0]
    g_of_e = jnp.arange(g_local) // egroup
    pdstr = rb_dst.T[g_of_e] + elo  # [G, d]
    not_me = (jnp.arange(ep) != me)
    send_true = (lc * not_me[None, :]).sum()
    self_true = (lc * (~not_me)[None, :]).sum()
    recv_true = lax.dynamic_slice(
        lc_full.sum(axis=0).astype(jnp.int32), (me,), (1,))[0] \
        - self_true
    return (lc.astype(jnp.int32), pdstr.astype(jnp.int32),
            jnp.stack([send_true, recv_true]).astype(jnp.int32))


def shard_tables(plan, me, *, e_total, ep, capacity):
    """Cut the replicated plan into shard `me`'s kernel prefetch tables."""
    # All i32. contrib: regions per dest d, packed in d order, each region =
    # groups asc, experts asc. recv: regions per (src asc, group asc).
    g_local = e_total // ep
    egroup = 1  # the push is always per expert
    n_grp = g_local // egroup
    B = ROWBLK  # every table value below is in 8-row BLOCK units
    la = lax.dynamic_slice(plan["len_a"], (me * g_local, 0), (g_local, ep))
    st = lax.dynamic_slice(plan["start_c"], (me * g_local, 0), (g_local, ep))
    elo = lax.dynamic_slice(plan["elo"], (me * g_local, 0), (g_local, ep))
    rme = lax.dynamic_slice(plan["reg_len"], (me, 0, 0), (1, n_grp, ep))[0]
    rb = plan["rb"]  # [d, s, g]

    not_me = (jnp.arange(ep) != me)
    # contrib includes the own-dest region, which hops to recvbuf later.
    out_total = rme.sum(axis=0)  # [d] incl. me
    contrib_base = jnp.cumsum(out_total) - out_total  # [d]
    grp_off = jnp.cumsum(rme, axis=0) - rme  # [g, d]

    g_of_e = jnp.arange(g_local) // egroup
    coff = contrib_base[None, :] + grp_off[g_of_e] + elo  # [G, d]

    push_src = contrib_base[None, :] + grp_off  # [g, d]
    push_len = rme
    # rb[d (receiver), me (src), g] for every d: [d, g] -> [g, d].
    rb_dst = lax.dynamic_slice(rb, (0, me, 0), (ep, 1, n_grp))[:, 0]
    push_dst = rb_dst.T
    crows = la.sum(axis=1)  # [G] aligned
    send_rows = (rme * not_me[None, :]).sum()
    # Rows this shard receives from itself.
    self_rows = jnp.sum(la * (~not_me)[None, :])
    recv_remote = lax.dynamic_slice(plan["recv_rows"], (me,), (1,))[0] \
        - self_rows
    tot = jnp.stack([send_rows // B, recv_remote // B]).astype(jnp.int32)

    def i32(a):
        return (a // B).astype(jnp.int32)

    return (i32(st), i32(la), i32(coff), i32(crows), i32(push_src),
            i32(push_len), i32(push_dst), tot)


# The kernel.
def vmem_estimate_bytes(
        g_local,
        capacity,
        hidden,
        inter,
        nbuf=NBUF,
        # Matches what the build passes; a default of False
        # under-reads the estimate by six to eight percent.
        fp8_wire=True,
        rhs_fp4=False,
        rhs_qb=FP4_QB):
    b = nbuf * capacity * hidden  # lhs_q (fp8)
    if rhs_fp4:
        # packed-u32 weight stream: half the eight-bit bytes
        b += (nbuf * hidden * 2 * inter + nbuf * inter * hidden) // 2
    else:
        b += nbuf * hidden * 2 * inter + nbuf * inter * hidden  # fp8
    # out_vm: bf16, or fp8 plus the f32 row-scale mirror
    b += 2 * capacity * hidden * (1 if fp8_wire else 2)
    if fp8_wire:
        b += 2 * capacity * max(1, hidden // 1024) * 4  # oscl_vm
    if rhs_fp4:
        # qb block scales resident: w1s [G, K/qb, 2I] + w2s [G, I/qb, H]
        b += (g_local * (hidden // rhs_qb) * 2 * inter * 4 + g_local *
              (inter // rhs_qb) * hidden * 4)
    else:
        b += g_local * 2 * inter * 4 + g_local * hidden * 4  # w1s/w2s
    b += g_local * capacity * 4  # ls_vm
    if not fp8_wire:
        b += g_local * capacity * 4  # wrow_vm
    b += capacity * (2 * inter * 4 + inter + hidden * 4 + hidden * 2)
    if rhs_fp4:
        # widen transients: 1-2 fp8 k-blocks of w1/w2 in flight
        b += 2 * rhs_qb * 2 * inter
    return b


def _build_fused_ep_moe_kernel(*,
                               g_local,
                               capacity,
                               hidden,
                               inter,
                               ep,
                               comm="a2a8",
                               interpret=False,
                               lhs_indirect=False,
                               pad_diet=False,
                               rhs_fp4=False,
                               rhs_qb=FP4_QB,
                               ragged_rows_alloc=None,
                               ragged=True,
                               defer_own_hop=True,
                               refill_priority=1,
                               early_weight_prefetch=True,
                               skip_empty_experts=False,
                               collective_id=0):
    """Build the pallas_call and the function that invokes it."""
    # lhs_indirect: the kernel fetches lhs rows by index from the ungathered
    # buffer qg [tokens, hidden//128, 128] fp8, using the token_gather table.
    # comm="none": fixed-capacity pipeline, out [G, cap, H], no collectives.
    # comm="a2a8": all-to-all on an fp8 e4m3 wire, carrying unweighted rows;
    # returns (arrival rows fp8, arrival scales f32).
    assert comm in ("none", "a2a8")
    G = g_local
    # Weight prefetch: each expert's slab streams into one of NBUF slots, and
    # the refill for expert e + DIST is issued at expert e's head.
    _prefetch = bool(early_weight_prefetch) and bool(ragged)
    _prefetch_dist = 2
    # That refill writes its slot while the readers of e - DIST .. e - 1 are
    # live, so NBUF must exceed DIST or a live reader's slot is overwritten.
    assert 0 < _prefetch_dist < NBUF, (
        f"weight prefetch distance {_prefetch_dist} must satisfy "
        f"0 < distance < NBUF={NBUF}, so that distance + 1 consecutive "
        "experts occupy distinct weight slots")
    if not _prefetch:
        raise NotImplementedError(
            "the kernel builds only the per-expert push schedule: pass "
            "early_weight_prefetch=True and ragged=True. Each expert's weight "
            "slab rotates through NBUF slots keyed on the expert counter, and "
            "the refill for a later expert is issued at an earlier expert's "
            "head. A schedule that gives each expert a single slot cannot "
            "express that -- it would race its own refill. This covers the "
            "four-bit weight path too, which rides the same rotation.")
    egroup = 1  # the push is always per expert
    assert capacity % ROWBLK == 0
    cblk = capacity // ROWBLK
    n_grp = G // egroup
    do_a2a8 = comm == "a2a8"
    if rhs_fp4:
        assert comm in ("none", "a2a8")
        assert hidden % rhs_qb == 0 and inter % rhs_qb == 0
        nb1, nb2 = hidden // rhs_qb, inter // rhs_qb
        # packed rows per k-block must land on the u32 sublane tile
        assert rhs_qb % (8 * FP4_PACK) == 0
    if lhs_indirect or pad_diet:
        assert (hidden % HIDDEN_LANE_BLOCK == 0
                and hidden <= HIDDEN_MAX_BLOCKS * HIDDEN_LANE_BLOCK)
        subq = hidden // HIDDEN_LANE_BLOCK  # token row = one (subq, 128) block
    if pad_diet:
        # True-length remote pushes: out_vm, contrib and recv take the per-row
        # [rows, subq, 128] geometry, which row-granular offsets require.
        assert do_a2a8
    slanes = max(1, hidden // 1024)  # f32 scale-mirror lanes per row
    # VMEM fit, so an overrunning (NBUF, capacity) pair is caught early.
    est = vmem_estimate_bytes(G,
                              capacity,
                              hidden,
                              inter,
                              nbuf=NBUF,
                              fp8_wire=do_a2a8,
                              rhs_fp4=rhs_fp4,
                              rhs_qb=rhs_qb)
    limit = vmem_limit()
    assert est <= limit, (
        f"VMEM estimate {est/2**20:.1f}MiB over limit "
        f"{limit/2**20:.1f}MiB at NBUF={NBUF} capacity={capacity}")
    do_a2a = do_a2a8
    do_xport = do_a2a8
    # The own-destination copy becomes an async start drained at the tail.
    assert defer_own_hop in (0, 1, False, True), \
        f"defer_own_hop={defer_own_hop!r}"
    defer_mehop = do_a2a and bool(defer_own_hop)
    # Each expert runs a fori_loop over ceil(rows[e] / tile_m) static
    # [tile_m, H] tiles, with tile_m = capacity. ragged_rows_alloc must cover
    # every tile READ as well as every commit -- the shard's total rows plus
    # tile_m, aligned to tile_m -- because a tail tile reads a full window.
    # Weight refills take DMA priority 1 to stay off the in-order
    # token-gather queue; Pallas allows only 0 and 1.
    _refill_prio = int(refill_priority)
    assert 0 <= _refill_prio <= 1, f"refill_priority={_refill_prio}"
    assert (comm == "none" or (do_a2a8 and pad_diet)) and lhs_indirect, \
        "ragged builds are comm='none' with lhs_indirect, or " \
        "comm='a2a8' with pad_diet and lhs_indirect"
    assert ragged_rows_alloc is not None \
        and ragged_rows_alloc % ROWBLK == 0 \
        and ragged_rows_alloc % capacity == 0, ragged_rows_alloc
    tile_m = capacity  # the tile height IS the capacity
    tblk = tile_m // ROWBLK

    # Empty-expert skip: the sweep over all G local experts becomes a traced
    # loop over the active list, so an empty expert's slab never streams.
    _skip_empty = bool(skip_empty_experts)
    if _skip_empty:
        assert ragged and do_a2a8 and pad_diet and lhs_indirect \
            and _prefetch, \
            ("skip_empty_experts is scoped to the served ragged a2a8 "
             "prefetch rotation (ragged + pad_diet + lhs_indirect + "
             "early_weight_prefetch; fp8 or fp4)")

    def kernel(*refs):
        it = iter(refs)
        if do_xport:
            (cstart_sm, clen_sm, coff_sm, crows_sm, psrc_sm, plen_sm, pdst_sm,
             tot_sm) = (next(it) for _ in range(8))
        if pad_diet:
            lc_sm, pdstr_sm, tott_sm = (next(it) for _ in range(3))
        if lhs_indirect:
            tg_sm = next(it)
            # (rows, base) i32 [G] row-unit tables from shard_tables_ragged.
            rows_sm = next(it)
            base_sm = next(it)
            if _skip_empty:
                # active_sm[e_i] is the real expert at compacted step e_i.
                active_sm = next(it)
                n_active_sm = next(it)
        # lhs_hbm is the ungathered token buffer [tokens, subq, 128].
        if do_a2a8:
            # a2a8 never ships the topk weight, so wrow is not an operand.
            (lhs_hbm, ls_hbm, w1_hbm, w2_hbm, w1s_hbm,
             w2s_hbm) = (next(it) for _ in range(6))
            wrow_hbm = None
        else:
            lhs_hbm, ls_hbm, wrow_hbm, w1_hbm, w2_hbm, w1s_hbm, w2s_hbm = (
                next(it) for _ in range(7))
        if rhs_fp4:
            # Packed-u32 weight stream: the four-bit [G, K, N] refs viewed as
            # [G, K//8, N] uint32, so the DMA moves half the bytes.
            w1_hbm = w1_hbm.bitcast(jnp.uint32)
            w2_hbm = w2_hbm.bitcast(jnp.uint32)
        if do_a2a8:
            recv_hbm = next(it)
            rscl_hbm = next(it)
            contrib_hbm = next(it)
            cscl_hbm = next(it)
        else:
            out_hbm = next(it)
        if do_a2a8:
            # a2a8 scratch drops wrow_vm; oscl_vm rides after out_vm.
            (lhs_vm, w1_vm, w2_vm, w1s_vm, w2s_vm, ls_vm,
             out_vm) = (next(it) for _ in range(7))
            wrow_vm = None
            oscl_vm = next(it)  # [2, cblk, RB, slanes] f32
        else:
            lhs_vm, w1_vm, w2_vm, w1s_vm, w2s_vm, ls_vm, wrow_vm, out_vm = (
                next(it) for _ in range(8))
        lhs_sems, w1_sems, w2_sems, cp_sem = (next(it) for _ in range(4))
        if do_a2a:
            # One commit sem per out_vm parity: with a shared sem the other
            # parity's bytes could satisfy a wait, and order is not promised.
            commit_sems, send_sem, recv_sem = (next(it) for _ in range(3))
            if do_a2a8:
                # Scale-mirror DMAs need their own sems: waits are per-buffer.
                (commit_scl_sems, send_scl_sem,
                 recv_scl_sem) = (next(it) for _ in range(3))
            if defer_mehop:
                # Never cp_sem: its start+wait users must not consume these.
                mehop_sem = next(it)
                mehop_scl_sem = next(it)
        else:
            out_sems = next(it)

        live_wire = do_a2a8
        if live_wire:
            me = lax.axis_index(AXIS)

            def full_barrier():
                """All-pairs barrier: transport peers are not only neighbors."""
                bsem = pltpu.get_barrier_semaphore()
                for i in range(ep):
                    pl.semaphore_signal(bsem,
                                        inc=1,
                                        device_id=(jnp.int32(i), ),
                                        device_id_type=pl.DeviceIdType.MESH)
                pl.semaphore_wait(bsem, ep)

                @functools.partial(pl.run_scoped,
                                   second=pltpu.SemaphoreType.REGULAR)
                def _(second):
                    for i in range(ep):
                        pl.semaphore_signal(
                            second,
                            inc=1,
                            device_id=(jnp.int32(i), ),
                            device_id_type=pl.DeviceIdType.MESH)
                    pl.semaphore_wait(second, ep)

        def sync(src, dst):
            c = pltpu.make_async_copy(src, dst, cp_sem)
            c.start()
            c.wait()

        def stream(src, vm, sems, s):
            return pltpu.make_async_copy(src, vm.at[s], sems.at[s])

        def w1_copy(t, s):
            return stream(w1_hbm.at[t], w1_vm, w1_sems, s)

        def w2_copy(t, s):
            return stream(w2_hbm.at[t], w2_vm, w2_sems, s)

        if do_xport:

            def rows_wait(sem, ref, rows):
                """Block until `rows` rows' worth of DMAs on `sem` landed."""
                pltpu.make_async_copy(ref.at[pl.ds(0, rows)],
                                      ref.at[pl.ds(0, rows)], sem).wait()

        # ---- prologue ----
        sync(w1s_hbm, w1s_vm)
        sync(w2s_hbm, w2s_vm)
        # Seed only the first DIST slots; the rest are head-issued in-loop.
        for b in range(_prefetch_dist):
            if _skip_empty:
                # Guarded on b < n_active, matching the head waits.
                @pl.when(jnp.int32(b) < n_active_sm[0])
                def _(b=b):
                    w1_copy(active_sm[b], b).start(priority=_refill_prio)
                    w2_copy(active_sm[b], b).start(priority=_refill_prio)
            else:
                w1_copy(b, b).start(priority=_refill_prio)
                w2_copy(b, b).start(priority=_refill_prio)
        if live_wire:
            full_barrier()

        def _fp4_w_of(slot):
            """(w1_of, w2_of) fp4 block accessors at weight slot `slot`."""
            # Each widens one k-block: u32 [qb/8, N] -> fp4 [qb, N] -> fp8.
            pk = rhs_qb // FP4_PACK

            def w1_of(b):
                return pltpu.bitcast(w1_vm[slot, pl.ds(b * pk, pk), :],
                                     FP4).astype(FP8)

            def w2_of(b):
                return pltpu.bitcast(w2_vm[slot, pl.ds(b * pk, pk), :],
                                     FP4).astype(FP8)

            return w1_of, w2_of

        # ---- ragged tile machinery ----
        # One-tile lookahead, double-buffered on the global tile parity g%2:
        # tile g's stream rides lhs_sems[g%2], waited once at tile g's head.
        def lhs_issue_tile(rbase, nblk, s):
            """Issue one [tile_m] tile's input stream into slot s."""

            def blk(i, _):
                for r in range(ROWBLK):
                    tok = tg_sm[rbase + i * ROWBLK + r]
                    pltpu.make_async_copy(lhs_hbm.at[tok],
                                          lhs_vm.at[s, i * ROWBLK + r],
                                          lhs_sems.at[s]).start()
                return _

            lax.fori_loop(0, nblk, blk, jnp.int32(0))
            # ls (+wrow) ride along on the same sem, a full [tile_m] window.
            pltpu.make_async_copy(ls_hbm.at[pl.ds(rbase, tile_m)], ls_vm.at[s],
                                  lhs_sems.at[s]).start()
            if not do_a2a8:
                pltpu.make_async_copy(wrow_hbm.at[pl.ds(rbase, tile_m)],
                                      wrow_vm.at[s], lhs_sems.at[s]).start()

        def lhs_ready_tile(ln, s):
            # Sems count bytes, so the split wait is exact under any landing
            # order. Interpret DMAs complete eagerly, so it waits a full tile.
            rows = tile_m if interpret else ln
            pltpu.make_async_copy(lhs_vm.at[s, pl.ds(0, rows)],
                                  lhs_vm.at[s, pl.ds(0, rows)],
                                  lhs_sems.at[s]).wait()
            pltpu.make_async_copy(ls_vm.at[s], ls_vm.at[s],
                                  lhs_sems.at[s]).wait()
            if not do_a2a8:
                pltpu.make_async_copy(wrow_vm.at[s], wrow_vm.at[s],
                                      lhs_sems.at[s]).wait()

        def out_wait_p(pp, nblk):
            """Byte-counted out_vm[pp] commit wait, in block units."""
            nb_ = tblk if interpret else nblk
            pltpu.make_async_copy(out_hbm.at[pl.ds(0, nb_)],
                                  out_hbm.at[pl.ds(0, nb_)],
                                  out_sems.at[pp]).wait()

        if do_a2a8:

            def commit_wait_a2a8(pp, nblk):
                """Wait one tile's commits on parity pp: data rows, then scales."""
                rows_wait(commit_sems.at[pp], contrib_hbm, nblk * ROWBLK)
                rows_wait(commit_scl_sems.at[pp], cscl_hbm, nblk)

            wait_p = commit_wait_a2a8
        else:
            wait_p = out_wait_p

        def commit_tile(pp, rblk, nblk):
            """Commit the tile's true rows [0, nblk*8) to the out slab at rblk."""
            # Interpret cannot discharge dynamic-size slices, so it loops.
            if interpret:

                def blk(b, _):
                    pltpu.make_async_copy(out_vm.at[pp,
                                                    b], out_hbm.at[rblk + b],
                                          out_sems.at[pp]).start()
                    return _

                lax.fori_loop(0, nblk, blk, jnp.int32(0))
            else:
                pltpu.make_async_copy(out_vm.at[pp, pl.ds(0, nblk)],
                                      out_hbm.at[pl.ds(rblk, nblk)],
                                      out_sems.at[pp]).start()

        def body_ragged(e, e_i, refill, carry):
            """One expert step: a fori_loop over its [tile_m, H] tiles."""
            # carry = (tp, r0, r1): the global tile counter plus the last
            # committed block count per out_vm parity, alternating per tile.
            # The weight slot indexes a contiguous counter so DIST + 1 of them
            # occupy distinct slots; the DMA base stays the real expert e.
            slot = lax.rem(e_i if _skip_empty else e, jnp.int32(NBUF))
            w1_copy(e, slot).wait()
            w2_copy(e, slot).wait()
            # Refill the slot for e + DIST: e - 1 last read it, and DIST < NBUF
            # keeps it off both live readers, so no wait is needed.
            if _skip_empty:

                @pl.when(e_i + _prefetch_dist < n_active_sm[0])
                def _():
                    rslot = lax.rem(e_i + _prefetch_dist, jnp.int32(NBUF))
                    en = active_sm[e_i + _prefetch_dist]
                    w1_copy(en, rslot).start(priority=_refill_prio)
                    w2_copy(en, rslot).start(priority=_refill_prio)
            else:

                @pl.when(e + _prefetch_dist < G)
                def _():
                    rslot = lax.rem(e + _prefetch_dist, jnp.int32(NBUF))
                    w1_copy(e + _prefetch_dist,
                            rslot).start(priority=_refill_prio)
                    w2_copy(e + _prefetch_dist,
                            rslot).start(priority=_refill_prio)

            rows = rows_sm[e]
            base = base_sm[e]
            nt = -(-rows // tile_m)

            def commit_tile_a2a8(pp, tb, nblk):
                """Commit the tile's intersection with each (expert, dest) run."""

                # The runs tile the expert slab contiguously, so the per-dest
                # lengths sum to exactly nblk. Data in rows, scales in blocks.

                def cst(c):
                    return c.start()

                for d in range(ep):
                    st = cstart_sm[e, d]
                    le = clen_sm[e, d]
                    lo = jnp.maximum(st, tb)
                    hi = jnp.minimum(st + le, tb + nblk)
                    lnb = jnp.maximum(hi - lo, 0)
                    src = lo - tb
                    dst = coff_sm[e, d] + (lo - st)
                    cst(
                        pltpu.make_async_copy(
                            out_vm.at[pp,
                                      pl.ds(src * ROWBLK, lnb * ROWBLK)],
                            contrib_hbm.at[pl.ds(dst * ROWBLK, lnb * ROWBLK)],
                            commit_sems.at[pp]))
                    cst(
                        pltpu.make_async_copy(oscl_vm.at[pp,
                                                         pl.ds(src, lnb)],
                                              cscl_hbm.at[pl.ds(dst, lnb)],
                                              commit_scl_sems.at[pp]))

            def tile_body(t, c):
                tp, r0, r1 = c
                rbase = base + t * tile_m
                rblk = rbase // ROWBLK
                ln = jnp.minimum(rows - t * tile_m, tile_m)
                nblk = ln // ROWBLK
                tb = t * tblk  # expert-local block base
                p = lax.rem(tp, jnp.int32(2))

                # out_vm[p] reuse guard. Predicated on pending > 0 rather than
                # tp >= 2, because group_waits_ragged zeros the counts.
                pend = jnp.where(p == 0, r0, r1)

                @pl.when(pend > 0)
                def _():
                    wait_p(p, pend)

                # Wait this tile's stream, then issue tile t+1's into the other
                # slot before compute, so the fetch runs under the MXU window.
                lhs_ready_tile(ln, p)
                q_s = jnp.int32(1) - p

                @pl.when(t + 1 < nt)
                def _():
                    ln2 = jnp.minimum(rows - (t + 1) * tile_m, tile_m)
                    lhs_issue_tile(rbase + tile_m, ln2 // ROWBLK, q_s)

                # A tile always computes its full tile_m rows; the commits
                # below span only its true rows.
                qv = lhs_vm[p].reshape(tile_m, hidden)
                ls_t = ls_vm[p]
                if rhs_fp4:
                    # The block scales apply inside _ffn_dots_pre_fp4, so the
                    # epilogues below must not apply w2s again.
                    w1s_e = w1s_vm[e]  # [nb1, 2*inter] f32
                    w2s_e = w2s_vm[e]  # [nb2, hidden] f32
                    w1_of, w2_of = _fp4_w_of(slot)
                    acc2, s2 = _ffn_dots_pre_fp4(qv,
                                                 ls_t,
                                                 w1_of,
                                                 w2_of,
                                                 w1s_e,
                                                 w2s_e,
                                                 qb=rhs_qb)
                else:
                    acc2, s2 = _ffn_dots_pre(qv, ls_t, w1_vm[slot],
                                             w2_vm[slot],
                                             w1s_vm[pl.ds(e, 1), :])
                if do_a2a8:
                    # The destination applies the router weight, not this.
                    val = ((acc2 * s2) if rhs_fp4 else
                           ((acc2 * s2) * w2s_vm[pl.ds(e, 1), :])).astype(
                               jnp.bfloat16)
                    q8, oscl = rowquant_fp8(val)
                    q8v = q8.reshape(tile_m, subq, 128)
                    osv = jnp.broadcast_to(oscl, (tile_m, slanes)).reshape(
                        tblk, ROWBLK, slanes)

                    def store_p(pp):
                        out_vm[pp] = q8v
                        oscl_vm[pp] = osv
                else:
                    # Four-bit weights carry w2s inside the block sums.
                    val = (((acc2 * s2) if rhs_fp4 else
                            (acc2 * s2 * w2s_vm[pl.ds(e, 1), :])) *
                           wrow_vm[p]).astype(jnp.bfloat16)
                    vv = val.reshape(tblk, ROWBLK, hidden)

                    def store_p(pp):
                        out_vm[pp] = vv

                # Store slots must be static, so these branches stay static.
                @pl.when(p == 0)
                def _():
                    store_p(0)

                @pl.when(p == 1)
                def _():
                    store_p(1)

                if do_a2a8:
                    commit_tile_a2a8(p, tb, nblk)
                else:
                    commit_tile(p, rblk, nblk)

                return (tp + 1, jnp.where(p == 0, nblk,
                                          r0), jnp.where(p == 1, nblk, r1))

            carry = lax.fori_loop(0, nt, tile_body, carry)
            # Cross-expert lookahead: issue the next expert's tile 0 here, so
            # its head wait finds the stream in flight. An empty next expert
            # issues nothing and its own tail issues the one after.
            tp_n = carry[0]

            if _skip_empty:

                @pl.when(e_i + 1 < n_active_sm[0])
                def _():
                    en = active_sm[e_i + 1]
                    lhs_issue_tile(base_sm[en],
                                   jnp.minimum(rows_sm[en], tile_m) // ROWBLK,
                                   lax.rem(tp_n, jnp.int32(2)))
            else:

                @pl.when(e + 1 < G)
                def _():
                    en = jnp.minimum(e + 1, G - 1)  # clamped SMEM
                    rows_n = rows_sm[en]

                    @pl.when(rows_n > 0)
                    def _():
                        lhs_issue_tile(base_sm[en],
                                       jnp.minimum(rows_n, tile_m) // ROWBLK,
                                       lax.rem(tp_n, jnp.int32(2)))

            return carry

        def group_waits_ragged(c):
            """Drain both parities' pending commits, then zero the counts."""
            # The tile reuse guards test pending > 0, so nothing double-waits.
            tp, r0, r1 = c

            @pl.when(r0 > 0)
            def _():
                wait_p(0, r0)

            @pl.when(r1 > 0)
            def _():
                wait_p(1, r1)

            return (tp, jnp.int32(0), jnp.int32(0))

        # Pipeline prologue: the first expert's tile 0, issued in flight.
        if _skip_empty:

            @pl.when(n_active_sm[0] > 0)
            def _():
                e0a = active_sm[0]
                lhs_issue_tile(base_sm[e0a],
                               jnp.minimum(rows_sm[e0a], tile_m) // ROWBLK,
                               jnp.int32(0))
        else:

            @pl.when(rows_sm[0] > 0)
            def _():
                lhs_issue_tile(base_sm[0],
                               jnp.minimum(rows_sm[0], tile_m) // ROWBLK,
                               jnp.int32(0))

        def group_push(g):
            """Push the group's remote regions and hop the own-dest region."""
            # Remote pushes go per (expert, dest) at true length lc; run starts
            # stay aligned, so recv positions are unchanged. Each is predicated
            # on its own length: a zero-length REMOTE DMA must never issue.
            for d in range(ep):
                ln = plen_sm[g, d]
                soff = psrc_sm[g, d]
                doff = pdst_sm[g, d]
                is_me = jnp.int32(d) == me

                def zp(pred, n):
                    return jnp.logical_and(pred, n > 0)

                @pl.when(zp(jnp.logical_not(is_me), ln))
                def _():
                    for k2 in range(egroup):
                        e2 = g * egroup + k2
                        ln2 = lc_sm[e2, d]

                        def _push_ed(e2=e2, ln2=ln2):
                            pltpu.make_async_remote_copy(
                                src_ref=contrib_hbm.at[pl.ds(
                                    coff_sm[e2, d] * ROWBLK, ln2)],
                                dst_ref=recv_hbm.at[pl.ds(
                                    pdstr_sm[e2, d], ln2)],
                                send_sem=send_sem,
                                recv_sem=recv_sem,
                                device_id=(jnp.int32(d), ),
                                device_id_type=pl.DeviceIdType.MESH).start()

                        # A nonempty region can hold empty (e, d) runs.
                        pl.when(ln2 > 0)(_push_ed)
                    if do_a2a8:  # scale mirror, same block offsets
                        pltpu.make_async_remote_copy(
                            src_ref=cscl_hbm.at[pl.ds(soff, ln)],
                            dst_ref=rscl_hbm.at[pl.ds(doff, ln)],
                            send_sem=send_scl_sem,
                            recv_sem=recv_scl_sem,
                            device_id=(jnp.int32(d), ),
                            device_id_type=pl.DeviceIdType.MESH).start()

                @pl.when(zp(is_me, ln))
                def _():
                    if defer_mehop:
                        # Safe to defer: recv is read only after the drain.
                        pltpu.make_async_copy(
                            contrib_hbm.at[pl.ds(soff * ROWBLK, ln * ROWBLK)],
                            recv_hbm.at[pl.ds(doff * ROWBLK, ln * ROWBLK)],
                            mehop_sem).start()
                        pltpu.make_async_copy(cscl_hbm.at[pl.ds(soff, ln)],
                                              rscl_hbm.at[pl.ds(doff, ln)],
                                              mehop_scl_sem).start()
                        return
                    sync(contrib_hbm.at[pl.ds(soff * ROWBLK, ln * ROWBLK)],
                         recv_hbm.at[pl.ds(doff * ROWBLK, ln * ROWBLK)])
                    sync(cscl_hbm.at[pl.ds(soff, ln)],
                         rscl_hbm.at[pl.ds(doff, ln)])

        def group_body(g, carry, refill_all=True):
            if _skip_empty:
                e_i = g
                e = active_sm[e_i]
                carry = body_ragged(e, e_i, refill_all, carry)
                if do_xport:
                    carry = group_waits_ragged(carry)
                    group_push(e)
                return carry
            e0 = g * egroup
            for k in range(egroup):
                refill = refill_all or (k < egroup - NBUF)
                carry = body_ragged(e0 + k, k, refill, carry)
            if do_xport:
                # Drain the pending tiles, then push the group's regions.
                carry = group_waits_ragged(carry)
                group_push(g)
            return carry

        if _skip_empty:
            # One traced loop over the compacted domain [0, n_active).
            fc = (jnp.int32(0), jnp.int32(0), jnp.int32(0))
            fc = lax.fori_loop(0, n_active_sm[0], group_body, fc)
        else:
            fc = (jnp.int32(0), jnp.int32(0), jnp.int32(0))
            if n_grp > 1:
                fc = lax.fori_loop(0, n_grp - 1, group_body, fc)
            fc = lax.fori_loop(n_grp - 1, n_grp,
                               functools.partial(group_body, refill_all=False),
                               fc)

        # ---- drain ----
        # The per-tile head waits consume every commit but the last one per
        # parity; the drains below consume those.
        if do_a2a8:
            rows_wait(send_sem, contrib_hbm, tott_sm[0])
            rows_wait(recv_sem, recv_hbm, tott_sm[1])
            rows_wait(send_scl_sem, cscl_hbm, tot_sm[0])
            rows_wait(recv_scl_sem, rscl_hbm, tot_sm[1])
            if defer_mehop:
                # Deferred own-destination drain: the total sums my own-dest
                # region lengths, and a skipped pair owes exactly zero rows.
                mh = plen_sm[0, me]
                for _g in range(1, n_grp):
                    mh = mh + plen_sm[_g, me]
                rows_wait(mehop_sem, recv_hbm, mh * ROWBLK)
                rows_wait(mehop_scl_sem, rscl_hbm, mh)
            full_barrier()
        else:
            # Tiles alternate parity, so at most the final two are pending.
            _tp, _r0, _r1 = fc

            @pl.when(_r0 > 0)
            def _():
                out_wait_p(0, _r0)

            @pl.when(_r1 > 0)
            def _():
                out_wait_p(1, _r1)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    scratch = [
        # The indirect form keeps each token row as one [subq, 128] block.
        pltpu.VMEM((NBUF, capacity, subq, 128) if lhs_indirect else
                   (NBUF, capacity, hidden), FP8),  # lhs_vm
        # rhs_fp4: w1/w2 stream in PACKED u32 words ([K/8, N]).
        (pltpu.VMEM((NBUF, hidden // FP4_PACK,
                     2 * inter), jnp.uint32) if rhs_fp4 else pltpu.VMEM(
                         (NBUF, hidden, 2 * inter), FP8)),  # w1_vm
        (pltpu.VMEM((NBUF, inter // FP4_PACK,
                     hidden), jnp.uint32) if rhs_fp4 else pltpu.VMEM(
                         (NBUF, inter, hidden), FP8)),  # w2_vm
        (pltpu.VMEM(
            (G, nb1, 2 * inter), jnp.float32) if rhs_fp4 else pltpu.VMEM(
                (G, 2 * inter), jnp.float32)),  # w1s_vm
        (pltpu.VMEM((G, nb2, hidden), jnp.float32) if rhs_fp4 else pltpu.VMEM(
            (G, hidden), jnp.float32)),  # w2s_vm
        # ls and wrow ride per tile, so they are sized by the buffer depth.
        pltpu.VMEM((NBUF, capacity, 1), jnp.float32),  # ls_vm
        pltpu.VMEM((NBUF, capacity, 1), jnp.float32),  # wrow_vm
        # The out pair is in TILE units.
        pltpu.VMEM((2, tile_m, subq, 128) if pad_diet else
                   (2, tblk, ROWBLK, hidden),
                   FP8 if do_a2a8 else jnp.bfloat16),  # out_vm
        pltpu.SemaphoreType.DMA((NBUF, )),  # lhs_sems
        pltpu.SemaphoreType.DMA((NBUF, )),  # w1_sems
        pltpu.SemaphoreType.DMA((NBUF, )),  # w2_sems
        pltpu.SemaphoreType.DMA,  # cp_sem
    ]
    if do_a2a8:
        # Drop wrow_vm (never read here) and ride oscl_vm after out_vm.
        scratch = scratch[:6] + scratch[7:8] + [
            pltpu.VMEM((2, tblk, ROWBLK, slanes), jnp.float32)  # oscl_vm
        ] + scratch[8:]
    if do_a2a8:
        scratch += [
            pltpu.SemaphoreType.DMA((2, )),  # commit_sems
            pltpu.SemaphoreType.DMA,  # send_sem
            pltpu.SemaphoreType.DMA,  # recv_sem
            pltpu.SemaphoreType.DMA((2, )),  # commit_scl_sems
            pltpu.SemaphoreType.DMA,  # send_scl_sem
            pltpu.SemaphoreType.DMA
        ]  # recv_scl_sem
        if defer_mehop:  # deferred own-destination hop
            scratch += [
                pltpu.SemaphoreType.DMA,  # mehop_sem
                pltpu.SemaphoreType.DMA
            ]  # mehop_scl_sem
        n_prefetch = (
            8 + (3 if pad_diet else 0)
            # tg
            + (1 if lhs_indirect else 0)
            # (rows, base)
            + 2
            # (active, n_active)
            + (2 if _skip_empty else 0))
    else:
        scratch += [pltpu.SemaphoreType.DMA((2, ))]  # out_sems
        n_prefetch = 3 if lhs_indirect else 0

    def make_call(recv_rows=None):
        if do_a2a8:
            assert recv_rows % ROWBLK == 0
            # contrib and cscl are sized for the no-drop worst case.
            c_rows = ragged_rows_alloc
            out_shape = [
                jax.ShapeDtypeStruct((recv_rows, subq, 128) if pad_diet else
                                     (recv_rows // ROWBLK, ROWBLK, hidden),
                                     FP8),
                jax.ShapeDtypeStruct((recv_rows // ROWBLK, ROWBLK, slanes),
                                     jnp.float32),
                jax.ShapeDtypeStruct((c_rows, subq, 128) if pad_diet else
                                     (G * cblk, ROWBLK, hidden), FP8),
                jax.ShapeDtypeStruct((c_rows // ROWBLK, ROWBLK, slanes),
                                     jnp.float32),
            ]
        else:
            # Dynamic-offset commits are block-granular on the leading dim.
            out_shape = [
                jax.ShapeDtypeStruct(
                    (ragged_rows_alloc // ROWBLK, ROWBLK, hidden),
                    jnp.bfloat16)
            ]
        name = (f"fused_ep_moe_v2_{comm}"
                f"{'_gather' if lhs_indirect else ''}"
                f"{'_ragged' if ragged else ''}"
                f"{'_truelen' if pad_diet else ''}"
                f"{'_defermehop' if defer_mehop else ''}"
                f"{'_fp4w' if rhs_fp4 else ''}"
                f"_g{G}_c{capacity}_nb{NBUF}")
        n_in = 6 if do_a2a8 else 7  # a2a8 drops the wrow HBM operand
        return pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=n_prefetch,
                in_specs=[hbm] * n_in,
                out_specs=[hbm] * len(out_shape),
                scratch_shapes=tuple(scratch),
                grid=()),
            out_shape=out_shape,
            compiler_params=pltpu.CompilerParams(
                # Shared with any other collective kernel using the same id;
                # a caller that runs two of them concurrently passes its own.
                collective_id=(collective_id if do_a2a else None),
                vmem_limit_bytes=(None if interpret else vmem_limit()),
                # Bounds checks cost time and change no computed value.
                disable_bounds_checks=(False if interpret else True)),
            interpret=interpret,
            name=name,
        )

    if do_a2a8 and lhs_indirect:

        def fn(tables,
               tg,
               rows,
               base,
               qg,
               ls,
               w1,
               w2,
               w1s,
               w2s,
               *,
               recv_rows,
               active=None,
               n_active=None):
            """The served ragged transport form; returns arrivals and scales."""
            # tables = the shard_tables and shard_tables_diet tuples,
            # (rows, base) from shard_tables_ragged, ls the [rows_alloc] slab.
            call = make_call(recv_rows)
            pre = [
                tg.astype(jnp.int32),
                rows.astype(jnp.int32),
                base.astype(jnp.int32)
            ]
            if _skip_empty:
                pre += [active.astype(jnp.int32), n_active.astype(jnp.int32)]
            res = call(*tables, *pre, qg.reshape(-1, subq, 128),
                       ls.reshape(ragged_rows_alloc, 1), w1, w2,
                       w1s.astype(jnp.float32), w2s.astype(jnp.float32))
            recv, rscl, _c, _cs = res
            return recv, rscl.reshape(recv_rows, slanes)
    else:

        def fn(tg, rows, base, qg, ls, wrow, w1, w2, w1s, w2s):
            """The ragged comm='none' arm; returns [rows_alloc, hidden] bf16."""
            # Row r of expert e lands at base[e] + r; gap rows hold garbage.
            call = make_call()
            res = call(tg.astype(jnp.int32), rows.astype(jnp.int32),
                       base.astype(jnp.int32), qg.reshape(-1, subq, 128),
                       ls.reshape(ragged_rows_alloc, 1),
                       wrow.reshape(ragged_rows_alloc, 1), w1, w2,
                       w1s.astype(jnp.float32), w2s.astype(jnp.float32))
            return res[0].reshape(ragged_rows_alloc, hidden)

    return fn


# Pallas keys its kernel-to-jaxpr cache on the `kernel` object itself, so
# memoizing the build is what makes pallas hit across builds.
_BUILD_CACHE = {}
# Two threads missing on the same key would each build a kernel, destroying
# the object identity this cache holds, so the miss path is serialized.
_BUILD_CACHE_LOCK = threading.Lock()


def build_fused_ep_moe_kernel(**kwargs):
    """Memoizing front door for _build_fused_ep_moe_kernel."""
    key = tuple(sorted(kwargs.items()))
    fn = _BUILD_CACHE.get(key)
    if fn is None:
        with _BUILD_CACHE_LOCK:
            fn = _BUILD_CACHE.get(key)
            if fn is None:
                fn = _build_fused_ep_moe_kernel(**kwargs)
                _BUILD_CACHE[key] = fn
    return fn
