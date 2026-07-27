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
"""Serving layer for the fused expert-parallel MoE kernel.

fused_ep_moe wraps the kernel builder in a shard_map over the
expert-parallel mesh axis: quantize, dispatch, kernel, combine.
"""
import threading

import jax
import jax.numpy as jnp
from jax import lax

from tpu_inference.kernels.fused_ep_moe.fused_ep_moe_v2 import (
    FP4_QB, ROWBLK, active_list, align_up, build_fused_ep_moe_kernel,
    plan_ragged_dispatch, rowquant_fp8, shard_tables, shard_tables_diet,
    shard_tables_ragged)
from tpu_inference.kernels.fused_ep_moe.router_ops import pallas_select


def _combine_arrivals(recv3, aux, pos, val, tw, out_dtype, _terms=False):
    """Combine arrivals [rows, subq, 128] into [t_local, hidden]."""
    t_local, topk = pos.shape
    _, subq, lanes = recv3.shape
    hidden = subq * lanes
    gik = jnp.where(val, pos, 0).T.reshape(-1)  # k-major [K*t_local]
    g8 = recv3[gik]  # f8 [K*t, subq, 128]
    sg = aux[gik][:, :1]  # f32 row scales
    twf = tw.astype(jnp.float32)
    terms = []
    for k in range(topk):
        rk = lax.slice(g8, (k * t_local, 0, 0),
                       ((k + 1) * t_local, subq, lanes))
        sk = lax.slice(sg, (k * t_local, 0), ((k + 1) * t_local, 1))
        term = (rk.astype(jnp.float32) * sk[:, :, None] *
                twf[:, k:k + 1, None])
        terms.append(jnp.where(val[:, k:k + 1, None], term, 0.0))
    if _terms:
        return jnp.stack(terms, axis=1).reshape(t_local, topk, hidden)
    while len(terms) > 1:  # pairwise tree
        terms = [
            terms[i] + terms[i + 1] if i + 1 < len(terms) else terms[i]
            for i in range(0, len(terms), 2)
        ]
    return terms[0].astype(out_dtype).reshape(t_local, hidden)


def _pack_routing_blob3(ti, tw, sc):
    """Fold idx [t, K], weights [t, K] and scales [t, 1] into [t, 2K+1] i32."""
    return jnp.concatenate([
        ti.astype(jnp.int32),
        lax.bitcast_convert_type(tw.astype(jnp.float32), jnp.int32),
        lax.bitcast_convert_type(sc.astype(jnp.float32), jnp.int32)
    ],
                           axis=1)


def _unpack_routing_blob3(blob_g, topk):
    """Inverse of _pack_routing_blob3 (bit-exact round trip)."""
    ti_g = blob_g[:, :topk]
    tw_g = lax.bitcast_convert_type(blob_g[:, topk:2 * topk], jnp.float32)
    sc_g = lax.bitcast_convert_type(blob_g[:, 2 * topk:], jnp.float32)
    return ti_g, tw_g, sc_g


# Per-config cache of the shard_map'd MoE callable; a fresh shard_map
# body would re-trace the whole kernel for every hidden layer.
_LAYER_SM_CACHE = {}
_LAYER_SM_CACHE_LOCK = threading.Lock()


def fused_ep_moe(x,
                 w1,
                 w2,
                 w1_scale,
                 w2_scale,
                 gating,
                 *,
                 topk,
                 renormalize,
                 mesh,
                 capacity,
                 block=256,
                 ragged=True,
                 ragged_stride=None,
                 egroup=None,
                 defer_own_hop=True,
                 routing_blob=False,
                 rhs_fp4=False,
                 rhs_qb=None,
                 refill_priority=1,
                 early_weight_prefetch=True,
                 pallas_router=None,
                 skip_empty_experts=False,
                 occ_sort=None,
                 collective_id=0):
    """Run one MoE layer through the fused expert-parallel kernel.

    x [tokens, hidden], w1 [experts, hidden, 2 * inter], w2 [experts,
    inter, hidden] and gating [tokens, experts], all sharded over the
    mesh axis; w1_scale and w2_scale are per channel for fp8 e4m3 and per
    contraction block for fp4 e2m1. Returns ([tokens, hidden] sharded
    like x, the plan's slab-overflow counter).

    The keyword arguments pin the kernel's schedule. The constrained ones
    are block (must divide tokens // ep * topk), ragged_stride (must
    reach the no-drop bound asserted below) and occ_sort (needs
    skip_empty_experts).
    """
    if egroup not in (None, 1):
        raise NotImplementedError(
            f"egroup={egroup} is not supported: the push is always "
            "per expert. Drop the egroup argument, or pass 1.")
    rhs_qb = FP4_QB if rhs_qb is None else int(rhs_qb)
    if not ragged:
        raise NotImplementedError(
            "only the ragged form is built; pass ragged=True")
    T, hidden = x.shape
    e_total = w1.shape[0]
    inter = w1.shape[2] // 2
    (ax, ) = mesh.axis_names
    ep = mesh.shape[ax]
    g_local = e_total // ep
    t_local = T // ep
    P = jax.sharding.PartitionSpec
    # No-drop bound: every routed row landing on one shard, plus the
    # row-block alignment padding and one tile-height tail-read window.
    stride_bound = align_up(T * topk + (ROWBLK - 1) * g_local * ep + capacity,
                            capacity)
    if ragged_stride is None or ragged_stride % capacity != 0:
        raise ValueError(
            f"ragged_stride must be a multiple of capacity {capacity}; got "
            f"{ragged_stride}. Pass {stride_bound}.")
    if ragged_stride < stride_bound:
        raise ValueError(
            f"ragged_stride {ragged_stride} is below the no-drop bound "
            f"{stride_bound}, so one shard's slab could bleed into the "
            f"next. Pass {stride_bound} or more.")
    _skip_empty = bool(skip_empty_experts)
    if occ_sort is not None:
        if occ_sort not in ("desc", "asc", "ilv"):
            raise ValueError(
                f"occ_sort={occ_sort!r} -- expected None, 'desc', 'asc', "
                "or 'ilv'")
        if not _skip_empty:
            raise ValueError(
                "occ_sort requires skip_empty_experts=True (it reorders "
                "the active-expert list the compaction builds)")
    kfn = build_fused_ep_moe_kernel(
        g_local=g_local,
        capacity=capacity,
        hidden=hidden,
        inter=inter,
        ep=ep,
        comm="a2a8",
        lhs_indirect=True,
        pad_diet=True,
        ragged=True,
        ragged_rows_alloc=ragged_stride,
        defer_own_hop=defer_own_hop,
        rhs_fp4=rhs_fp4,
        rhs_qb=rhs_qb,
        refill_priority=refill_priority,
        early_weight_prefetch=early_weight_prefetch,
        skip_empty_experts=skip_empty_experts,
        collective_id=collective_id)

    def local_fn(x_l, w1_l, w2_l, s1_l, s2_l, gate_l):
        me = lax.axis_index(ax)
        q_l, sc_l = rowquant_fp8(x_l.astype(jnp.bfloat16))
        q_g = lax.all_gather(q_l.reshape(t_local, hidden // 128, 128),
                             ax,
                             axis=0,
                             tiled=True)
        if not routing_blob:
            sc_g = lax.all_gather(sc_l, ax, axis=0, tiled=True)

        scores = jax.nn.softmax(gate_l, axis=-1)
        if pallas_router:
            _br = 256
            while t_local % _br:
                _br //= 2
            tw, ti = pallas_select(scores, topk=topk, block_rows=_br)
        else:
            tw, ti = lax.top_k(scores, topk)
        if renormalize:
            tw = tw / tw.sum(axis=-1, keepdims=True)
        if routing_blob:
            blob_g = lax.all_gather(_pack_routing_blob3(ti, tw, sc_l),
                                    ax,
                                    axis=0,
                                    tiled=True)
            ti_g, tw_g, sc_g = _unpack_routing_blob3(blob_g, topk)
        else:
            ti_g = lax.all_gather(ti.astype(jnp.int32), ax, axis=0, tiled=True)
            tw_g = lax.all_gather(tw.astype(jnp.float32),
                                  ax,
                                  axis=0,
                                  tiled=True)

        plan = plan_ragged_dispatch(ti_g,
                                    tw_g,
                                    e_total=e_total,
                                    ep=ep,
                                    t_local=t_local,
                                    block=block,
                                    tile_m=capacity,
                                    shard_stride=ragged_stride)
        tabs = shard_tables(plan,
                            me,
                            e_total=e_total,
                            ep=ep,
                            capacity=capacity)
        dtabs = shard_tables_diet(plan, me, e_total=e_total, ep=ep)
        tg = lax.dynamic_slice(plan["token_gather"], (me * ragged_stride, ),
                               (ragged_stride, ))
        # a2a8 never ships the topk weight to the kernel; the destination
        # combine applies it in f32 from `tw` below.
        sc_bits = lax.bitcast_convert_type(jnp.repeat(sc_g[:, 0], topk),
                                           jnp.int32)
        ls_full = lax.bitcast_convert_type(
            jnp.zeros(
                (ep * ragged_stride + 1, ),
                jnp.int32).at[plan["didx"]].add(sc_bits,
                                                mode="promise_in_bounds")[:-1],
            jnp.float32)
        ls = lax.dynamic_slice(ls_full, (me * ragged_stride, ),
                               (ragged_stride, ))[:, None]
        rows_g, base_g = shard_tables_ragged(plan, me, e_total=e_total, ep=ep)
        r_stat = align_up(t_local * topk + (ROWBLK - 1) * e_total, ROWBLK)
        # fp4 block scales [E, nb, 1, N] reshape to the kernel's
        # [G, nb, N]; fp8 keeps the per-channel [G, N] form.
        s1k = (s1_l.reshape(g_local, -1, 2 * inter)
               if rhs_fp4 else s1_l.reshape(g_local, 2 * inter))
        s2k = (s2_l.reshape(g_local, -1, hidden) if rhs_fp4 else s2_l.reshape(
            g_local, hidden))
        _se_kw = {}
        if _skip_empty:
            # tile_m=capacity: on the ragged path the plan tile height is
            # the layer capacity, which is the boundary 'ilv' splits on.
            active_g, n_active_g = active_list(rows_g,
                                               g_local,
                                               occ_sort,
                                               tile_m=capacity)
            _se_kw = dict(active=active_g, n_active=n_active_g)

        recv3, aux = kfn((*tabs, *dtabs),
                         tg,
                         rows_g,
                         base_g,
                         q_g,
                         ls,
                         w1_l,
                         w2_l,
                         s1k,
                         s2k,
                         recv_rows=r_stat,
                         **_se_kw)

        # dest routing tables (pos = per-token topk arrival row, val = mask).
        pos = lax.dynamic_slice(plan["pos"], (me * t_local, 0),
                                (t_local, topk))
        val = lax.dynamic_slice(plan["valid"].reshape(T, topk),
                                (me * t_local, 0), (t_local, topk))
        out = _combine_arrivals(recv3, aux, pos, val, tw, x_l.dtype)
        return out, plan["stride_over"]

    in_specs = (P(ax), P(ax), P(ax), P(ax), P(ax), P(ax))
    args = (x, w1, w2, w1_scale, w2_scale, gating)
    NS = jax.sharding.NamedSharding
    args = tuple(
        jax.device_put(a, NS(mesh, sp)) for a, sp in zip(args, in_specs))
    # local_fn closes over config-static values only -- the per-call data
    # are the shard_map arguments -- so this key is exact.
    key = (mesh, T, hidden, e_total, inter, topk, bool(renormalize), capacity,
           block, ragged_stride, bool(defer_own_hop), bool(routing_blob),
           bool(rhs_fp4), rhs_qb, int(refill_priority),
           bool(early_weight_prefetch), bool(pallas_router), bool(_skip_empty),
           occ_sort, x.dtype, w1.dtype, w1_scale.dtype, gating.dtype)
    sm = _LAYER_SM_CACHE.get(key)
    if sm is None:
        with _LAYER_SM_CACHE_LOCK:
            sm = _LAYER_SM_CACHE.get(key)
            if sm is None:
                sm = jax.shard_map(local_fn,
                                   mesh=mesh,
                                   in_specs=in_specs,
                                   out_specs=(P(ax), P()),
                                   check_vma=False)
                _LAYER_SM_CACHE[key] = sm
    return sm(*args)
