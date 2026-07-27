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
"""Serving adapter for the fused expert-parallel MoE kernel: moe_apply
routes a call here once the token count reaches MOE_FUSED_EP_MIN_TOKENS
and unsupported_reason accepts it.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference import envs
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.utils import get_mesh_shape_product

# _ROWBLK mirrors the kernel's transport row block (re-asserted against
# the kernel at import); _TILE_M is its tile height and keys both the
# capacity argument and the ragged-stride worst case below.
_ROWBLK = 8
_TILE_M = 128
# DMA priority for the weight refills, passed through to the kernel layer.
_REFILL_PRIORITY = 1

_kernel = None
# Resolved from the kernel's own constants at import, so no literal here
# can go stale.
_fp4_packed_row_tile = None
_vmem_estimate = None
_vmem_budget = None
_weight_buffers = None
_hidden_lane_block = None
_hidden_max_blocks = None


def _import_kernel():
    """Resolve the in-tree kernel entry point; raises ImportError."""
    global _kernel, _fp4_packed_row_tile, _vmem_estimate, _vmem_budget
    global _weight_buffers, _hidden_lane_block, _hidden_max_blocks
    if _kernel is not None:
        return _kernel
    try:
        from tpu_inference.kernels import fused_ep_moe as kernel_package
    except ImportError as e:
        raise ImportError(
            f"fused EP MoE: importing the fused_ep_moe kernel package "
            f"failed ({e}). There is no fallback; either fix the tree or "
            "raise MOE_FUSED_EP_MIN_TOKENS above the served token counts."
        ) from e
    assert kernel_package.ROWBLK == _ROWBLK, (
        f"kernel ROWBLK={kernel_package.ROWBLK} != adapter's {_ROWBLK}; the "
        "ragged-stride worst-case arithmetic below would be wrong")
    assert kernel_package.AXIS == "d", kernel_package.AXIS
    _fp4_packed_row_tile = kernel_package.ROWBLK * kernel_package.FP4_PACK
    _vmem_estimate = kernel_package.vmem_estimate_bytes
    _vmem_budget = kernel_package.vmem_limit
    _weight_buffers = kernel_package.NBUF
    _hidden_lane_block = kernel_package.HIDDEN_LANE_BLOCK
    _hidden_max_blocks = kernel_package.HIDDEN_MAX_BLOCKS
    _kernel = kernel_package.fused_ep_moe_v2
    return _kernel


_PROVEN_MESH_AXES = ('data', 'attn_dp', 'attn_dp_expert', 'expert', 'model',
                     'dcp')


def _mesh_reason(mesh: Mesh) -> str | None:
    """Why this serving mesh cannot be re-wrapped as one axis, or None."""
    ndev = mesh.devices.size
    names = tuple(mesh.axis_names)
    # Any axis outside this list is accepted only when degenerate: a
    # size-1 axis cannot permute the flattened device order.
    if tuple(n for n in names if n in _PROVEN_MESH_AXES) != _PROVEN_MESH_AXES:
        return (f"mesh axes {names} do not contain {_PROVEN_MESH_AXES} in "
                "that order")
    for name in names:
        if name not in _PROVEN_MESH_AXES and mesh.shape[name] != 1:
            return (f"mesh axis {name!r} has size {mesh.shape[name]} != 1; "
                    f"every axis outside {_PROVEN_MESH_AXES} must be "
                    "degenerate for the re-wrap to preserve device order")
    if mesh.shape['data'] != 1:
        return (f"mesh data axis size {mesh.shape['data']} != 1; with "
                "data > 1 the expert-shard order diverges from the flat "
                "device order and the re-wrap would permute shards")
    attn_dp = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_DATA)
    if attn_dp != ndev:
        return (f"attention is not pure DP over all devices "
                f"(enable_dp_attention): ATTN_DATA product {attn_dp} != "
                f"device count {ndev}")
    return None


def _single_axis_mesh(mesh: Mesh) -> Mesh:
    """The kernel-layer mesh: same devices, one axis named 'd'."""
    reason = _mesh_reason(mesh)
    if reason is not None:
        raise ValueError(f"fused EP MoE: {reason}")
    return Mesh(mesh.devices.reshape(-1), ("d", ))


def _ragged_stride(num_tokens: int, topk: int, num_experts: int) -> int:
    """Static per-shard ragged slab rows: the no-drop worst case."""
    pads = (_ROWBLK - 1) * num_experts
    need = num_tokens * topk + pads + _TILE_M
    return -(-need // _TILE_M) * _TILE_M


def _plan_block(num_tokens: int, topk: int, ep: int) -> int:
    """Largest power-of-two plan block <= 256 dividing t_local*topk."""
    b = 256
    while ((num_tokens // ep) * topk) % b:
        b //= 2
    return b


def unsupported_reason(layer,
                       x: jax.Array,
                       gating_output,
                       weights,
                       mesh: Mesh,
                       activation: str,
                       scatter_results: bool,
                       extra_backend_kwargs: dict | None = None,
                       defer_all_reduce: bool = False) -> str | None:
    """Why this MoE call cannot run on the fused EP kernel, or None; every
    condition is a trace-time constant and the answer is never an
    exception, so the caller can pick the path at compile time.
    """
    try:
        _import_kernel()
    except ImportError as e:
        return f"the fused EP MoE kernel is not importable here: {e}"

    num_tokens, hidden = x.shape
    num_experts, w_hidden, two_inter = weights.w13_weight.shape
    inter = two_inter // 2

    if not isinstance(gating_output, jax.Array):
        return (f"gating output is {type(gating_output).__name__}; the "
                "kernel routes from one logits array")
    if not getattr(layer, "use_ep", False):
        return "the kernel is expert-parallel and layer.use_ep is False"
    if not scatter_results:
        return ("the kernel layer ends with each rank holding its own token "
                "rows, which is scatter_results semantics; the caller asked "
                "for the all-reduced form")
    if layer.scoring_func != "softmax":
        return (f"kernel routing is softmax + top_k; got scoring_func="
                f"{layer.scoring_func!r}")
    if activation != "silu":
        return (f"kernel FFN math is silu(gate) * up; got activation="
                f"{activation!r}")
    if weights.w13_bias is not None or weights.w2_bias is not None:
        return "the kernel has no MoE bias operands"
    if defer_all_reduce:
        return ("the caller asked for defer_all_reduce, which returns "
                "per-shard partial sums; the kernel combines rows to their "
                "token owners inside itself and has no partial-sum output")

    # Routing modifiers fused_moe_func honours and this kernel does not:
    # each one changes which experts a token goes to.
    kw = extra_backend_kwargs or {}
    for name, value in (("hash_based_topk_indices",
                         kw.get("hash_based_topk_indices")),
                        ("e_score_correction_bias",
                         kw.get("e_score_correction_bias")),
                        ("num_valid_tokens", kw.get("num_valid_tokens"))):
        if value is not None:
            return (f"the caller passed {name}, which selects experts (or "
                    "gates rows) differently from the kernel's own softmax "
                    "top-k; the kernel has no operand for it")
    if envs.MOE_APPROX_TOPK:
        return ("MOE_APPROX_TOPK asks for approximate top-k selection; the "
                "kernel selects exactly")

    # Which of the two weight forms this is, decided by the expert weight
    # dtype. Every other dtype is refused, never reinterpreted.
    rhs_fp4 = weights.w13_weight.dtype == jnp.float4_e2m1fn
    if not rhs_fp4 and weights.w13_weight.dtype != jnp.float8_e4m3fn:
        return (f"kernel weights must be fp8 e4m3 or fp4 e2m1, got "
                f"{weights.w13_weight.dtype}")
    if weights.w2_weight.dtype != weights.w13_weight.dtype:
        return (f"both expert weights must carry one dtype; w13 is "
                f"{weights.w13_weight.dtype} and w2 is "
                f"{weights.w2_weight.dtype}")
    if w_hidden != hidden:
        return (f"padded hidden ({w_hidden}) != activation hidden "
                f"({hidden}); the kernel takes unpadded-hidden weights only")
    # The transport stages a token row as a whole number of 128-lane
    # blocks, and its row staging holds a bounded number of them.
    if hidden % _hidden_lane_block != 0:
        return (f"hidden {hidden} is not a whole number of "
                f"{_hidden_lane_block}-lane blocks, which the kernel's "
                "per-row transport geometry requires")
    if hidden > _hidden_lane_block * _hidden_max_blocks:
        return (f"hidden {hidden} is wider than the "
                f"{_hidden_lane_block * _hidden_max_blocks} the kernel's "
                "row staging holds")

    s13, s2 = weights.w13_weight_scale, weights.w2_weight_scale
    if s13 is None or s2 is None:
        return "the kernel requires quantized MoE weights (scales present)"
    if rhs_fp4:
        # Derive the block size from the w13 scale shape, then require w2
        # to carry the same one -- shape-verified, never assumed.
        if weights.w2_weight.shape != (num_experts, inter, hidden):
            return (f"fp4 w2 shape {weights.w2_weight.shape} != "
                    f"{(num_experts, inter, hidden)}")
        if not (s13.ndim == 4 and s13.shape[0] == num_experts
                and s13.shape[2] == 1 and s13.shape[3] == two_inter):
            return (f"fp4 w13 scale layout {s13.shape} is not the block "
                    f"form (E, hidden//qb, 1, 2*inter) = ({num_experts}, "
                    f"blocks, 1, {two_inter})")
        if hidden % s13.shape[1] != 0:
            return (f"fp4 hidden {hidden} is not divisible by the w13 scale "
                    f"block count {s13.shape[1]}")
        rhs_qb = hidden // s13.shape[1]
        if inter % rhs_qb != 0:
            return (f"fp4 intermediate size {inter} is not divisible by the "
                    f"derived block size {rhs_qb}; the kernel blocks BOTH "
                    "matmuls at one block size")
        if s2.shape != (num_experts, inter // rhs_qb, 1, hidden):
            return (f"fp4 w2 scale layout {s2.shape} != "
                    f"{(num_experts, inter // rhs_qb, 1, hidden)} -- both "
                    f"matmuls must carry the same block size {rhs_qb}, and "
                    "a mismatch is refused, never resampled")
        if rhs_qb % _fp4_packed_row_tile != 0:
            return (f"fp4 block size {rhs_qb} is not a whole number of the "
                    f"kernel's packed-weight row tile "
                    f"({_fp4_packed_row_tile} rows)")
    else:
        rhs_qb = None
        if s13.shape != (num_experts, 1, 1, two_inter):
            return (f"w13 scale layout {s13.shape} is not the per-channel "
                    f"form {(num_experts, 1, 1, two_inter)} the kernel "
                    "consumes")
        if s2.shape != (num_experts, 1, 1, hidden):
            return (f"w2 scale layout {s2.shape} != "
                    f"{(num_experts, 1, 1, hidden)}")

    reason = _mesh_reason(mesh)
    if reason is not None:
        return reason
    ep = mesh.devices.size
    if num_experts % ep != 0:
        return (f"expert count {num_experts} is not divisible by the "
                f"expert-parallel width {ep}")
    if num_tokens % ep != 0:
        return (f"token count {num_tokens} is not divisible by the "
                f"expert-parallel width {ep}")
    # The routing plan packs a per-shard position and an alignment slot
    # into one word as position * 64 + slot. The slot runs up to
    # (ROWBLK - 1) * (ep - 1), so past ep = 10 it overflows 64 and the
    # two fields corrupt each other silently.
    if (_ROWBLK - 1) * (ep - 1) >= 64:
        return (f"expert-parallel width {ep} needs up to "
                f"{(_ROWBLK - 1) * (ep - 1)} alignment slots and the "
                "routing plan packs them into a 64-wide field; the kernel "
                f"holds widths up to {1 + 63 // (_ROWBLK - 1)}")
    topk = int(layer.top_k)
    if topk < 2:
        return (f"top_k is {topk}; the kernel's NaN-score guard zeroes such "
                "a row by summing at least two sentinel weights to -inf, "
                "which one selection slot cannot do")
    if not bool(getattr(layer, "renormalize", False)):
        return ("the layer does not renormalize the selected weights; the "
                "kernel's NaN-score guard zeroes such a row through that "
                "renormalization and has no other defence for it")
    if _plan_block(num_tokens, topk, ep) < 8:
        return (f"the routing plan needs a block of at least 8 dividing "
                f"{(num_tokens // ep) * topk} rows per shard; tokens="
                f"{num_tokens}, top_k={topk}, ep={ep} admits none")
    # VMEM fit, in the kernel's own arithmetic: the buffered weight slabs
    # dominate and do not shrink with the token count.
    est = _vmem_estimate(num_experts // ep,
                         _TILE_M,
                         hidden,
                         inter,
                         nbuf=_weight_buffers,
                         rhs_fp4=rhs_fp4,
                         rhs_qb=rhs_qb or hidden)
    try:
        budget = _vmem_budget()
    except Exception as e:  # no device record to read the capacity from
        return (f"the kernel's VMEM budget cannot be read here ({e}); it "
                "is the capacity of the chip the call is being built for, "
                "and a host that can name no chip cannot run the kernel")
    if est > budget:
        return (f"the kernel's buffers need {est / 2**20:.1f}MiB of VMEM "
                f"for {num_experts // ep} local experts of "
                f"{hidden}x{inter}, over the {budget / 2**20:.1f}MiB budget")
    return None


def moe_fused_ep_apply(
    layer,
    x: jax.Array,
    gating_output: jax.Array,
    weights,
    mesh: Mesh,
    activation: str,
    scatter_results: bool,
) -> jax.Array:
    """Run one MoE call through the fused expert-parallel kernel layer.

    Not bit-identical to the general MoE path: each expert's output rows
    cross the wire as fp8 e4m3 with one f32 scale per row, where
    fused_moe_func carries them in bf16.
    """
    kernel = _import_kernel()
    reason = unsupported_reason(layer, x, gating_output, weights, mesh,
                                activation, scatter_results)
    if reason is not None:
        raise ValueError(f"fused EP MoE: {reason}")

    num_tokens, hidden = x.shape
    num_experts = weights.w13_weight.shape[0]
    topk = int(layer.top_k)
    s13, s2 = weights.w13_weight_scale, weights.w2_weight_scale
    rhs_fp4 = weights.w13_weight.dtype == jnp.float4_e2m1fn
    rhs_qb = hidden // s13.shape[1] if rhs_fp4 else None

    mesh1 = _single_axis_mesh(mesh)
    ep = mesh1.devices.size

    stride = _ragged_stride(num_tokens, topk, num_experts)
    block = _plan_block(num_tokens, topk, ep)

    # The router runs on f32 logits, so softmax and top-k tie behavior
    # follow the logit dtype rather than the activation dtype.
    gating_f32 = gating_output.astype(jnp.float32)

    # jax refuses a shard_map whose mesh differs from the ambient context,
    # so the trace enters the single-axis mesh for the kernel call alone.
    fp4_kwargs = {"rhs_fp4": True, "rhs_qb": rhs_qb} if rhs_fp4 else {}
    with jax.sharding.use_abstract_mesh(mesh1.abstract_mesh):
        out, _stride_over = kernel(x,
                                   weights.w13_weight,
                                   weights.w2_weight,
                                   s13,
                                   s2,
                                   gating_f32,
                                   topk=topk,
                                   renormalize=bool(layer.renormalize),
                                   mesh=mesh1,
                                   capacity=_TILE_M,
                                   block=block,
                                   ragged_stride=stride,
                                   refill_priority=_REFILL_PRIORITY,
                                   **fp4_kwargs)
    # The overflow counter is structurally zero at this stride, so it is
    # dropped. The result re-tags onto the serving mesh.
    return jax.lax.with_sharding_constraint(
        out, NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None)))
