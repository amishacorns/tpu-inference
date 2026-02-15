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
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.megablox.gmm import GroupMetadata, gmm, make_group_metadata
from tpu_inference.kernels.megablox.gmm_gather import gmm_gather
from tpu_inference.kernels.megablox.tuned_block_sizes import get_tuned_block_sizes
from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import get_device_vmem_limit
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
    num_total_groups = group_sizes.shape[0]

    quant_block_size = (k // rhs_scale.shape[1]) if rhs_scale is not None else k

    tiling = get_tuned_block_sizes(
        m=m, k=k, n=n,
        num_total_groups=num_total_groups,
        num_current_groups=g,
        lhs_dtype=str(lhs.dtype),
        rhs_dtype=str(rhs.dtype),
        quant_block_size=quant_block_size,
    )

    gmm_res = gmm(
        lhs=lhs,
        rhs=rhs,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_sizes=group_sizes,
        preferred_element_type=lhs.dtype,
        tiling=tiling,
        group_offset=group_offset[0],
        vmem_limit_bytes=get_device_vmem_limit(),
        group_metadata=group_metadata,
        num_active_tiles=num_active_tiles,
    )

    return gmm_res


def gmm_gather_wrapper(lhs, token_indices, rhs, rhs_scale, rhs_bias,
                       group_sizes, group_offset, group_metadata=None,
                       num_active_tiles=None):
    """Like gmm_wrapper but uses gmm_gather to fuse the token permute."""
    m = token_indices.shape[0]
    g, k, n = rhs.shape
    num_total_groups = group_sizes.shape[0]

    quant_block_size = (k // rhs_scale.shape[1]) if rhs_scale is not None else k

    tiling = get_tuned_block_sizes(
        m=m, k=k, n=n,
        num_total_groups=num_total_groups,
        num_current_groups=g,
        lhs_dtype=str(lhs.dtype),
        rhs_dtype=str(rhs.dtype),
        quant_block_size=quant_block_size,
    )

    return gmm_gather(
        lhs=lhs,
        token_indices=token_indices,
        rhs=rhs,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_sizes=group_sizes,
        preferred_element_type=lhs.dtype,
        tiling=tiling,
        group_offset=group_offset[0],
        vmem_limit_bytes=get_device_vmem_limit(),
        group_metadata=group_metadata,
        num_active_tiles=num_active_tiles,
    )


def _gmm_compute_local(
    hidden_states: jax.Array,
    token_indices: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    *,
    activation: str,
    zero_w2_bias: bool = False,
) -> jax.Array:
    """Pure compute: GMM1 (gate+up) -> activation -> GMM2 (down).

    No routing, no reduction — just the matmuls on pre-sorted tokens.
    Factored out so both TP and EP paths can call it.

    GMM1 uses gmm_gather to fuse the token permute (avoids materializing
    the permuted hidden_states tensor in HBM).
    GMM2 uses regular gmm since its LHS is the activation output which
    is already in expert-sorted order.

    Args:
        hidden_states: UN-PERMUTED activations [num_tokens, hidden_size].
        token_indices: Sorted token indices [m] (m = num_tokens * topk).
        w1: Fused gate+up weights [local_experts, 2*intermediate, hidden_size].
        w2: Down-projection weights [local_experts, hidden_size, intermediate].
        group_sizes: Per-expert token counts [global_num_experts].
        group_offset: Starting expert index for this shard [1].
        activation: Activation function name.
        zero_w2_bias: If True, zero out w2_bias on all shards except shard 0
            (needed for TP where w2_bias is replicated).
    """
    # Pre-compute group metadata once and share between GMM1 and GMM2.
    # Both calls have the same (m, g, group_sizes, group_offset) and the
    # same tm (which depends only on m and g, not k/n).
    m = token_indices.shape[0]
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

    # GMM1 computes hidden_states @ (W_up | W_gate) with fused gather.
    # hidden_states stays in HBM; token_indices tells the kernel which
    # rows to load for each tile.
    gmm1_res_gate_up = gmm_gather_wrapper(
        hidden_states, token_indices, w1, w1_scale, w1_bias,
        group_sizes, group_offset, group_metadata, num_active_tiles)
    gmm1_res_gate, gmm1_res_up = jnp.split(gmm1_res_gate_up, 2, -1)
    gmm1_res = apply_act_fn(activation, gmm1_res_gate, gmm1_res_up)

    if zero_w2_bias and w2_bias is not None:
        shard_id = jax.lax.axis_index(ShardingAxisName.MLP_TENSOR).sum()
        w2_bias = jnp.where(shard_id == 0, w2_bias, 0)

    gmm2_res = gmm_wrapper(gmm1_res, w2, w2_scale, w2_bias, group_sizes,
                           group_offset, group_metadata, num_active_tiles)
    return gmm2_res


def moe_gmm_local(
    hidden_states: jax.Array,
    token_indices: jax.Array,
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
    """Main MoE logic on a local shard — TP or EP.

    For TP: all-reduce (psum) across tensor-parallel shards.
    For EP: reduce-scatter (psum_scatter) across expert-parallel shards.
    """

    assert parallelism in ["tp", "ep"]

    gmm2_res = _gmm_compute_local(
        hidden_states, token_indices,
        w1, w1_scale, w1_bias, w2, w2_scale, w2_bias,
        group_sizes, group_offset,
        activation=activation,
        zero_w2_bias=(parallelism == "tp"),
    )

    # First run local reduction on topk experts owned by the rank for all tokens
    token_topk_hidden = gmm2_res[topk_argsort_revert_indices].reshape(
        (-1, topk, gmm2_res.shape[-1]))
    token_topk_hidden = token_topk_hidden * jnp.expand_dims(topk_weights,
                                                            axis=-1)
    token_hidden = token_topk_hidden.sum(axis=-2)

    if parallelism == "tp":
        # TP: all-reduce across tensor-parallel shards.
        return jax.lax.psum(token_hidden, axis_name=ShardingAxisName.MLP_TENSOR)
    else:
        # EP: reduce-scatter across expert-parallel shards.  Each device
        # computed partial results for all tokens; psum_scatter sums them
        # and scatters dim-0 so each device keeps only its 1/EP slice of
        # tokens.  This halves communication vs psum (reduce-scatter vs
        # all-reduce) and is the same pattern MaxText uses.
        return jax.lax.psum_scatter(
            token_hidden, axis_name=ShardingAxisName.EXPERT,
            scatter_dimension=0, tiled=True)



def tensor_parallel_gmm(
    hidden_states: jax.Array,
    token_indices: jax.Array,
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
        in_specs=(data_p_spec, data_p_spec,
                  w1_spec, w1_scale_spec, w1_bias_spec, w2_spec,
                  w2_scale_spec, w2_bias_spec, data_p_spec, data_p_spec,
                  data_p_spec, data_p_spec),
        out_specs=(data_p_spec),
        check_vma=False,
    )(hidden_states, token_indices,
      w1, w1_scale, w1_bias, w2, w2_scale, w2_bias, group_sizes,
      group_offset, topk_argsort_revert_indices, topk_weights)


def expert_parallel_gmm(
    hidden_states: jax.Array,
    token_indices: jax.Array,
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

    # moe_gmm_local with parallelism="ep" uses psum_scatter, which
    # reduce-scatters dim-0 across EP shards.  Each device returns
    # num_tokens/ep_size rows, so out_specs shards on the EXPERT axis
    # (== the EP mesh axis) to reassemble the full token dimension.
    return jax.shard_map(
        functools.partial(moe_gmm_local,
                          activation=activation,
                          topk=topk,
                          parallelism="ep"),
        mesh=mesh,
        in_specs=(data_p_spec, data_p_spec,
                  ep_p_spec, w1_scale_spec, w1_bias_spec,
                  ep_p_spec, w2_scale_spec, w2_bias_spec, data_p_spec,
                  ep_p_spec, data_p_spec, data_p_spec),
        out_specs=(ep_p_spec),
        check_vma=False,
    )(hidden_states, token_indices,
      w1, w1_scale, w1_bias, w2, w2_scale, w2_bias, group_sizes,
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

        # Don't materialize x = hidden_states_local[token_indices_sorted].
        # Instead return token_indices_sorted so gmm_gather can fuse the
        # permute into the first matmul.
        return token_indices_sorted, group_sizes_local, topk_argsort_revert_indices

    token_indices_sorted, group_sizes, topk_argsort_revert_indices = jax.shard_map(
        _process_tokens_locally,
        mesh=mesh,
        in_specs=(P(ShardingAxisName.MLP_DATA,
                    None), P(ShardingAxisName.MLP_DATA, None)),
        out_specs=(P(ShardingAxisName.MLP_DATA),
                   P(ShardingAxisName.MLP_DATA),
                   P(ShardingAxisName.MLP_DATA)))(hidden_states, topk_indices)

    # Pad hidden_states (not the permuted x — gmm_gather reads from this).
    hidden_states_padded = jnp.pad(
        hidden_states, ((0, 0), (0, padded_hidden_size - hidden_size)))

    if use_ep:
        x = expert_parallel_gmm(hidden_states_padded,
                                token_indices_sorted,
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
        x = tensor_parallel_gmm(hidden_states_padded,
                                token_indices_sorted,
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
