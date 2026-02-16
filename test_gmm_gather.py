#!/usr/bin/env python3
"""Quick correctness test: gmm_gather vs gmm on small + realistic problems."""

import os
os.environ["PYTHONUNBUFFERED"] = "1"

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.megablox.gmm import gmm, make_group_metadata
from tpu_inference.kernels.megablox.gmm_gather import gmm_gather


def run_test(label, num_tokens, topk, k, n, num_experts, tiling, lhs_dtype, rhs_dtype):
    """Compare gmm_gather output to gmm output."""
    from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import get_device_vmem_limit
    vmem_limit = get_device_vmem_limit()
    print(f"\n{'='*60}")
    print(f"Test: {label}")
    print(f"  num_tokens={num_tokens}, topk={topk}, k={k}, n={n}, "
          f"E={num_experts}, tiling={tiling}")
    print(f"  lhs_dtype={lhs_dtype}, rhs_dtype={rhs_dtype}")

    m = num_tokens * topk
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    hidden_states = jax.random.normal(k1, (num_tokens, k), dtype=jnp.bfloat16)
    hidden_states = hidden_states.astype(lhs_dtype)

    expert_ids = jax.random.randint(k2, (num_tokens, topk), 0, num_experts)
    expert_ids_flat = expert_ids.flatten()
    argsort_indices = jnp.argsort(expert_ids_flat)

    token_ids = jnp.arange(num_tokens, dtype=jnp.int32).repeat(topk)
    token_indices_sorted = token_ids[argsort_indices]

    group_sizes = jnp.bincount(expert_ids_flat, length=num_experts).astype(jnp.int32)

    x_permuted = hidden_states[token_indices_sorted]

    rhs_arr = jax.random.normal(k3, (num_experts, k, n), dtype=jnp.bfloat16)
    rhs_arr = rhs_arr.astype(rhs_dtype)

    # Create rhs_scale for quantized dtypes.
    # Per-channel quantization: one scale for the entire k dimension.
    rhs_scale = None
    if rhs_dtype in (jnp.float8_e4m3fn, jnp.float4_e2m1fn):
        rhs_scale = jnp.ones((num_experts, 1, 1, n), dtype=jnp.float32)

    group_offset = jnp.array([0], dtype=jnp.int32)

    print("  Running regular gmm...", end="", flush=True)
    out_gmm = gmm(
        lhs=x_permuted, rhs=rhs_arr, group_sizes=group_sizes,
        preferred_element_type=lhs_dtype, tiling=tiling,
        group_offset=group_offset[0], rhs_scale=rhs_scale,
        vmem_limit_bytes=vmem_limit,
    )
    out_gmm.block_until_ready()
    print(f" shape={out_gmm.shape}")

    print("  Running gmm_gather...", end="", flush=True)
    out_gather = gmm_gather(
        lhs=hidden_states, token_indices=token_indices_sorted,
        rhs=rhs_arr, group_sizes=group_sizes,
        preferred_element_type=lhs_dtype, tiling=tiling,
        group_offset=group_offset[0], rhs_scale=rhs_scale,
        vmem_limit_bytes=vmem_limit,
    )
    out_gather.block_until_ready()
    print(f" shape={out_gather.shape}")

    max_diff = jnp.max(jnp.abs(out_gmm.astype(jnp.float32) -
                                out_gather.astype(jnp.float32)))
    rel_diff = max_diff / (jnp.max(jnp.abs(out_gmm.astype(jnp.float32))) + 1e-8)
    print(f"  Max abs diff: {max_diff:.6f}, rel diff: {rel_diff:.6f}")

    if rel_diff < 1e-3:
        print(f"  ✅ PASS")
        return True
    else:
        print(f"  ❌ FAIL")
        return False


if __name__ == "__main__":
    results = []

    # Test 1: small bf16 problem
    results.append(run_test(
        "small bf16", num_tokens=256, topk=4, k=1024, n=512,
        num_experts=8, tiling=(128, 1024, 128),
        lhs_dtype=jnp.bfloat16, rhs_dtype=jnp.bfloat16,
    ))

    # Test 2: DeepSeek-R1 GMM1 dimensions (bf16 × bf16)
    results.append(run_test(
        "DS-R1 GMM1 bf16", num_tokens=128, topk=8, k=7168, n=4096,
        num_experts=32, tiling=(128, 7168, 2048),
        lhs_dtype=jnp.bfloat16, rhs_dtype=jnp.bfloat16,
    ))

    # Test 3: DeepSeek-R1 GMM1 dimensions (fp8 × fp8)
    results.append(run_test(
        "DS-R1 GMM1 fp8xfp8", num_tokens=128, topk=8, k=7168, n=4096,
        num_experts=32, tiling=(128, 7168, 2048),
        lhs_dtype=jnp.float8_e4m3fn, rhs_dtype=jnp.float8_e4m3fn,
    ))

    # Test 4: DeepSeek-R1 GMM2 dimensions (fp8 × fp8)
    results.append(run_test(
        "DS-R1 GMM2 fp8xfp8", num_tokens=128, topk=8, k=2048, n=7168,
        num_experts=32, tiling=(128, 2048, 7168),
        lhs_dtype=jnp.float8_e4m3fn, rhs_dtype=jnp.float8_e4m3fn,
    ))

    print(f"\n{'='*60}")
    print(f"Results: {sum(results)}/{len(results)} passed")
