#!/usr/bin/env python3
"""A/B benchmark: gmm (with materialized permute) vs gmm_gather (fused).

Measures isolated GMM time on realistic DeepSeek-R1 dimensions.
Tests both GMM1 (k=7168, n=4096) and GMM2 (k=2048, n=7168).
"""

import os, time
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=rematerialization"

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.megablox.gmm import gmm, make_group_metadata
from tpu_inference.kernels.megablox.gmm_gather import gmm_gather
from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import get_device_vmem_limit

HIDDEN, INTER, EXPERTS = 7168, 2048, 256
EP = 8
LOCAL_EXPERTS = EXPERTS // EP   # 32
TOPK = 8
ACT_DTYPE = jnp.float8_e4m3fn
RHS_DTYPE = jnp.float4_e2m1fn
TOKEN_COUNTS = [1024, 4096, 16384]
WARMUP = 10
ITERS = 30


def bench_one(label, num_tokens, k, n, vmem_limit):
    """Compare gmm vs gmm_gather for one (num_tokens, k, n) config."""
    m = num_tokens * TOPK
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    # Un-permuted hidden states (what gmm_gather reads from HBM).
    hidden = jax.random.normal(k1, (num_tokens, k), dtype=jnp.bfloat16).astype(ACT_DTYPE)

    # Expert assignment → sorted token indices.
    # In real EP: group_sizes has length E=256, only 32 local experts are nonzero.
    expert_ids = jax.random.randint(k2, (num_tokens, TOPK), 0, LOCAL_EXPERTS)
    expert_ids_flat = expert_ids.flatten()
    argsort = jnp.argsort(expert_ids_flat)
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32).repeat(TOPK)
    token_indices_sorted = token_ids[argsort]
    # group_sizes length = EXPERTS (256), only first LOCAL_EXPERTS (32) are non-zero.
    local_group_sizes = jnp.bincount(expert_ids_flat, length=LOCAL_EXPERTS).astype(jnp.int32)
    group_sizes = jnp.zeros(EXPERTS, dtype=jnp.int32).at[:LOCAL_EXPERTS].set(local_group_sizes)

    # RHS weights.
    rhs = jax.random.normal(k3, (LOCAL_EXPERTS, k, n), dtype=jnp.bfloat16).astype(RHS_DTYPE)
    # Per-channel quantization: one scale for the entire k dim (shape[1]=1).
    rhs_scale = jnp.ones((LOCAL_EXPERTS, 1, 1, n), dtype=jnp.float32)
    group_offset = jnp.array(0, dtype=jnp.int32)

    # Pre-compute group metadata (shared by both).
    # Use tuned tiling to pick tm/tk/tn.
    from tpu_inference.kernels.megablox.tuned_block_sizes import get_tuned_block_sizes
    tiling = get_tuned_block_sizes(
        m=m, k=k, n=n,
        num_total_groups=EXPERTS, num_current_groups=LOCAL_EXPERTS,
        lhs_dtype=str(jnp.dtype(ACT_DTYPE)), rhs_dtype=str(jnp.dtype(RHS_DTYPE)),
        quant_block_size=k,
    )
    assert tiling is not None, f"No tiling for m={m}, k={k}, n={n}"
    tm, tk, tn = tiling

    group_metadata, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes, m=m, tm=tm,
        start_group=0, num_nonzero_groups=LOCAL_EXPERTS,
        visit_empty_groups=False,
    )

    # ── Baseline: materialize permute + gmm ──
    def run_baseline():
        # In real code, the permute is done outside gmm(). We include it here
        # to measure the full cost.
        x_p = hidden[token_indices_sorted]
        return gmm(
            lhs=x_p, rhs=rhs, group_sizes=group_sizes,
            preferred_element_type=ACT_DTYPE,
            tiling=tiling, group_offset=group_offset,
            rhs_scale=rhs_scale,
            vmem_limit_bytes=vmem_limit,
            group_metadata=group_metadata,
            num_active_tiles=num_active_tiles,
        )

    # ── New: gmm_gather (fused) ──
    def run_gather():
        return gmm_gather(
            lhs=hidden, token_indices=token_indices_sorted,
            rhs=rhs, group_sizes=group_sizes,
            preferred_element_type=ACT_DTYPE,
            tiling=tiling, group_offset=group_offset,
            rhs_scale=rhs_scale,
            vmem_limit_bytes=vmem_limit,
            group_metadata=group_metadata,
            num_active_tiles=num_active_tiles,
        )

    # Warmup both.
    for _ in range(WARMUP):
        run_baseline().block_until_ready()
    for _ in range(WARMUP):
        run_gather().block_until_ready()

    # Benchmark baseline.
    times_base = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        run_baseline().block_until_ready()
        t1 = time.perf_counter()
        times_base.append((t1 - t0) * 1000)

    # Benchmark gather.
    times_gather = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        run_gather().block_until_ready()
        t1 = time.perf_counter()
        times_gather.append((t1 - t0) * 1000)

    med_b = np.median(times_base)
    med_g = np.median(times_gather)
    speedup = med_b / med_g if med_g > 0 else float('inf')

    print(f"  {label:20s}  N={num_tokens:>5}  M={m:>6}  tiling=({tm},{tk},{tn})")
    print(f"    baseline (permute+gmm):  {med_b:7.2f}ms  (min={np.min(times_base):.2f}, max={np.max(times_base):.2f})")
    print(f"    gmm_gather (fused):      {med_g:7.2f}ms  (min={np.min(times_gather):.2f}, max={np.max(times_gather):.2f})")
    print(f"    speedup: {speedup:.3f}x")
    return med_b, med_g


def main():
    print(f"JAX: {jax.__version__}, Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    vmem_limit = get_device_vmem_limit()
    print(f"VMEM limit: {vmem_limit / 1e6:.0f} MB")
    print(f"Config: H={HIDDEN}, I={INTER}, E={EXPERTS}, EP={EP}, local_E={LOCAL_EXPERTS}, topk={TOPK}")
    print(f"Dtypes: act={ACT_DTYPE}, rhs={RHS_DTYPE}")
    print(f"Warmup={WARMUP}, Iters={ITERS}\n")

    for ntok in TOKEN_COUNTS:
        print(f"{'='*70}")
        # GMM1: x @ w1 → (m, k=H) @ (E, H, 2*I) = (m, 4096)
        bench_one("GMM1 (k=H, n=2I)", ntok, HIDDEN, INTER * 2, vmem_limit)
        print()
        # GMM2: act @ w2 → (m, k=I) @ (E, I, H) = (m, 7168)
        bench_one("GMM2 (k=I, n=H)", ntok, INTER, HIDDEN, vmem_limit)
        print()


if __name__ == "__main__":
    main()
