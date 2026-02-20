"""Sweep GMM tile sizes for DeepSeek-R1 MoE shapes on TPU v7.

Sweeps all valid (tm, tk, tn) tile combinations for both GMM1 and GMM2,
benchmarks each, and prints the optimal tile per token count.

Results go into tuned_block_sizes.py.

DeepSeek-R1 MoE:
  GMM1: lhs=[M, 7168] @ rhs=[32, 7168, 4096]  (hidden → 2*intermediate)
  GMM2: lhs=[M, 2048] @ rhs=[32, 2048, 7168]  (intermediate → hidden)
  256 experts total, EP=8 → 32 local experts per device
  topk=8, so M = num_tokens * topk

Usage:
  python benchmark_gmm_tile_sweep.py --lhs-dtype bf16 --rhs-dtype bf16
  python benchmark_gmm_tile_sweep.py --lhs-dtype bf16 --rhs-dtype fp8
  python benchmark_gmm_tile_sweep.py --lhs-dtype fp8 --rhs-dtype fp8
  python benchmark_gmm_tile_sweep.py --lhs-dtype fp8 --rhs-dtype fp4
  python benchmark_gmm_tile_sweep.py --lhs-dtype fp8 --rhs-dtype fp4 --quant-block-size 256
  python benchmark_gmm_tile_sweep.py --lhs-dtype fp8 --rhs-dtype fp8 --quant-block-size 512
"""

import argparse
import os
import sys
import time
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tpu_inference.kernels.megablox.gmm import gmm, make_group_metadata

# Map CLI names to (jnp dtype, str for LUT key)
DTYPE_MAP = {
    "bf16": (jnp.bfloat16, "bfloat16"),
    "fp8": (jnp.float8_e4m3fn, "float8_e4m3fn"),
    "fp4": (jnp.float4_e2m1fn, "float4_e2m1fn"),
}

# ── DeepSeek-R1 config ──
HIDDEN = 7168
INTERMEDIATE = 2048
NUM_EXPERTS = 256
TOP_K = 8
EP_SIZE = 8
LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE  # 32

# GMM shapes (per device, after EP sharding):
# GMM1: lhs=[M, 7168] @ rhs=[32, 7168, 4096]  (gate+up fused = 2*2048)
# GMM2: lhs=[M, 2048] @ rhs=[32, 2048, 7168]
GMM_SHAPES = {
    "GMM1": {"k": HIDDEN, "n": INTERMEDIATE * 2, "label": "hidden→2*inter"},
    "GMM2": {"k": INTERMEDIATE, "n": HIDDEN, "label": "inter→hidden"},
}

# Token counts to sweep (total tokens, M = tokens * topk)
TOKEN_COUNTS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

WARMUP = 3
ITERS = 5


def round_up_128(x):
    return ((x + 127) // 128) * 128


def compute_tm(m, g):
    """Same as _compute_tm in fused_moe_gmm.py."""
    raw = 2 * m // g
    if raw <= 128:
        tm = 128
    elif raw < 512:
        tm = round_up_128(raw)
    else:
        # Find largest multiple of 128 <= 512 that divides raw evenly
        tm = 512
        for candidate in range(512, 127, -128):
            if raw % candidate == 0:
                tm = candidate
                break
    return min(tm, m)


def divisors_of(n, min_val=128):
    """All divisors of n >= min_val, sorted descending."""
    divs = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            if i >= min_val:
                divs.add(i)
            if n // i >= min_val:
                divs.add(n // i)
    return sorted(divs, reverse=True)


def generate_tile_candidates(m, k, n, g, quant_block_size=None):
    """Generate all valid (tm, tk, tn) tile combinations."""
    if quant_block_size is None:
        quant_block_size = k
    tm = compute_tm(m, g)

    # tk: must divide k, be >= 128, be multiple of 128
    # With subchannel: tk % quant_block_size == 0 or quant_block_size % tk == 0
    tk_cands = [d for d in divisors_of(k, 128) if d % 128 == 0]
    if quant_block_size < k:
        tk_cands = [tk for tk in tk_cands
                    if tk % quant_block_size == 0 or quant_block_size % tk == 0]
    if not tk_cands:
        tk_cands = [k]

    # tn: multiples of 128 that divide n, plus n itself
    tn_cands = []
    for tn in range(128, n + 1, 128):
        if n % tn == 0:
            tn_cands.append(tn)
    if n not in tn_cands:
        tn_cands.append(n)
    tn_cands.sort(reverse=True)

    candidates = []
    for tk in tk_cands:
        for tn in tn_cands:
            tiles_k = -(-k // tk)
            tiles_n = -(-n // tn)
            total_tiles = tiles_k * tiles_n
            candidates.append((tm, tk, tn, total_tiles))

    return candidates


def try_benchmark_gmm(lhs, rhs, group_sizes, group_offset,
                      tiling, vmem_limit, rhs_scale=None,
                      warmup=WARMUP, iters=ITERS):
    """Benchmark a single GMM tile config. Returns median ms or None on error."""
    try:
        m = lhs.shape[0]
        g = rhs.shape[0]
        tm = tiling[0]

        # Pre-compute group metadata
        group_metadata, num_active_tiles = make_group_metadata(
            group_sizes=group_sizes,
            m=m,
            tm=tm,
            start_group=group_offset,
            num_nonzero_groups=g,
            visit_empty_groups=False,
        )

        def run():
            return gmm(
                lhs=lhs,
                rhs=rhs,
                group_sizes=group_sizes,
                preferred_element_type=lhs.dtype,
                rhs_scale=rhs_scale,
                tiling=tiling,
                group_offset=group_offset,
                vmem_limit_bytes=vmem_limit,
                group_metadata=group_metadata,
                num_active_tiles=num_active_tiles,
            )

        # Warmup — also catches OOM
        for _ in range(warmup):
            out = run()
            out.block_until_ready()

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = run()
            out.block_until_ready()
            times.append(time.perf_counter() - t0)

        # Clear JIT cache to avoid accumulating compiled kernels
        gmm.clear_cache()

        return np.median(times) * 1e3
    except Exception as e:
        gmm.clear_cache()
        err = str(e)[:200]
        if "RESOURCE_EXHAUSTED" not in err and "out of memory" not in err.lower():
            print(f"      ERROR: {type(e).__name__}: {err}")
        return None


def sweep_one_shape(name, k, n, m, g, group_sizes, group_offset,
                    lhs_jnp_dtype=jnp.bfloat16, rhs_jnp_dtype=jnp.bfloat16,
                    quant_block_size=None):
    """Sweep all tile candidates for one GMM shape at one token count."""
    qbs = quant_block_size if quant_block_size is not None else k
    candidates = generate_tile_candidates(m, k, n, g, quant_block_size=qbs)
    lhs_str = str(lhs_jnp_dtype.__name__ if hasattr(lhs_jnp_dtype, '__name__') else lhs_jnp_dtype)
    rhs_str = str(rhs_jnp_dtype.__name__ if hasattr(rhs_jnp_dtype, '__name__') else rhs_jnp_dtype)
    qbs_str = f", qbs={qbs}" if qbs != k else ""
    print(f"    {name} (M={m}, K={k}, N={n}, lhs={lhs_str}, rhs={rhs_str}{qbs_str}): "
          f"{len(candidates)} tile candidates")

    # Create test data
    key = jax.random.key(42)
    k1, k2, k3 = jax.random.split(key, 3)
    lhs = (jax.random.normal(k1, (m, k), dtype=jnp.bfloat16) / 10.0).astype(lhs_jnp_dtype)
    rhs = (jax.random.normal(k2, (g, k, n), dtype=jnp.bfloat16) / 10.0).astype(rhs_jnp_dtype)

    # Generate rhs_scale for subchannel quantization
    rhs_scale = None
    if qbs < k:
        num_blocks = k // qbs
        # Scale shape: (num_groups, num_blocks, 1, n) — bf16 scales
        rhs_scale = jax.random.uniform(k3, (g, num_blocks, 1, n),
                                       dtype=jnp.bfloat16,
                                       minval=0.5, maxval=1.5)

    # VMEM limits to try: 64MB is the actual TPU v7 VMEM per core
    vmem_limits = [
        int(64 * 1024 * 1024 * 0.85),  # 54.5 MB
        int(64 * 1024 * 1024 * 0.90),  # 57.6 MB
        int(64 * 1024 * 1024 * 0.95),  # 60.8 MB
        64 * 1024 * 1024,              # 64.0 MB (full device VMEM)
    ]

    best_ms = float('inf')
    best_cfg = None
    tried = 0
    successes = 0

    # Sort candidates by total tiles (ascending = fewer tiles first = likely faster)
    candidates.sort(key=lambda c: (c[3], -c[1]))  # tiles ASC, tk DESC

    for tm, tk, tn, total_tiles in candidates:
        tiling = (tm, tk, tn)
        tiles_k = -(-k // tk)
        tiles_n = -(-n // tn)

        # Try each VMEM limit until one works
        ms = None
        used_vmem = None
        for vlim in vmem_limits:
            ms = try_benchmark_gmm(lhs, rhs, group_sizes, group_offset,
                                   tiling, vlim, rhs_scale=rhs_scale)
            if ms is not None:
                used_vmem = vlim
                break

        tried += 1
        if ms is not None:
            successes += 1
            marker = ""
            if ms < best_ms:
                best_ms = ms
                best_cfg = (tm, tk, tn, ms, used_vmem)
                marker = " ★"
            vlim_str = f"{used_vmem/1e6:.1f}MB" if used_vmem else "none"
            # Only print the best few and winners
            if successes <= 5 or marker:
                flops = 2 * m * k * n
                tflops = flops / (ms * 1e-3) / 1e12
                print(f"      [{successes:>2}] ({tm:>4},{tk:>5},{tn:>5}) "
                      f"tiles={tiles_k}×{tiles_n}={total_tiles:>4} "
                      f"vmem={vlim_str} → {ms:.3f}ms {tflops:.1f}TF/s{marker}")
        else:
            if tried <= 3:
                print(f"      [{tried:>2}] ({tm:>4},{tk:>5},{tn:>5}) "
                      f"tiles={tiles_k}×{tiles_n}={total_tiles:>4} → OOM/FAIL")

    if best_cfg:
        tm, tk, tn, ms, vlim = best_cfg
        flops = 2 * m * k * n
        tflops = flops / (ms * 1e-3) / 1e12
        vlim_str = f"{vlim/1e6:.1f}MB" if vlim else "none"
        print(f"    ★ {name} BEST: ({tm}, {tk}, {tn}) → {ms:.3f}ms "
              f"({tflops:.1f} TF/s) vmem_limit={vlim_str}")
    else:
        print(f"    ✗ {name}: ALL FAILED ({tried} tried)")

    return best_cfg


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lhs-dtype", default="bf16",
                        choices=["bf16", "fp8"],
                        help="LHS activation dtype (default: bf16)")
    parser.add_argument("--rhs-dtype", default="bf16",
                        choices=["bf16", "fp8", "fp4"],
                        help="RHS weight dtype (default: bf16)")
    parser.add_argument("--quant-block-size", type=int, default=None,
                        choices=[128, 256, 512],
                        help="Subchannel quantization block size on K dim "
                             "(default: None = full K, no subchannel)")
    args = parser.parse_args()

    lhs_jnp_dtype, lhs_dtype_str = DTYPE_MAP[args.lhs_dtype]
    rhs_jnp_dtype, rhs_dtype_str = DTYPE_MAP[args.rhs_dtype]

    qbs = args.quant_block_size  # None means full-K (no subchannel)

    print(f"TPU: {jax.devices()[0]}")
    print(f"Devices: {len(jax.devices())}")
    print(f"DeepSeek-R1: H={HIDDEN}, I={INTERMEDIATE}, E={NUM_EXPERTS}, "
          f"topk={TOP_K}, EP={EP_SIZE}, local_experts={LOCAL_EXPERTS}")
    print(f"LHS dtype: {args.lhs_dtype} ({lhs_dtype_str})")
    print(f"RHS dtype: {args.rhs_dtype} ({rhs_dtype_str})")
    if qbs is not None:
        print(f"Subchannel quant_block_size: {qbs} (rhs_scale applied per block)")
    else:
        print(f"No subchannel — quant_block_size=K")
    print()

    # Results storage
    results = {}  # (name, num_tokens) -> (tm, tk, tn, ms, vmem_limit)

    for num_tokens in TOKEN_COUNTS:
        m = num_tokens * TOP_K  # expanded token count
        g = LOCAL_EXPERTS       # 32 local experts

        # Create realistic group_sizes: distribute m tokens across 256 experts
        # (uniform-ish with some noise)
        avg_per_expert = m / NUM_EXPERTS
        key = jax.random.key(num_tokens)
        noise = jax.random.uniform(key, (NUM_EXPERTS,), minval=0.5, maxval=1.5)
        raw_sizes = (noise * avg_per_expert).astype(jnp.int32)
        # Fix sum to match m exactly
        diff = m - raw_sizes.sum()
        raw_sizes = raw_sizes.at[0].add(diff)
        group_sizes = raw_sizes

        # group_offset for first shard
        group_offset = jnp.int32(0)

        print(f"═══ tokens={num_tokens} (M={m}, tok/expert≈{m//NUM_EXPERTS}) ═══")

        for name, shape in GMM_SHAPES.items():
            best = sweep_one_shape(
                name, shape["k"], shape["n"], m, g,
                group_sizes, group_offset,
                lhs_jnp_dtype=lhs_jnp_dtype,
                rhs_jnp_dtype=rhs_jnp_dtype,
                quant_block_size=qbs,
            )
            if best:
                results[(name, num_tokens)] = best
        print()

    # ── Summary table ──
    print()
    print("=" * 120)
    qbs_label = f", qbs={qbs}" if qbs else ""
    print(f"SUMMARY: Optimal GMM tiles for DeepSeek-R1 on TPU v7 "
          f"(lhs={args.lhs_dtype}, rhs={args.rhs_dtype}{qbs_label})")
    print("=" * 120)
    print()
    print(f"{'tokens':>8} {'M':>8} {'kernel':>5} {'tm':>5} {'tk':>6} {'tn':>6} "
          f"{'tiles':>8} {'ms':>9} {'TF/s':>8} {'vmem_lim':>10}")
    print("-" * 100)

    for num_tokens in TOKEN_COUNTS:
        m = num_tokens * TOP_K
        for name, shape in GMM_SHAPES.items():
            key = (name, num_tokens)
            if key not in results:
                print(f"{num_tokens:>8} {m:>8} {name:>5}  --- FAILED ---")
                continue
            tm, tk, tn, ms, vlim = results[key]
            k, n = shape["k"], shape["n"]
            tiles_k = -(-k // tk)
            tiles_n = -(-n // tn)
            total = tiles_k * tiles_n
            flops = 2 * m * k * n
            tflops = flops / (ms * 1e-3) / 1e12
            vlim_str = f"{vlim/1e6:.0f}MB" if vlim else "none"
            print(f"{num_tokens:>8} {m:>8} {name:>5} {tm:>5} {tk:>6} {tn:>6} "
                  f"{total:>8} {ms:>9.3f} {tflops:>8.1f} {vlim_str:>10}")

    # ── Print entries for tuned_block_sizes.py ──
    print()
    print("=" * 120)
    qbs_label2 = f", qbs={qbs}" if qbs else ""
    print(f"TUNED_BLOCK_SIZES entries for lhs={args.lhs_dtype}, rhs={args.rhs_dtype}"
          f"{qbs_label2} (copy into tuned_block_sizes.py):")
    print("=" * 120)
    print()

    for num_tokens in TOKEN_COUNTS:
        m = num_tokens * TOP_K
        for name, shape in GMM_SHAPES.items():
            key = (name, num_tokens)
            if key not in results:
                continue
            tm, tk, tn, ms, vlim = results[key]
            k, n = shape["k"], shape["n"]
            # Key format: (m, k, n, num_total_groups, num_current_groups,
            #              lhs_dtype, rhs_dtype, quant_block_size)
            actual_qbs = qbs if qbs is not None else k
            print(f"    ({m}, {k}, {n}, {NUM_EXPERTS}, {LOCAL_EXPERTS}, "
                  f"'{lhs_dtype_str}', '{rhs_dtype_str}', {actual_qbs}): ({tm}, {tk}, {tn}),  "
                  f"# {name} tokens={num_tokens} {ms:.3f}ms")

    print()
    print("DONE")


if __name__ == "__main__":
    run()
