"""Benchmark GMM kernel across all dtype and subchannel configurations.

Uses the already-tuned optimal tiles from tuned_block_sizes.py — no tile
searching. Just runs each (dtype, qbs, M) combo with its best-known tile
and reports a comparison table.

DeepSeek-R1 MoE shapes:
  GMM1: lhs=[M, 7168] @ rhs=[32, 7168, 4096]  (hidden → gate+up)
  GMM2: lhs=[M, 2048] @ rhs=[32, 2048, 7168]  (intermediate → hidden)

Usage:
  python benchmark_gmm_configs.py
  python benchmark_gmm_configs.py --configs fp8xfp8 fp8xfp4
  python benchmark_gmm_configs.py --configs fp8xfp8:256 fp8xfp4:256
  python benchmark_gmm_configs.py --tokens 1024 8192
"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.megablox.gmm import gmm, make_group_metadata
from tpu_inference.kernels.megablox.tuned_block_sizes import (
    TUNED_BLOCK_SIZES, get_tuned_block_sizes,
)

# ── DeepSeek-R1 config ──
HIDDEN = 7168
INTERMEDIATE = 2048
NUM_EXPERTS = 256
TOP_K = 8
EP_SIZE = 8
LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE  # 32

GMM_SHAPES = [
    ("GMM1", HIDDEN, INTERMEDIATE * 2),   # K=7168, N=4096
    ("GMM2", INTERMEDIATE, HIDDEN),        # K=2048, N=7168
]

TOKEN_COUNTS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

DTYPE_MAP = {
    "bf16": jnp.bfloat16,
    "fp8":  jnp.float8_e4m3fn,
    "fp4":  jnp.float4_e2m1fn,
}

DTYPE_STR = {
    jnp.bfloat16:       "bfloat16",
    jnp.float8_e4m3fn:  "float8_e4m3fn",
    jnp.float4_e2m1fn:  "float4_e2m1fn",
}

# All configurations to sweep: (label, lhs_dtype, rhs_dtype, qbs_or_None)
# qbs=None means use full K (no subchannel)
ALL_CONFIGS = [
    ("bf16×bf16",      "bf16", "bf16", None),
    ("bf16×fp8",       "bf16", "fp8",  None),
    ("bf16×fp4",       "bf16", "fp4",  None),
    ("fp8×fp8",        "fp8",  "fp8",  None),
    ("fp8×fp8 q512",   "fp8",  "fp8",  512),
    ("fp8×fp8 q256",   "fp8",  "fp8",  256),
    ("fp8×fp4",        "fp8",  "fp4",  None),
    ("fp8×fp4 q512",   "fp8",  "fp4",  512),
    ("fp8×fp4 q256",   "fp8",  "fp4",  256),
]

# Short name → config lookup for --configs filter
CONFIG_ALIASES = {}
for cfg in ALL_CONFIGS:
    label = cfg[0]
    # "fp8×fp8 q256" → aliases: "fp8xfp8:256", "fp8xfp8q256"
    short = label.replace("×", "x").replace(" ", "")
    CONFIG_ALIASES[short] = cfg
    # Also allow colon form: fp8xfp8:256
    short2 = label.replace("×", "x").replace(" q", ":")
    CONFIG_ALIASES[short2] = cfg
    # And the full label
    CONFIG_ALIASES[label] = cfg

WARMUP = 3
ITERS = 5
VMEM_LIMITS = [
    int(64 * 1024 * 1024 * 0.85),  # ~54.5 MB
    int(64 * 1024 * 1024 * 0.89),  # ~57.6 MB
    int(64 * 1024 * 1024 * 0.95),  # ~60.8 MB
    64 * 1024 * 1024,              # 64.0 MB
]


def benchmark_one(lhs, rhs, group_sizes, group_offset, tiling,
                  rhs_scale=None, warmup=WARMUP, iters=ITERS):
    """Run one GMM config and return median ms. Tries multiple VMEM limits."""
    for vmem_limit in VMEM_LIMITS:
        result = _benchmark_one_vmem(lhs, rhs, group_sizes, group_offset,
                                     tiling, vmem_limit, rhs_scale=rhs_scale,
                                     warmup=warmup, iters=iters)
        if result is not None:
            return result
    return None


def _benchmark_one_vmem(lhs, rhs, group_sizes, group_offset, tiling,
                        vmem_limit, rhs_scale=None, warmup=WARMUP, iters=ITERS):
    """Run one GMM config at a specific VMEM limit. Returns median ms or None."""
    try:
        m = lhs.shape[0]
        g = rhs.shape[0]
        tm = tiling[0]

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

        for _ in range(warmup):
            out = run()
            out.block_until_ready()

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = run()
            out.block_until_ready()
            times.append(time.perf_counter() - t0)

        gmm.clear_cache()
        return np.median(times) * 1e3
    except Exception:
        gmm.clear_cache()
        return None


def get_tiling(m, k, n, lhs_dtype, rhs_dtype, qbs):
    """Look up tuned tile from LUT."""
    key = (m, k, n, NUM_EXPERTS, LOCAL_EXPERTS,
           DTYPE_STR[lhs_dtype], DTYPE_STR[rhs_dtype], qbs)
    if key in TUNED_BLOCK_SIZES:
        return TUNED_BLOCK_SIZES[key]
    return None


def run():
    parser = argparse.ArgumentParser(
        description="Benchmark GMM across dtype/subchannel configs")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Configs to run, e.g. fp8xfp8 fp8xfp4:256 "
                             "(default: all)")
    parser.add_argument("--tokens", nargs="+", type=int, default=None,
                        help="Token counts to test (default: all)")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    args = parser.parse_args()

    # Filter configs
    if args.configs:
        configs = []
        for name in args.configs:
            name_clean = name.replace("×", "x")
            if name_clean in CONFIG_ALIASES:
                configs.append(CONFIG_ALIASES[name_clean])
            else:
                print(f"Unknown config: {name}")
                print(f"Available: {list(CONFIG_ALIASES.keys())}")
                sys.exit(1)
    else:
        configs = ALL_CONFIGS

    token_counts = args.tokens if args.tokens else TOKEN_COUNTS

    print(f"TPU: {jax.devices()[0]}")
    print(f"Devices: {jax.device_count()}")
    print(f"DeepSeek-R1: H={HIDDEN}, I={INTERMEDIATE}, E={NUM_EXPERTS}, "
          f"topk={TOP_K}, EP={EP_SIZE}, local_experts={LOCAL_EXPERTS}")
    print(f"VMEM limits: {[f'{v/1e6:.1f}MB' for v in VMEM_LIMITS]}")
    print(f"Warmup: {args.warmup}, Iters: {args.iters}")
    print(f"Configs: {[c[0] for c in configs]}")
    print(f"Token counts: {token_counts}")
    print()

    warmup = args.warmup
    iters = args.iters

    # ── Run all benchmarks ──
    # results[(gmm_name, tokens)][config_label] = ms
    results = {}
    g = LOCAL_EXPERTS
    group_offset = jnp.int32(0)

    for gmm_name, k, n in GMM_SHAPES:
        print(f"{'='*70}")
        print(f"  {gmm_name}: K={k}, N={n}")
        print(f"{'='*70}")

        for tokens in token_counts:
            m = tokens * TOP_K

            # Even group distribution
            base = m // g
            remainder = m % g
            sizes = [base + (1 if i < remainder else 0) for i in range(g)]
            group_sizes = jnp.array(sizes, dtype=jnp.int32)

            key = jax.random.key(42)
            k1, k2 = jax.random.split(key, 2)

            row_key = (gmm_name, tokens)
            if row_key not in results:
                results[row_key] = {}

            for cfg_label, lhs_name, rhs_name, qbs_raw in configs:
                lhs_dtype = DTYPE_MAP[lhs_name]
                rhs_dtype = DTYPE_MAP[rhs_name]
                qbs = qbs_raw if qbs_raw is not None else k

                tiling = get_tiling(m, k, n, lhs_dtype, rhs_dtype, qbs)
                if tiling is None:
                    print(f"  tokens={tokens:>5} {cfg_label:>14}: "
                          f"NO TUNED TILE (skipped)")
                    results[row_key][cfg_label] = None
                    continue

                # Create data
                lhs = (jax.random.normal(k1, (m, k), dtype=jnp.bfloat16)
                       / 10.0).astype(lhs_dtype)
                rhs = (jax.random.normal(k2, (g, k, n), dtype=jnp.bfloat16)
                       / 10.0).astype(rhs_dtype)

                rhs_scale = None
                if qbs < k:
                    num_blocks = k // qbs
                    k3 = jax.random.key(99)
                    rhs_scale = jax.random.uniform(
                        k3, (g, num_blocks, 1, n),
                        dtype=jnp.bfloat16, minval=0.5, maxval=1.5)

                try:
                    ms = benchmark_one(lhs, rhs, group_sizes, group_offset,
                                       tiling, rhs_scale=rhs_scale,
                                       warmup=warmup, iters=iters)
                    if ms is None:
                        print(f"  tokens={tokens:>5} {cfg_label:>14}: "
                              f"ALL VMEM LIMITS FAILED")
                        results[row_key][cfg_label] = None
                        continue
                    results[row_key][cfg_label] = ms
                    flops = 2 * m * k * n
                    tflops = flops / (ms * 1e-3) / 1e12
                    tm, tk, tn = tiling
                    print(f"  tokens={tokens:>5} {cfg_label:>14}: "
                          f"{ms:>7.3f} ms  {tflops:>7.1f} TF/s  "
                          f"tile=({tm},{tk},{tn})")
                except Exception as e:
                    err = str(e)[:120]
                    print(f"  tokens={tokens:>5} {cfg_label:>14}: "
                          f"FAILED ({type(e).__name__}: {err})")
                    results[row_key][cfg_label] = None

            del lhs, rhs, rhs_scale
            print()

    # ── Print summary tables ──
    cfg_labels = [c[0] for c in configs]

    for gmm_name, k, n in GMM_SHAPES:
        print()
        print(f"{'='*100}")
        print(f"  {gmm_name}: K={k}, N={n} — Optimal tile latency (ms)")
        print(f"{'='*100}")

        # Header
        hdr = f"{'tokens':>6} {'M':>7}"
        for label in cfg_labels:
            hdr += f"  {label:>14}"
        print(hdr)
        print("-" * len(hdr))

        for tokens in token_counts:
            m = tokens * TOP_K
            row_key = (gmm_name, tokens)
            line = f"{tokens:>6} {m:>7}"
            for label in cfg_labels:
                val = results.get(row_key, {}).get(label)
                if val is not None:
                    line += f"  {val:>14.3f}"
                else:
                    line += f"  {'—':>14}"
            print(line)

    # ── Combined GMM1+GMM2 table ──
    print()
    print(f"{'='*100}")
    print(f"  GMM1 + GMM2 Combined (ms)")
    print(f"{'='*100}")
    hdr = f"{'tokens':>6} {'M':>7}"
    for label in cfg_labels:
        hdr += f"  {label:>14}"
    print(hdr)
    print("-" * len(hdr))

    for tokens in token_counts:
        m = tokens * TOP_K
        line = f"{tokens:>6} {m:>7}"
        for label in cfg_labels:
            g1 = results.get(("GMM1", tokens), {}).get(label)
            g2 = results.get(("GMM2", tokens), {}).get(label)
            if g1 is not None and g2 is not None:
                line += f"  {g1 + g2:>14.3f}"
            else:
                line += f"  {'—':>14}"
        print(line)

    # ── Overhead vs no-subchannel table ──
    # Find baseline configs (qbs=k) for each dtype pair
    baselines = {}
    for cfg_label, lhs_name, rhs_name, qbs_raw in configs:
        if qbs_raw is None:
            baselines[(lhs_name, rhs_name)] = cfg_label

    sc_configs = [(label, lhs, rhs, qbs)
                  for label, lhs, rhs, qbs in configs if qbs is not None]

    if sc_configs and baselines:
        print()
        print(f"{'='*100}")
        print(f"  Subchannel overhead vs no-subchannel (GMM1+GMM2 combined)")
        print(f"{'='*100}")
        sc_labels = [c[0] for c in sc_configs]
        hdr = f"{'tokens':>6} {'M':>7}"
        for label in sc_labels:
            hdr += f"  {label:>14}"
        print(hdr)
        print("-" * len(hdr))

        for tokens in token_counts:
            line = f"{tokens:>6} {tokens * TOP_K:>7}"
            for cfg_label, lhs_name, rhs_name, qbs_raw in sc_configs:
                base_label = baselines.get((lhs_name, rhs_name))
                if not base_label:
                    line += f"  {'—':>14}"
                    continue
                g1 = results.get(("GMM1", tokens), {}).get(cfg_label)
                g2 = results.get(("GMM2", tokens), {}).get(cfg_label)
                b1 = results.get(("GMM1", tokens), {}).get(base_label)
                b2 = results.get(("GMM2", tokens), {}).get(base_label)
                if all(v is not None for v in [g1, g2, b1, b2]):
                    sc_total = g1 + g2
                    base_total = b1 + b2
                    pct = (sc_total / base_total - 1) * 100
                    line += f"  {f'+{pct:.0f}%':>14}"
                else:
                    line += f"  {'—':>14}"
            print(line)

    print()
    print("DONE")


if __name__ == "__main__":
    run()
