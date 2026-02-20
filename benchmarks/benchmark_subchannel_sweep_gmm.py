#!/usr/bin/env python3
"""Sweep subchannel (quant block) size using pure gmm_v2 kernel.

Sweeps quant_block_size from 16 to 2048 in powers of two, plus channelwise
and bf16×bf16 baseline. Pure GMM kernel — no permute, gating, activation,
all-to-all, or EP overhead.

Uses auto-tiling. DeepSeek-R1 shapes: H=7168, I=2048, 32 local experts.
"""

import os
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=rematerialization"

import time
import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2

HIDDEN = 7168
INTER = 2048
EXPERTS_LOCAL = 32  # EP=8, 256/8
TOPK = 8
WARMUP = 5
ITERS = 20

TOKEN_COUNTS = [128, 256, 512, 1024, 2048, 4096, 8192]
SUBCHANNEL_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]


def bench_gmm(m, k, n, g, lhs_dtype, rhs_dtype, rhs_scale):
    """Benchmark one gmm_v2 call. Returns median ms or None."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    lhs = (jax.random.normal(k1, (m, k), dtype=jnp.bfloat16) * 0.1).astype(lhs_dtype)
    rhs = (jax.random.normal(k2, (g, k, n), dtype=jnp.bfloat16) * 0.01).astype(rhs_dtype)
    group_sizes = jnp.full((g,), m // g, dtype=jnp.int32)
    group_offset = jnp.array([0], dtype=jnp.int32)

    try:
        def run():
            return gmm_v2(
                lhs=lhs, rhs=rhs,
                group_sizes=group_sizes,
                group_offset=group_offset,
                preferred_element_type=jnp.bfloat16,
                rhs_scale=rhs_scale,
            )

        for _ in range(WARMUP):
            run().block_until_ready()

        times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            run().block_until_ready()
            times.append((time.perf_counter() - t0) * 1000)

        return np.median(times)
    except Exception as e:
        return None, str(e)[:150]


def run_config(config_name, qbs, is_baseline=False):
    """Run one config across all token counts, both GMM1 and GMM2."""
    g = EXPERTS_LOCAL

    rhs_dtype = jnp.bfloat16 if is_baseline else jnp.dtype("float4_e2m1fn")
    lhs_dtype = jnp.bfloat16

    results = {}  # ntok -> (gmm1_ms, gmm2_ms, total_ms)

    for ntok in TOKEN_COUNTS:
        m = ntok * TOPK

        # GMM1: [m, H] @ [g, H, 2I] -> [m, 2I]
        k1, n1 = HIDDEN, INTER * 2
        # GMM2: [m, I] @ [g, I, H] -> [m, H]
        k2, n2 = INTER, HIDDEN

        # Scales
        if is_baseline:
            s1, s2 = None, None
        elif qbs is None:
            # Channelwise: 1 block
            s1 = jnp.ones((g, 1, 1, n1), dtype=jnp.float32)
            s2 = jnp.ones((g, 1, 1, n2), dtype=jnp.float32)
        else:
            nb1 = k1 // qbs
            nb2 = k2 // qbs
            s1 = jnp.ones((g, nb1, 1, n1), dtype=jnp.float32)
            s2 = jnp.ones((g, nb2, 1, n2), dtype=jnp.float32)

        r1 = bench_gmm(m, k1, n1, g, lhs_dtype, rhs_dtype, s1)
        r2 = bench_gmm(m, k2, n2, g, lhs_dtype, rhs_dtype, s2)

        # Handle errors
        if isinstance(r1, tuple):
            print(f"  {config_name:>16}  N={ntok:>5}  GMM1 FAILED: {r1[1]}")
            r1_ms = None
        else:
            r1_ms = r1

        if isinstance(r2, tuple):
            print(f"  {config_name:>16}  N={ntok:>5}  GMM2 FAILED: {r2[1]}")
            r2_ms = None
        else:
            r2_ms = r2

        if r1_ms is not None and r2_ms is not None:
            total = r1_ms + r2_ms
            results[ntok] = (r1_ms, r2_ms, total)
            print(f"  {config_name:>16}  N={ntok:>5}  m={m:>6}  "
                  f"GMM1={r1_ms:6.2f}ms  GMM2={r2_ms:6.2f}ms  "
                  f"total={total:6.2f}ms", flush=True)
        else:
            results[ntok] = None

    return results


def main():
    print(f"JAX: {jax.__version__}, Devices: {jax.device_count()} x "
          f"{jax.devices()[0].device_kind}")
    print(f"DeepSeek-R1: H={HIDDEN}, I={INTER}, local_experts={EXPERTS_LOCAL}, topk={TOPK}")
    print(f"GMM1: [{HIDDEN}, {INTER*2}]  GMM2: [{INTER}, {HIDDEN}]")
    print(f"Warmup={WARMUP}, Iters={ITERS}")
    print(f"Pure gmm_v2 kernel (no permute/gating/activation/comms)")
    print(f"Sweep: subchannel sizes {SUBCHANNEL_SIZES} + channelwise + bf16 baseline")
    print()

    all_results = {}

    # --- bf16×bf16 baseline ---
    name = "bf16×bf16"
    print(f"\n{'='*70}")
    print(f"  {name} (baseline, no scale)")
    print(f"{'='*70}")
    all_results[name] = run_config(name, None, is_baseline=True)

    # --- bf16×fp4 channelwise ---
    name = "channelwise"
    print(f"\n{'='*70}")
    print(f"  bf16×fp4 {name} (scale blocks=1)")
    print(f"{'='*70}")
    all_results[name] = run_config(name, None, is_baseline=False)

    # --- Subchannel sweep ---
    for qbs in SUBCHANNEL_SIZES:
        if HIDDEN % qbs != 0 or INTER % qbs != 0:
            print(f"\n  qbs={qbs}: SKIPPED (doesn't divide H={HIDDEN} or I={INTER})")
            continue

        name = f"sub_{qbs}"
        nb1 = HIDDEN // qbs
        nb2 = INTER // qbs
        print(f"\n{'='*70}")
        print(f"  bf16×fp4 subchannel qbs={qbs}  "
              f"(GMM1: {nb1} blocks, GMM2: {nb2} blocks)")
        print(f"{'='*70}")
        all_results[name] = run_config(name, qbs)

    # --- Summary tables ---
    configs_order = ["bf16×bf16", "channelwise"] + [
        f"sub_{q}" for q in SUBCHANNEL_SIZES
        if HIDDEN % q == 0 and INTER % q == 0
    ]

    for label, idx in [("GMM1 (K=7168,N=4096)", 0),
                       ("GMM2 (K=2048,N=7168)", 1),
                       ("GMM1+GMM2 total", 2)]:
        print(f"\n\n{'='*80}")
        print(f"SUMMARY — {label}  (median ms, pure gmm_v2)")
        print(f"{'='*80}")

        header = f"{'Config':>16}"
        for ntok in TOKEN_COUNTS:
            header += f"  {ntok:>7}"
        print(header)
        print("-" * len(header))

        for cfg in configs_order:
            res = all_results.get(cfg)
            row = f"{cfg:>16}"
            if res is None:
                row += "  FAILED"
            else:
                for ntok in TOKEN_COUNTS:
                    r = res.get(ntok)
                    if r is not None:
                        row += f"  {r[idx]:7.2f}"
                    else:
                        row += f"  {'—':>7}"
            print(row)

    # Slowdown vs channelwise (total)
    cw_res = all_results.get("channelwise")
    if cw_res:
        print(f"\n\n{'='*80}")
        print(f"SLOWDOWN vs channelwise  (GMM1+GMM2 total)")
        print(f"{'='*80}")
        header = f"{'Config':>16}"
        for ntok in TOKEN_COUNTS:
            header += f"  {ntok:>7}"
        print(header)
        print("-" * len(header))

        for cfg in configs_order:
            if cfg == "channelwise":
                continue
            res = all_results.get(cfg)
            row = f"{cfg:>16}"
            if res is None:
                row += "  —"
            else:
                for ntok in TOKEN_COUNTS:
                    r = res.get(ntok)
                    cw = cw_res.get(ntok)
                    if r is not None and cw is not None:
                        ratio = r[2] / cw[2]
                        row += f"  {ratio:6.2f}x"
                    else:
                        row += f"  {'—':>7}"
            print(row)

    print()


if __name__ == "__main__":
    main()
