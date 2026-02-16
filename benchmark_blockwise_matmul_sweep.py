"""Sweep blockwise quantized matmul block sizes for DeepSeek-R1 shared expert shapes on TPU v7.

Sweeps valid (batch_block_size, out_block_size, in_block_size, n_lane_multiplier)
combinations for the shared expert's per-chip shapes after TP=8 sharding,
benchmarks each, and prints optimal configs for tuned_block_sizes.py.

DeepSeek-R1 shared expert (TP=8):
  gate_proj: x[bs, 7168] @ w[18432, 7168]^T  → per-chip w=[2304, 7168]  n_out=2304, n_in=7168
  up_proj:   x[bs, 7168] @ w[18432, 7168]^T  → per-chip w=[2304, 7168]  n_out=2304, n_in=7168
  down_proj: x[bs, 18432] @ w[7168, 18432]^T → per-chip x=[bs,2304]     n_out=7168, n_in=2304

Uses channelwise quantization (block_size = n_in, so w_scale has shape [1, 1, n_out]).

Usage:
  python benchmark_blockwise_matmul_sweep.py
"""

import os
import sys
import time
import functools

os.environ["PYTHONUNBUFFERED"] = "1"

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.quantized_matmul.blockwise_kernel import (
    quantized_matmul_kernel, MXU_SIZE,
)
from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import TunedValue

# ── Shapes: DeepSeek-R1 shared expert after TP=8 ──
# gate/up projections: n_out=2304, n_in=7168 (channelwise)
# down projection:     n_out=7168, n_in=2304 (channelwise)
SHAPES = {
    "gate_up": {"n_out": 2304, "n_in": 7168, "label": "gate/up proj"},
    "down":    {"n_out": 7168, "n_in": 2304, "label": "down proj"},
}

# Batch sizes to sweep — these are the token counts that hit the shared expert
BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

WARMUP = 3
ITERS = 5


def next_multiple(x, m):
    return ((x + m - 1) // m) * m


def divisors_of(n, min_val=1):
    """All divisors of n >= min_val."""
    divs = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            if i >= min_val:
                divs.add(i)
            if n // i >= min_val:
                divs.add(n // i)
    return sorted(divs)


def generate_candidates(n_batch, n_out, n_in):
    """Generate valid (batch_block_size, out_block_size, in_block_size, n_lane_multiplier) tuples.

    Constraints from blockwise_kernel.py:
    - out_block_size >= MXU_SIZE * n_lane_multiplier (256 for nlm=1)
      so steps_n = out_block_size // (MXU_SIZE * nlm) >= 1
    - batch_block_size must divide padded_n_batch
    - out_block_size must divide padded_n_out
    - in_block_size must divide padded_n_in
    - For channelwise: block_size becomes in_block_size, steps_k = 1
    """
    candidates = []

    # n_lane_multiplier options
    for nlm in [1, 2, 4]:
        min_obs = MXU_SIZE * nlm  # minimum out_block_size

        # out_block_size: multiples of min_obs that divide n_out (or padded n_out)
        # Since we want to avoid padding, prefer divisors of n_out
        obs_cands = set()
        for obs in range(min_obs, n_out + 1, min_obs):
            # Check it divides n_out or a reasonable padded version
            padded = next_multiple(n_out, obs)
            # Don't pad more than 50%
            if padded <= n_out * 1.5:
                obs_cands.add(obs)
        # Also try n_out itself if >= min_obs
        if n_out >= min_obs:
            obs_cands.add(n_out)

        # in_block_size: divisors of n_in that are multiples of 128
        ibs_cands = set()
        for d in divisors_of(n_in, 128):
            if d % 128 == 0:
                ibs_cands.add(d)
        if not ibs_cands:
            ibs_cands.add(n_in)

        # batch_block_size: powers of 2 from 16 to min(n_batch, 2048)
        bbs_cands = set()
        bbs = 16
        while bbs <= min(n_batch, 2048):
            bbs_cands.add(bbs)
            bbs *= 2
        # also try n_batch itself if small
        if n_batch <= 2048:
            bbs_cands.add(n_batch)

        for obs in sorted(obs_cands):
            for ibs in sorted(ibs_cands):
                for bbs in sorted(bbs_cands):
                    candidates.append((bbs, obs, ibs, nlm))

    return candidates


def try_benchmark(x, w_q, w_scale, block_size, tuned_value, warmup=WARMUP, iters=ITERS):
    """Benchmark one config. Returns median ms or None on error."""
    try:
        # Warmup (also catches OOM)
        for _ in range(warmup):
            out = quantized_matmul_kernel(
                x, w_q, w_scale,
                block_size=block_size,
                x_q_dtype=jnp.float8_e4m3fn,
                tuned_value=tuned_value,
            )
            out.block_until_ready()

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = quantized_matmul_kernel(
                x, w_q, w_scale,
                block_size=block_size,
                x_q_dtype=jnp.float8_e4m3fn,
                tuned_value=tuned_value,
            )
            out.block_until_ready()
            times.append(time.perf_counter() - t0)

        # Clear JIT cache
        quantized_matmul_kernel.clear_cache()
        return np.median(times) * 1e3
    except Exception as e:
        quantized_matmul_kernel.clear_cache()
        err = str(e)[:200]
        if "RESOURCE_EXHAUSTED" not in err and "out of memory" not in err.lower():
            print(f"      ERROR: {type(e).__name__}: {err}")
        return None


def sweep_one_shape(name, n_out, n_in, n_batch):
    """Sweep all candidate block sizes for one matmul shape."""
    candidates = generate_candidates(n_batch, n_out, n_in)
    print(f"    {name} (bs={n_batch}, n_out={n_out}, n_in={n_in}): "
          f"{len(candidates)} candidates")

    # Create test data — channelwise quantization
    key = jax.random.key(42)
    k1, k2 = jax.random.split(key, 2)

    x = jax.random.normal(k1, (n_batch, n_in), dtype=jnp.bfloat16) / 10.0
    w_q = (jax.random.normal(k2, (n_out, n_in), dtype=jnp.bfloat16) / 10.0).astype(jnp.float8_e4m3fn)
    # Channelwise scale: [1, 1, n_out]
    w_scale = jnp.ones((1, 1, n_out), dtype=jnp.bfloat16)

    # block_size = n_in for channelwise
    block_size = n_in

    best_ms = float('inf')
    best_cfg = None
    tried = 0
    successes = 0

    for bbs, obs, ibs, nlm in candidates:
        tuned_value = TunedValue(
            batch_block_size=bbs,
            out_block_size=obs,
            in_block_size=ibs,
            n_lane_multiplier=nlm,
        )

        tried += 1
        ms = try_benchmark(x, w_q, w_scale, block_size, tuned_value)

        if ms is not None:
            successes += 1
            marker = ""
            if ms < best_ms:
                best_ms = ms
                best_cfg = (bbs, obs, ibs, nlm, ms)
                marker = " ★"
            flops = 2 * n_batch * n_out * n_in
            tflops = flops / (ms * 1e-3) / 1e12
            if successes <= 3 or marker:
                print(f"      [{successes:>3}] bbs={bbs:>5} obs={obs:>5} ibs={ibs:>5} nlm={nlm} "
                      f"→ {ms:.3f}ms {tflops:.1f}TF/s{marker}")
        else:
            if tried <= 2:
                print(f"      [{tried:>3}] bbs={bbs:>5} obs={obs:>5} ibs={ibs:>5} nlm={nlm} → FAIL")

    if best_cfg:
        bbs, obs, ibs, nlm, ms = best_cfg
        flops = 2 * n_batch * n_out * n_in
        tflops = flops / (ms * 1e-3) / 1e12
        nlm_str = f", {nlm}" if nlm > 1 else ""
        print(f"    ★ {name} BEST: ({bbs}, {obs}, {ibs}{nlm_str}) → {ms:.3f}ms ({tflops:.1f} TF/s)")
    else:
        print(f"    ✗ {name}: ALL {tried} FAILED")

    return best_cfg


def main():
    print(f"TPU: {jax.devices()[0]}")
    print(f"Devices: {len(jax.devices())}")
    print(f"Kernel: blockwise_quantized_matmul_kernel (MXU_SIZE={MXU_SIZE})")
    print(f"Quantization: channelwise fp8×fp8 (x_q_dtype=float8_e4m3fn, w_q_dtype=float8_e4m3fn)")
    print(f"DeepSeek-R1 shared expert shapes after TP=8:")
    for name, shape in SHAPES.items():
        print(f"  {name}: n_out={shape['n_out']}, n_in={shape['n_in']} ({shape['label']})")
    print()

    results = {}

    for n_batch in BATCH_SIZES:
        print(f"═══ n_batch={n_batch} ═══")
        for name, shape in SHAPES.items():
            best = sweep_one_shape(name, shape["n_out"], shape["n_in"], n_batch)
            if best:
                results[(name, n_batch)] = best
        print()

    # ── Summary ──
    print()
    print("=" * 110)
    print("SUMMARY: Optimal block sizes for DeepSeek-R1 shared expert (TP=8) on TPU v7")
    print("=" * 110)
    print()
    print(f"{'n_batch':>8} {'shape':>8} {'n_out':>6} {'n_in':>6} "
          f"{'bbs':>5} {'obs':>5} {'ibs':>5} {'nlm':>4} "
          f"{'ms':>9} {'TF/s':>8}")
    print("-" * 90)

    for n_batch in BATCH_SIZES:
        for name, shape in SHAPES.items():
            key = (name, n_batch)
            if key not in results:
                print(f"{n_batch:>8} {name:>8} {shape['n_out']:>6} {shape['n_in']:>6}  --- FAILED ---")
                continue
            bbs, obs, ibs, nlm, ms = results[key]
            flops = 2 * n_batch * shape["n_out"] * shape["n_in"]
            tflops = flops / (ms * 1e-3) / 1e12
            print(f"{n_batch:>8} {name:>8} {shape['n_out']:>6} {shape['n_in']:>6} "
                  f"{bbs:>5} {obs:>5} {ibs:>5} {nlm:>4} "
                  f"{ms:>9.3f} {tflops:>8.1f}")

    # ── tuned_block_sizes.py entries ──
    print()
    print("=" * 110)
    print("TUNED_BLOCK_SIZES entries (copy into tuned_block_sizes.py):")
    print("=" * 110)
    print()

    for n_batch in BATCH_SIZES:
        for name, shape in SHAPES.items():
            key = (name, n_batch)
            if key not in results:
                continue
            bbs, obs, ibs, nlm, ms = results[key]
            n_out, n_in = shape["n_out"], shape["n_in"]
            nlm_str = f", {nlm}" if nlm > 1 else ""
            flops = 2 * n_batch * n_out * n_in
            tflops = flops / (ms * 1e-3) / 1e12
            print(f"    (7, {n_batch}, {n_out}, {n_in}, "
                  f"'float8_e4m3fn', 'float8_e4m3fn'): "
                  f"({bbs}, {obs}, {ibs}{nlm_str}),  "
                  f"# {name} {ms:.3f}ms {tflops:.1f}TF/s")

    print()
    print("DONE")


if __name__ == "__main__":
    main()
