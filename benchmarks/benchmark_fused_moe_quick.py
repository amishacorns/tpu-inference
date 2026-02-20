#!/usr/bin/env python3
"""A/B benchmark: pre-quantized fp8 activation for GMM2.

Compares bf16×fp8 (today) vs fp8×fp8 (pre-quantize activation after SiLU)
on the full fused_moe_func EP path with channelwise fp8 weights.
"""

import os, time, sys
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=rematerialization"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func

HIDDEN, INTER, EXPERTS, TOPK, EP = 7168, 2048, 256, 8, 8
WEIGHT_DTYPE = jnp.float8_e4m3fn
ACT_DTYPE = jnp.bfloat16  # activations are bf16 in production
TOKEN_COUNTS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
WARMUP = 10
ITERS = 30


def main():
    print(f"JAX: {jax.__version__}, Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    devices = np.array(jax.devices()[:EP]).reshape(1, EP)
    mesh = Mesh(devices, ("data", "model"))

    key = jax.random.PRNGKey(0)
    k1, k2, key = jax.random.split(key, 3)

    # Channelwise fp8 weights with scales (matches DeepSeek-R1 production)
    # weight_block_size=[1, K] → one scale per row per expert
    experts_per_dev = EXPERTS // EP
    w1_shape = (EXPERTS, HIDDEN, INTER * 2)
    w2_shape = (EXPERTS, INTER, HIDDEN)
    w_s = NamedSharding(mesh, P("model", None, None))

    w1 = jax.device_put(
        (jax.random.normal(k1, w1_shape, dtype=jnp.bfloat16) / 100).astype(WEIGHT_DTYPE),
        w_s)
    w2 = jax.device_put(
        (jax.random.normal(k2, w2_shape, dtype=jnp.bfloat16) / 100).astype(WEIGHT_DTYPE),
        w_s)

    # Channelwise scales: [num_experts, num_blocks=1, 1, N]
    # num_blocks=1 because quant_block_size=K (whole row)
    w1_scale_shape = (EXPERTS, 1, 1, INTER * 2)
    w2_scale_shape = (EXPERTS, 1, 1, HIDDEN)
    w1_scale = jax.device_put(
        jnp.ones(w1_scale_shape, dtype=jnp.float32), w_s[:3] if False else
        NamedSharding(mesh, P("model", None, None, None)))
    w2_scale = jax.device_put(
        jnp.ones(w2_scale_shape, dtype=jnp.float32),
        NamedSharding(mesh, P("model", None, None, None)))

    tok_s = NamedSharding(mesh, P("model", None))

    configs = [
        ("bf16×fp8 (baseline)", False),
        ("fp8×fp8 (quantize_activation)", True),
    ]

    # Collect results for summary
    all_results = {name: {} for name, _ in configs}

    for config_name, q_act in configs:
        print(f"\n{'='*60}")
        print(f"  {config_name}")
        print(f"{'='*60}")
        print(f"{'N':>6} {'M':>7}  {'median_ms':>10}  {'min_ms':>8}  {'max_ms':>8}")
        print("-" * 50)

        for ntok in TOKEN_COUNTS:
            k1, k2, key = jax.random.split(key, 3)
            tokens = jax.device_put(
                (jax.random.normal(k1, (ntok, HIDDEN), dtype=jnp.bfloat16) / 10
                 ).astype(ACT_DTYPE), tok_s)
            gating = jax.device_put(
                jax.random.normal(k2, (ntok, EXPERTS), dtype=jnp.bfloat16), tok_s)

            def run_once():
                return fused_moe_func(
                    hidden_states=tokens, w1=w1, w2=w2,
                    w1_scale=w1_scale, w2_scale=w2_scale,
                    w1_bias=None, w2_bias=None,
                    gating_output=gating, topk=TOPK, renormalize=True,
                    mesh=mesh, use_ep=True, activation="silu",
                    scoring_fn="sigmoid", quantize_activation=q_act,
                )

            # Warmup
            for _ in range(WARMUP):
                run_once().block_until_ready()

            # Benchmark
            times = []
            for _ in range(ITERS):
                t0 = time.perf_counter()
                run_once().block_until_ready()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            med = np.median(times)
            mn = np.min(times)
            mx = np.max(times)
            print(f"{ntok:>6} {ntok*TOPK:>7}  {med:>10.2f}  {mn:>8.2f}  {mx:>8.2f}")
            all_results[config_name][ntok] = med

    # Summary table
    print(f"\n{'='*70}")
    print(f"  Summary: bf16×fp8 vs fp8×fp8 (quantize_activation) — median ms")
    print(f"{'='*70}")
    baseline_name = configs[0][0]
    qlhs_name = configs[1][0]
    print(f"{'tokens':>6} {'M':>7}  {'bf16×fp8':>10}  {'fp8×fp8':>10}  {'speedup':>8}")
    print("-" * 55)
    for ntok in TOKEN_COUNTS:
        bl = all_results[baseline_name].get(ntok)
        ql = all_results[qlhs_name].get(ntok)
        if bl and ql:
            speedup = (bl - ql) / bl * 100
            print(f"{ntok:>6} {ntok*TOPK:>7}  {bl:>10.2f}  {ql:>10.2f}  {speedup:>7.1f}%")
    print("DONE")


if __name__ == "__main__":
    main()
