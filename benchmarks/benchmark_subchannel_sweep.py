#!/usr/bin/env python3
"""Sweep subchannel (quant block) size for bf16 × fp4 GMM via fused_moe_func.

Sweeps quant_block_size from 16 to 2048 in powers of two, plus channelwise
(scale has 1 block = no subchannel) and no-scale bf16×bf16 baseline.

All configs use EP=8, DeepSeek-R1 shapes (H=7168, I=2048, E=256, topk=8).
"""

import os, time, sys
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=rematerialization"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func

# DeepSeek-R1 shapes
HIDDEN = 7168
INTER = 2048
EXPERTS = 256
TOPK = 8
EP = 8

TOKEN_COUNTS = [128, 256, 512, 1024, 2048, 4096, 8192]
WARMUP = 5
ITERS = 20

# Subchannel sizes: powers of two from 16 to 2048, plus channelwise and bf16 baseline
SUBCHANNEL_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]


def make_weights(key, shape, dtype):
    w = jax.random.normal(key, shape, dtype=jnp.bfloat16) * 0.01
    return w.astype(dtype)


def make_scale(shape):
    return jnp.ones(shape, dtype=jnp.float32)


def run_config(mesh, config_name, quant_block_size, is_baseline=False):
    """Benchmark one subchannel config in EP mode."""
    key = jax.random.PRNGKey(42)

    rhs_dtype = jnp.bfloat16 if is_baseline else jnp.dtype("float4_e2m1fn")
    lhs_dtype = jnp.bfloat16

    w1_shape = (EXPERTS, HIDDEN, INTER * 2)
    w2_shape = (EXPERTS, INTER, HIDDEN)

    w_shard = NamedSharding(mesh, P("model", None, None))
    scale_shard = NamedSharding(mesh, P("model", None, None, None))

    k1, k2, key = jax.random.split(key, 3)
    w1 = jax.device_put(make_weights(k1, w1_shape, rhs_dtype), w_shard)
    w2 = jax.device_put(make_weights(k2, w2_shape, rhs_dtype), w_shard)

    w1_scale = None
    w2_scale = None
    if not is_baseline:
        if quant_block_size is None:
            # Channelwise: 1 block over K
            w1_scale_shape = (EXPERTS, 1, 1, INTER * 2)
            w2_scale_shape = (EXPERTS, 1, 1, HIDDEN)
        else:
            # Subchannel: num_blocks = K // quant_block_size
            w1_num_blocks = HIDDEN // quant_block_size
            w2_num_blocks = INTER // quant_block_size
            w1_scale_shape = (EXPERTS, w1_num_blocks, 1, INTER * 2)
            w2_scale_shape = (EXPERTS, w2_num_blocks, 1, HIDDEN)
        w1_scale = jax.device_put(make_scale(w1_scale_shape), scale_shard)
        w2_scale = jax.device_put(make_scale(w2_scale_shape), scale_shard)

    tok_shard = NamedSharding(mesh, P("model", None))

    results = {}
    for ntok in TOKEN_COUNTS:
        m = ntok * TOPK
        if m % 16 != 0:
            continue

        k1, k2, key = jax.random.split(key, 3)
        tokens = jax.device_put(
            (jax.random.normal(k1, (ntok, HIDDEN), dtype=jnp.bfloat16) * 0.1),
            tok_shard)
        gating = jax.device_put(
            jax.random.normal(k2, (ntok, EXPERTS), dtype=jnp.bfloat16), tok_shard)

        def run_once():
            return fused_moe_func(
                hidden_states=tokens, w1=w1, w2=w2,
                w1_scale=w1_scale, w2_scale=w2_scale,
                w1_bias=None, w2_bias=None,
                gating_output=gating, topk=TOPK, renormalize=True,
                mesh=mesh, use_ep=True, activation="silu",
                scoring_fn="sigmoid",
            )

        for _ in range(WARMUP):
            run_once().block_until_ready()

        times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            run_once().block_until_ready()
            times.append((time.perf_counter() - t0) * 1000)

        med = np.median(times)
        results[ntok] = med
        print(f"  {config_name:>16}  N={ntok:>5}  m={m:>6}  "
              f"median={med:6.2f}ms  min={min(times):6.2f}  max={max(times):6.2f}",
              flush=True)

    return results


def main():
    print(f"JAX: {jax.__version__}, Devices: {jax.device_count()} x "
          f"{jax.devices()[0].device_kind}")
    print(f"DeepSeek-R1: H={HIDDEN}, I={INTER}, E={EXPERTS}, topk={TOPK}, EP={EP}")
    print(f"Warmup={WARMUP}, Iters={ITERS}")
    print(f"Sweep: subchannel sizes {SUBCHANNEL_SIZES} + channelwise + bf16 baseline")
    print()

    devices = np.array(jax.devices()[:EP]).reshape(1, EP)
    mesh = Mesh(devices, ("data", "model"))

    all_results = {}

    # --- bf16×bf16 baseline ---
    config_name = "bf16×bf16"
    print(f"\n{'='*70}")
    print(f"  {config_name} (baseline, no scale)")
    print(f"{'='*70}")
    try:
        all_results[config_name] = run_config(mesh, config_name, None, is_baseline=True)
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results[config_name] = None

    # --- bf16×fp4 channelwise (no subchannel) ---
    config_name = "channelwise"
    print(f"\n{'='*70}")
    print(f"  bf16×fp4 {config_name} (scale blocks=1)")
    print(f"{'='*70}")
    try:
        all_results[config_name] = run_config(mesh, config_name, None, is_baseline=False)
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results[config_name] = None

    # --- Subchannel sweep ---
    for qbs in SUBCHANNEL_SIZES:
        # Check divisibility
        if HIDDEN % qbs != 0 or INTER % qbs != 0:
            print(f"\n  qbs={qbs}: SKIPPED (doesn't divide H={HIDDEN} or I={INTER})")
            continue

        config_name = f"sub_{qbs}"
        w1_blocks = HIDDEN // qbs
        w2_blocks = INTER // qbs
        print(f"\n{'='*70}")
        print(f"  bf16×fp4 subchannel qbs={qbs}  "
              f"(GMM1: {w1_blocks} blocks, GMM2: {w2_blocks} blocks)")
        print(f"{'='*70}")
        try:
            all_results[config_name] = run_config(mesh, config_name, qbs)
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results[config_name] = None

    # --- Summary table ---
    print(f"\n\n{'='*80}")
    print("SUMMARY — subchannel sweep  (median ms, EP mode)")
    print(f"{'='*80}")

    configs_order = ["bf16×bf16", "channelwise"] + [f"sub_{q}" for q in SUBCHANNEL_SIZES
                                                     if HIDDEN % q == 0 and INTER % q == 0]

    header = f"{'Config':>16}"
    for ntok in TOKEN_COUNTS:
        header += f"  {ntok:>7}"
    print(header)
    print("-" * len(header))

    for config_name in configs_order:
        res = all_results.get(config_name)
        row = f"{config_name:>16}"
        if res is None:
            row += "  FAILED"
        else:
            for ntok in TOKEN_COUNTS:
                if ntok in res:
                    row += f"  {res[ntok]:7.2f}"
                else:
                    row += f"  {'—':>7}"
        print(row)

    # Speedup vs channelwise
    cw_res = all_results.get("channelwise")
    if cw_res:
        print()
        print(f"{'Slowdown vs CW':>16}", end="")
        for ntok in TOKEN_COUNTS:
            print(f"  {'':>7}", end="")
        print()
        print("-" * len(header))
        for config_name in configs_order:
            if config_name == "channelwise":
                continue
            res = all_results.get(config_name)
            row = f"{config_name:>16}"
            if res is None:
                row += "  —"
            else:
                for ntok in TOKEN_COUNTS:
                    if ntok in res and ntok in cw_res:
                        ratio = res[ntok] / cw_res[ntok]
                        row += f"  {ratio:6.2f}x"
                    else:
                        row += f"  {'—':>7}"
            print(row)

    print()


if __name__ == "__main__":
    main()
