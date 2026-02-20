#!/usr/bin/env python3
"""Benchmark EP vs TP for fused_moe_func GMM path.

Compares expert parallelism (EP=8) vs tensor parallelism (TP=8)
on the fused_moe_func path with DeepSeek-R1 shapes and fp8 weights.

EP: expert dim sharded across 8 cores, full-rank GMMs, psum_scatter.
    Tokens are replicated (same as production — attention outputs replicated).
TP: weight dims sharded across 8 cores, all experts local, all-reduce.
    Tokens are replicated.
"""

import os, time
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=rematerialization"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func

# DeepSeek-R1 MoE shapes
HIDDEN = 7168
INTER = 2048      # intermediate_size_moe
EXPERTS = 256
TOPK = 8
N_CORES = 8

WEIGHT_DTYPE = jnp.float8_e4m3fn
ACT_DTYPE = jnp.bfloat16
QUANTIZE_ACTIVATION = True

TOKEN_COUNTS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
WARMUP = 10
ITERS = 30


def make_weights(key, mesh, mode):
    """Create weights with appropriate sharding for EP or TP mode.

    EP: w1/w2 shape [E, K, N] — expert dim (0) sharded on "model"
    TP: w1 shape [E, K, N] — N dim (2) sharded on "model"
        w2 shape [E, K, N] — K dim (1) sharded on "model"
    """
    k1, k2 = jax.random.split(key)

    w1_shape = (EXPERTS, HIDDEN, INTER * 2)    # [256, 7168, 4096]
    w2_shape = (EXPERTS, INTER, HIDDEN)        # [256, 2048, 7168]
    w1_scale_shape = (EXPERTS, 1, 1, INTER * 2)  # [256, 1, 1, 4096]
    w2_scale_shape = (EXPERTS, 1, 1, HIDDEN)      # [256, 1, 1, 7168]

    if mode == "ep":
        # Expert dim sharded: each core gets 32 experts, full rank
        w_spec = P("model", None, None)
        w1_sc_spec = P("model", None, None, None)
        w2_sc_spec = P("model", None, None, None)
    else:
        # TP: weight output dims sharded
        # w1: [E, H, 2I] — shard dim 2 (output) on "model"
        # w2: [E, I, H]  — shard dim 1 (input, which was output of w1) on "model"
        w_spec_w1 = P(None, None, "model")
        w_spec_w2 = P(None, "model", None)
        # w1 scale: [E, 1, 1, 2I] — shard last dim to match w1
        w1_sc_spec = P(None, None, None, "model")
        # w2 scale: [E, 1, 1, H] — NOT sharded (H is the output dim of w2,
        # but for channelwise scale with num_blocks=1, w2_scale_spec should
        # match the code's expectation)
        w2_sc_spec = None  # num_blocks=1, code sets w2_scale_spec=None

    w1 = jax.device_put(
        (jax.random.normal(k1, w1_shape, dtype=jnp.bfloat16) / 100
         ).astype(WEIGHT_DTYPE),
        NamedSharding(mesh, w_spec if mode == "ep" else w_spec_w1))
    w2 = jax.device_put(
        (jax.random.normal(k2, w2_shape, dtype=jnp.bfloat16) / 100
         ).astype(WEIGHT_DTYPE),
        NamedSharding(mesh, w_spec if mode == "ep" else w_spec_w2))

    w1_scale = jax.device_put(
        jnp.ones(w1_scale_shape, dtype=jnp.float32),
        NamedSharding(mesh, w1_sc_spec))
    w2_scale = jax.device_put(
        jnp.ones(w2_scale_shape, dtype=jnp.float32),
        NamedSharding(mesh, w2_sc_spec if w2_sc_spec is not None
                      else P()))

    return w1, w2, w1_scale, w2_scale


def benchmark_mode(mode, mesh, w1, w2, w1_scale, w2_scale, tok_s,
                   use_gmm_gather=True):
    """Benchmark one mode (ep or tp) across all token counts."""
    use_ep = (mode == "ep")
    results = {}

    for ntok in TOKEN_COUNTS:
        key = jax.random.PRNGKey(ntok)
        k1, k2 = jax.random.split(key)

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
                mesh=mesh, use_ep=use_ep, activation="silu",
                scoring_fn="sigmoid", quantize_activation=QUANTIZE_ACTIVATION,
                use_gmm_gather=use_gmm_gather,
            )

        # Warmup (first call compiles)
        for i in range(WARMUP):
            run_once().block_until_ready()
            if i == 0:
                print(f"    ntok={ntok}: compiled")

        # Timed iterations
        times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            run_once().block_until_ready()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        med = np.median(times)
        mn = np.min(times)
        mx = np.max(times)
        print(f"  {ntok:>6} {ntok*TOPK:>7}  {med:>10.2f}  {mn:>8.2f}  {mx:>8.2f}")
        results[ntok] = med

    return results


def run_config(label, mode, mesh, key, tok_s, use_gmm_gather):
    """Run one config: create weights, benchmark, free weights."""
    gather_str = "gmm_gather" if use_gmm_gather else "gmm"
    if mode == "ep":
        shapes = "[32, 7168, 4096] and [32, 2048, 7168]"
    else:
        shapes = "[256, 7168, 512] and [256, 256, 7168]"

    print(f"\n{'='*60}")
    print(f"  {label}: {mode.upper()}=8, kernel={gather_str}")
    print(f"  GMM shapes per core: {shapes}")
    print(f"{'='*60}")
    print(f"  {'N':>6} {'M':>7}  {'median_ms':>10}  {'min_ms':>8}  {'max_ms':>8}")
    print(f"  {'-'*48}")

    w1, w2, w1s, w2s = make_weights(key, mesh, mode)
    results = benchmark_mode(mode, mesh, w1, w2, w1s, w2s, tok_s,
                             use_gmm_gather=use_gmm_gather)
    del w1, w2, w1s, w2s
    return results


def main():
    print(f"JAX: {jax.__version__}, "
          f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    print(f"DeepSeek-R1 MoE: H={HIDDEN}, I={INTER}, E={EXPERTS}, topk={TOPK}")

    devices = np.array(jax.devices()[:N_CORES]).reshape(1, N_CORES)
    mesh = Mesh(devices, ("data", "model"))

    # Production: tokens are replicated (attention outputs P(None, None)).
    # Both EP and TP see the same replicated input tokens.
    tok_s = NamedSharding(mesh, P(None, None))

    key = jax.random.PRNGKey(42)

    configs = [
        ("EP+gmm",         "ep", tok_s, False),
        ("TP+gmm",         "tp", tok_s, False),
        ("EP+gmm_gather",  "ep", tok_s, True),
        ("TP+gmm_gather",  "tp", tok_s, True),
    ]

    all_results = {}
    for label, mode, tok_s, use_gather in configs:
        all_results[label] = run_config(label, mode, mesh, key, tok_s,
                                        use_gather)

    # ── Summary ──
    labels = [c[0] for c in configs]
    print(f"\n{'='*90}")
    print(f"  Summary — median ms (lower is better)")
    print(f"{'='*90}")
    header = f"{'tokens':>6} {'M':>7}"
    for l in labels:
        header += f"  {l:>14}"
    header += f"  {'best':>14}"
    print(header)
    print("-" * 90)
    for ntok in TOKEN_COUNTS:
        row = f"{ntok:>6} {ntok*TOPK:>7}"
        vals = {}
        for l in labels:
            v = all_results[l].get(ntok)
            if v is not None:
                row += f"  {v:>14.2f}"
                vals[l] = v
            else:
                row += f"  {'---':>14}"
        if vals:
            best = min(vals, key=vals.get)
            row += f"  {best:>14}"
        print(row)

    print("\nDONE")


if __name__ == "__main__":
    main()
