#!/usr/bin/env python3
"""Measure EP MoE performance under varying load imbalance.

Tests TWO kernels:
  1. GMM EP   — replicate-all-tokens + psum (current default)
  2. Fused EP — point-to-point all-to-all DMA inside a single Pallas kernel

Forces different routing patterns via crafted gating logits:
  balanced  - each token picks exactly 1 expert per device (perfect balance)
  natural   - random gating logits (realistic baseline)
  skew_1.2x - ~1.2× hottest device load (matches real DeepSeek-R1 routing)
  skew_1.5x - ~1.5× hottest device load
  skew_2x   - 2 of 8 topk picks land on device 0 (2× hottest device load)
  skew_4x   - 4 of 8 topk picks land on device 0 (4× hottest)
  skew_8x   - all 8 picks on device 0 (8× hottest, worst case)

All configs: EP=8, DeepSeek-R1 shapes, fp4 subchannel qbs=256.
"""

import os, time, sys
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=rematerialization"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_PALLAS_DUMP_MLIR"] = "0"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func
from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe

# DeepSeek-R1 shapes
HIDDEN = 7168
INTER = 2048
EXPERTS = 256
TOPK = 8
EP = 8
EXPERTS_PER_DEVICE = EXPERTS // EP  # 32

QBS = 256  # subchannel quant block size

TOKEN_COUNTS = [128, 256, 512, 1024, 2048, 4096, 8192]
WARMUP = 5
ITERS = 30


# ---- Gating pattern generators ----

def make_balanced_gating(rng, ntok):
    """Each token picks exactly 1 expert from each of the 8 devices.
    Result: every device processes exactly N tokens (perfect balance).
    """
    logits = np.full((ntok, EXPERTS), -100.0, dtype=np.float32)
    for i in range(ntok):
        for dev in range(EP):
            expert_idx = dev * EXPERTS_PER_DEVICE + rng.integers(EXPERTS_PER_DEVICE)
            logits[i, expert_idx] = 10.0 + rng.random()
    return jnp.array(logits, dtype=jnp.bfloat16)


def make_natural_gating(rng, ntok):
    """Random gating logits — realistic baseline."""
    return jnp.array(
        rng.standard_normal((ntok, EXPERTS)).astype(np.float32),
        dtype=jnp.bfloat16,
    )


def make_skewed_gating(rng, ntok, hot_picks):
    """Force `hot_picks` of the 8 topk selections onto device 0."""
    logits = np.full((ntok, EXPERTS), -100.0, dtype=np.float32)
    cold_picks = TOPK - hot_picks

    for i in range(ntok):
        hot_experts = rng.choice(EXPERTS_PER_DEVICE, size=hot_picks, replace=False)
        for e in hot_experts:
            logits[i, e] = 10.0 + rng.random()

        if cold_picks > 0:
            cold_devices = rng.choice(np.arange(1, EP), size=cold_picks, replace=False)
            for dev in cold_devices:
                expert_idx = dev * EXPERTS_PER_DEVICE + rng.integers(EXPERTS_PER_DEVICE)
                logits[i, expert_idx] = 5.0 + rng.random()

    return jnp.array(logits, dtype=jnp.bfloat16)


def make_fractional_skew_gating(rng, ntok, target_ratio):
    """Craft gating that gives approx `target_ratio` max/avg device load."""
    p = (target_ratio - 1.0) / (1.0 - target_ratio / EP)
    p = np.clip(p, 0.0, 1.0)

    logits = np.full((ntok, EXPERTS), -100.0, dtype=np.float32)
    for i in range(ntok):
        extra_on_dev0 = 1 if rng.random() < p else 0
        picks_dev0 = 1 + extra_on_dev0
        picks_rest = TOPK - picks_dev0

        dev0_experts = rng.choice(EXPERTS_PER_DEVICE, size=picks_dev0, replace=False)
        for e in dev0_experts:
            logits[i, e] = 10.0 + rng.random()

        cold_devices = rng.choice(np.arange(1, EP), size=picks_rest, replace=False)
        for dev in cold_devices:
            expert_idx = dev * EXPERTS_PER_DEVICE + rng.integers(EXPERTS_PER_DEVICE)
            logits[i, expert_idx] = 5.0 + rng.random()

    return jnp.array(logits, dtype=jnp.bfloat16)


def measure_device_loads(gating, ep, experts_per_device, topk):
    """Compute per-device token-expert slot counts from gating logits."""
    from scipy.special import expit
    gating_np = np.array(gating, dtype=np.float32)
    weights = expit(gating_np)
    topk_indices = np.argsort(-weights, axis=-1)[:, :topk]
    device_counts = np.zeros(ep)
    for dev in range(ep):
        mask = (topk_indices >= dev * experts_per_device) & \
               (topk_indices < (dev + 1) * experts_per_device)
        device_counts[dev] = mask.sum()
    return device_counts


# ---- Weight setup ----

def make_gmm_weights(mesh, key):
    """Create weights for the GMM EP kernel.
    
    GMM path weight shapes:
      w1: (E, H, I*2)  — gate+up fused on last dim
      w2: (E, I, H)
      w1_scale: (E, H//QBS, 1, I*2)
      w2_scale: (E, I//QBS, 1, H)
    """
    rhs_dtype = jnp.dtype("float4_e2m1fn")
    w_shard = NamedSharding(mesh, P("model", None, None))
    scale_shard = NamedSharding(mesh, P("model", None, None, None))

    k1, k2, key = jax.random.split(key, 3)
    w1 = jax.device_put(
        (jax.random.normal(k1, (EXPERTS, HIDDEN, INTER * 2), dtype=jnp.bfloat16) * 0.01).astype(rhs_dtype),
        w_shard)
    w2 = jax.device_put(
        (jax.random.normal(k2, (EXPERTS, INTER, HIDDEN), dtype=jnp.bfloat16) * 0.01).astype(rhs_dtype),
        w_shard)

    w1_scale = jax.device_put(
        jnp.ones((EXPERTS, HIDDEN // QBS, 1, INTER * 2), dtype=jnp.float32),
        scale_shard)
    w2_scale = jax.device_put(
        jnp.ones((EXPERTS, INTER // QBS, 1, HIDDEN), dtype=jnp.float32),
        scale_shard)

    return w1, w2, w1_scale, w2_scale, key


def make_fused_weights(mesh, key):
    """Create weights for the Fused EP MoE kernel.
    
    Fused kernel weight shapes:
      w1: (E, 2, H, I)  — gate and up on dim 1
      w2: (E, I, H)
      w1_scale: (E, 2, H//QBS, 1, I)
      w2_scale: (E, I//QBS, 1, H)
    """
    rhs_dtype = jnp.dtype("float4_e2m1fn")
    w1_shard = NamedSharding(mesh, P("model", None, None, None))
    w2_shard = NamedSharding(mesh, P("model", None, None))
    w1_scale_shard = NamedSharding(mesh, P("model", None, None, None, None))
    w2_scale_shard = NamedSharding(mesh, P("model", None, None, None))

    k1, k2, key = jax.random.split(key, 3)
    w1 = jax.device_put(
        (jax.random.normal(k1, (EXPERTS, 2, HIDDEN, INTER), dtype=jnp.bfloat16) * 0.01).astype(rhs_dtype),
        w1_shard)
    w2 = jax.device_put(
        (jax.random.normal(k2, (EXPERTS, INTER, HIDDEN), dtype=jnp.bfloat16) * 0.01).astype(rhs_dtype),
        w2_shard)

    w1_scale = jax.device_put(
        jnp.ones((EXPERTS, 2, HIDDEN // QBS, 1, INTER), dtype=jnp.float32),
        w1_scale_shard)
    w2_scale = jax.device_put(
        jnp.ones((EXPERTS, INTER // QBS, 1, HIDDEN), dtype=jnp.float32),
        w2_scale_shard)

    return w1, w2, w1_scale, w2_scale, key


# ---- Benchmark runners ----

def run_gmm_ep(tokens, gating, w1, w2, w1_scale, w2_scale, mesh):
    """Run one iteration of the GMM EP kernel."""
    return fused_moe_func(
        hidden_states=tokens, w1=w1, w2=w2,
        w1_scale=w1_scale, w2_scale=w2_scale,
        w1_bias=None, w2_bias=None,
        gating_output=gating, topk=TOPK, renormalize=True,
        mesh=mesh, use_ep=True, activation="silu",
        scoring_fn="sigmoid",
    )


def run_fused_ep(tokens, gating, w1, w2, w1_scale, w2_scale, mesh):
    """Run one iteration of the Fused EP MoE kernel."""
    # fp4 packing = 2, so t_packing=2
    # Block sizes must divide hidden_size=7168 and intermediate_size=2048.
    # 7168 = 1024*7, so valid divisors >= 128: 128, 256, 512, 1024, 3584, 7168
    # bd1c = subc_quant_w1_sz * t_packing = 256 * 2 = 512
    # bd2c = subc_quant_w1_sz * t_packing = 256 * 2 = 512 (but 512 must divide bd2)
    local_ntok = tokens.shape[0]  # already sharded
    bt = min(local_ntok, 128)
    return fused_ep_moe(
        mesh=mesh,
        tokens=tokens,
        w1=w1,
        w2=w2,
        gating_output=gating,
        top_k=TOPK,
        renormalize_topk_logits=True,
        act_fn="silu",
        scoring_fn="sigmoid",
        subc_quant_w1_sz=QBS,
        subc_quant_w2_sz=QBS,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        b1=None,
        b2=None,
        bt=bt,
        bf=1024,
        bd1=1024,
        bd2=1024,
        btc=min(bt, 8),
        bfc=QBS,        # 256, divides bf=1024
        bd1c=QBS * 2,   # 512, divides bd1=1024, aligned to t_packing*128=256
        bd2c=QBS * 2,   # 512, divides bd2=1024, aligned to t_packing*128=256
        ep_axis_name="model",
    )


def benchmark_kernel(kernel_name, run_fn, tokens, gating, w1, w2, w1_scale, w2_scale, mesh):
    """Warmup + timed iterations, return median ms."""
    for _ in range(WARMUP):
        run_fn(tokens, gating, w1, w2, w1_scale, w2_scale, mesh).block_until_ready()

    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        run_fn(tokens, gating, w1, w2, w1_scale, w2_scale, mesh).block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)

    return np.median(times)


# ---- Main ----

def main():
    print(f"JAX: {jax.__version__}, Devices: {jax.device_count()} x "
          f"{jax.devices()[0].device_kind}")
    print(f"DeepSeek-R1: H={HIDDEN}, I={INTER}, E={EXPERTS}, topk={TOPK}, EP={EP}")
    print(f"Subchannel qbs={QBS}, Warmup={WARMUP}, Iters={ITERS}")
    print()

    devices = np.array(jax.devices()[:EP]).reshape(1, EP)
    mesh = Mesh(devices, ("data", "model"))
    key = jax.random.PRNGKey(42)

    # ---- Set up weights for both kernels ----
    print("Setting up GMM EP weights...", flush=True)
    gmm_w1, gmm_w2, gmm_w1s, gmm_w2s, key = make_gmm_weights(mesh, key)

    print("Setting up Fused EP weights...", flush=True)
    fused_w1, fused_w2, fused_w1s, fused_w2s, key = make_fused_weights(mesh, key)
    print()

    # Token sharding:
    #   GMM EP   → replicated (each device has all tokens, uses psum)
    #   Fused EP → sharded on EP axis (each device has N/EP tokens, uses DMA a2a)
    gmm_tok_shard = NamedSharding(mesh, P(None, None))
    fused_tok_shard = NamedSharding(mesh, P("model", None))

    scenarios = [
        ("balanced",  lambda rng, n: make_balanced_gating(rng, n)),
        ("natural",   lambda rng, n: make_natural_gating(rng, n)),
        ("skew_1.2x", lambda rng, n: make_fractional_skew_gating(rng, n, 1.2)),
        ("skew_1.5x", lambda rng, n: make_fractional_skew_gating(rng, n, 1.5)),
        ("skew_2x",   lambda rng, n: make_skewed_gating(rng, n, hot_picks=2)),
        ("skew_4x",   lambda rng, n: make_skewed_gating(rng, n, hot_picks=4)),
        ("skew_8x",   lambda rng, n: make_skewed_gating(rng, n, hot_picks=8)),
    ]

    kernels = [
        ("gmm_ep",  run_gmm_ep,  gmm_w1, gmm_w2, gmm_w1s, gmm_w2s, gmm_tok_shard),
        ("fused_ep", run_fused_ep, fused_w1, fused_w2, fused_w1s, fused_w2s, fused_tok_shard),
    ]

    # results[kernel_name][scenario_name][ntok] = median_ms
    all_results = {}
    rng = np.random.default_rng(123)

    for kernel_name, run_fn, w1, w2, w1s, w2s, tok_shard in kernels:
        print(f"\n{'#'*80}")
        print(f"  KERNEL: {kernel_name}")
        print(f"{'#'*80}")

        kernel_results = {}
        for scenario_name, gating_fn in scenarios:
            print(f"\n{'='*70}")
            print(f"  {kernel_name} / {scenario_name}")
            print(f"{'='*70}")

            # Reset RNG per scenario so both kernels see identical gating
            scenario_rng = np.random.default_rng(hash((scenario_name, 42)) & 0xFFFFFFFF)

            results = {}
            for ntok in TOKEN_COUNTS:
                m = ntok * TOPK
                if m % 16 != 0:
                    continue

                k1, key = jax.random.split(key)

                # Tokens: generate full set, then shard appropriately
                all_tokens = jax.random.normal(k1, (ntok, HIDDEN), dtype=jnp.bfloat16) * 0.1
                tokens = jax.device_put(all_tokens, tok_shard)

                # Gating: always replicated (both kernels need full gating for routing)
                gating_full = gating_fn(scenario_rng, ntok)

                if kernel_name == "fused_ep":
                    # Fused kernel expects gating sharded on EP axis
                    gating = jax.device_put(gating_full, NamedSharding(mesh, P("model", None)))
                else:
                    # GMM kernel expects gating replicated
                    gating = jax.device_put(gating_full, NamedSharding(mesh, P(None, None)))

                # Measure device loads (always from full gating)
                device_counts = measure_device_loads(gating_full, EP, EXPERTS_PER_DEVICE, TOPK)
                max_load = device_counts.max()
                avg_load = device_counts.mean()
                ratio = max_load / avg_load if avg_load > 0 else 0

                med = benchmark_kernel(
                    kernel_name, run_fn, tokens, gating, w1, w2, w1s, w2s, mesh)
                results[ntok] = med

                print(f"  {scenario_name:>10}  N={ntok:>5}  median={med:6.2f}ms  "
                      f"dev_loads=[{', '.join(f'{int(c):>5}' for c in device_counts)}]  "
                      f"max/avg={ratio:.2f}x",
                      flush=True)

            kernel_results[scenario_name] = results

        all_results[kernel_name] = kernel_results

    # ---- Summary tables ----
    for kernel_name in [k[0] for k in kernels]:
        kr = all_results[kernel_name]
        print(f"\n\n{'='*80}")
        print(f"SUMMARY — {kernel_name}  (median ms, EP=8, sub_256)")
        print(f"{'='*80}")
        header = f"{'Scenario':>12}" + "".join(f"  {n:>7}" for n in TOKEN_COUNTS)
        print(header)
        print("-" * len(header))
        for name in [s[0] for s in scenarios]:
            res = kr.get(name, {})
            vals = "".join(f"  {res.get(n, float('nan')):7.2f}" for n in TOKEN_COUNTS)
            print(f"{name:>12}{vals}")

        # Slowdown vs balanced
        bal = kr.get("balanced", {})
        if bal:
            print(f"\n{'Slowdown vs balanced':>12}" + "".join(f"  {n:>7}" for n in TOKEN_COUNTS))
            print("-" * len(header))
            for name in [s[0] for s in scenarios]:
                if name == "balanced":
                    continue
                res = kr.get(name, {})
                vals = "".join(
                    f"  {res.get(n, 0) / bal.get(n, 1):7.2f}x"
                    for n in TOKEN_COUNTS
                )
                print(f"{name:>12}{vals}")

    # ---- Cross-kernel comparison ----
    if len(all_results) == 2:
        gmm_r = all_results.get("gmm_ep", {})
        fused_r = all_results.get("fused_ep", {})
        print(f"\n\n{'='*80}")
        print(f"CROSS-KERNEL: fused_ep / gmm_ep  (ratio, <1 = fused faster)")
        print(f"{'='*80}")
        header = f"{'Scenario':>12}" + "".join(f"  {n:>7}" for n in TOKEN_COUNTS)
        print(header)
        print("-" * len(header))
        for name in [s[0] for s in scenarios]:
            gmm_res = gmm_r.get(name, {})
            fused_res = fused_r.get(name, {})
            vals = "".join(
                f"  {fused_res.get(n, 0) / gmm_res.get(n, 1):7.2f}x"
                if gmm_res.get(n) else f"  {'N/A':>7}"
                for n in TOKEN_COUNTS
            )
            print(f"{name:>12}{vals}")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
