#!/usr/bin/env python3
"""Benchmark fused_moe_func with dtype combos × {EP, TP} using GMM v2.

Dtype combos:
  1. bf16 × bf16  (no scale)
  2. bf16 × fp8   (channelwise scale)
  3. fp8  × fp8   (channelwise scale, pre-quantized activations)
  4. fp8  × fp4   (channelwise scale, pre-quantized activations)
  5. bf16 × fp4   (channelwise scale)
  6. bf16 × fp4_sub (subchannel scaling, quant_block_size=256)
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

DTYPE_COMBOS = [
    # (name, lhs_dtype, rhs_dtype, use_scale, subchannel)
    ("bf16×fp4_sub", jnp.bfloat16, jnp.dtype("float4_e2m1fn"), True, True),
    ("bf16×bf16", jnp.bfloat16, jnp.bfloat16, False, False),
    ("bf16×fp8",  jnp.bfloat16, jnp.float8_e4m3fn, True, False),
    ("bf16×fp4",  jnp.bfloat16, jnp.dtype("float4_e2m1fn"), True, False),
    ("fp8×fp8",   jnp.float8_e4m3fn, jnp.float8_e4m3fn, True, False),
    ("fp8×fp4",   jnp.float8_e4m3fn, jnp.dtype("float4_e2m1fn"), True, False),
]


def make_weights(key, shape, rhs_dtype):
    """Create weight tensor in the target dtype."""
    k1, k2 = jax.random.split(key)
    w_bf16 = jax.random.normal(k1, shape, dtype=jnp.bfloat16) * 0.01
    return w_bf16.astype(rhs_dtype)


def make_scale(shape, use_scale):
    """Scale tensor: [num_experts, num_blocks, 1, N]."""
    if not use_scale:
        return None
    return jnp.ones(shape, dtype=jnp.float32)


QUANT_BLOCK_SIZE = 256  # DeepSeek-R1 subchannel quant block size


def run_config(mesh, mode, combo_name, lhs_dtype, rhs_dtype, use_scale, subchannel=False):
    """Benchmark one (mode, dtype_combo) configuration."""
    use_ep = (mode == "EP")
    key = jax.random.PRNGKey(42)

    # Weight shapes: rhs = [E, K, N].
    # GMM1: activations[m, H] @ w1[E, H, 2*I] → [m, 2*I]
    # GMM2: activations[m, I] @ w2[E, I, H]   → [m, H]
    w1_shape = (EXPERTS, HIDDEN, INTER * 2)
    w2_shape = (EXPERTS, INTER, HIDDEN)

    if use_ep:
        w1_shard = NamedSharding(mesh, P("model", None, None))
        w2_shard = NamedSharding(mesh, P("model", None, None))
        w1_scale_shard = NamedSharding(mesh, P("model", None, None, None))
        w2_scale_shard = NamedSharding(mesh, P("model", None, None, None))
    else:
        # TP: w1 shards on last dim (N=2*I), w2 shards on K dim (=I)
        w1_shard = NamedSharding(mesh, P(None, None, "model"))
        w2_shard = NamedSharding(mesh, P(None, "model", None))
        w1_scale_shard = NamedSharding(mesh, P(None, None, None, "model"))
        if subchannel:
            # w2_scale blocks dim tracks K, which is sharded in TP.
            w2_scale_shard = NamedSharding(mesh, P(None, "model", None, None))
        else:
            # w2_scale: num_blocks=1 so replicated.
            w2_scale_shard = NamedSharding(mesh, P(None, None, None, None))

    k1, k2, key = jax.random.split(key, 3)
    w1 = jax.device_put(make_weights(k1, w1_shape, rhs_dtype), w1_shard)
    w2 = jax.device_put(make_weights(k2, w2_shape, rhs_dtype), w2_shard)

    w1_scale = None
    w2_scale = None
    if use_scale:
        if subchannel:
            # Subchannel: scale shape is [E, num_blocks, 1, N]
            w1_num_blocks = HIDDEN // QUANT_BLOCK_SIZE   # 7168/256 = 28
            w2_num_blocks = INTER // QUANT_BLOCK_SIZE    # 2048/256 = 8
            w1_scale_shape = (EXPERTS, w1_num_blocks, 1, INTER * 2)
            w2_scale_shape = (EXPERTS, w2_num_blocks, 1, HIDDEN)
        else:
            # Per-channel: scale shape is [E, 1, 1, N]
            w1_scale_shape = (EXPERTS, 1, 1, INTER * 2)
            w2_scale_shape = (EXPERTS, 1, 1, HIDDEN)
        w1_scale = jax.device_put(make_scale(w1_scale_shape, True), w1_scale_shard)
        w2_scale = jax.device_put(make_scale(w2_scale_shape, True), w2_scale_shard)

    tok_shard = NamedSharding(mesh, P("model", None) if use_ep else P("data", None))

    results = {}
    for ntok in TOKEN_COUNTS:
        # Ensure num_tokens * topk is multiple of 16
        m = ntok * TOPK
        if m % 16 != 0:
            continue

        k1, k2, key = jax.random.split(key, 3)
        tokens = jax.device_put(
            (jax.random.normal(k1, (ntok, HIDDEN), dtype=jnp.bfloat16) * 0.1
             ).astype(lhs_dtype), tok_shard)
        gating = jax.device_put(
            jax.random.normal(k2, (ntok, EXPERTS), dtype=jnp.bfloat16), tok_shard)

        def run_once():
            return fused_moe_func(
                hidden_states=tokens, w1=w1, w2=w2,
                w1_scale=w1_scale, w2_scale=w2_scale,
                w1_bias=None, w2_bias=None,
                gating_output=gating, topk=TOPK, renormalize=True,
                mesh=mesh, use_ep=use_ep, activation="silu",
                scoring_fn="sigmoid",
            )

        # Warmup
        for _ in range(WARMUP):
            run_once().block_until_ready()

        # Timed runs
        times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            run_once().block_until_ready()
            times.append((time.perf_counter() - t0) * 1000)

        med = np.median(times)
        results[ntok] = med
        print(f"  {mode:>2} {combo_name:>10}  N={ntok:>5}  m={m:>6}  "
              f"median={med:6.2f}ms  min={min(times):6.2f}  max={max(times):6.2f}")

    return results


def main():
    print(f"JAX: {jax.__version__}, Devices: {jax.device_count()} x "
          f"{jax.devices()[0].device_kind}")
    print(f"DeepSeek-R1: H={HIDDEN}, I={INTER}, E={EXPERTS}, topk={TOPK}")
    print(f"Warmup={WARMUP}, Iters={ITERS}")
    print()

    devices = np.array(jax.devices()[:EP]).reshape(1, EP)
    mesh_ep = Mesh(devices, ("data", "model"))

    devices_tp = np.array(jax.devices()[:EP]).reshape(1, EP)
    mesh_tp = Mesh(devices_tp, ("data", "model"))

    all_results = {}

    for combo_name, lhs_dtype, rhs_dtype, use_scale, subchannel in DTYPE_COMBOS:
        print(f"\n{'='*70}")
        print(f"  {combo_name}")
        print(f"{'='*70}")

        for mode, mesh in [("EP", mesh_ep), ("TP", mesh_tp)]:
            try:
                key = f"{mode}_{combo_name}"
                all_results[key] = run_config(
                    mesh, mode, combo_name, lhs_dtype, rhs_dtype, use_scale,
                    subchannel=subchannel)
            except Exception as e:
                print(f"  {mode} {combo_name}: FAILED — {e}")
                all_results[f"{mode}_{combo_name}"] = None

    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY (median ms)")
    print(f"{'='*70}")
    header = f"{'Config':>18}"
    for ntok in TOKEN_COUNTS:
        header += f"  {ntok:>7}"
    print(header)
    print("-" * len(header))

    for combo_name, _, _, _, _ in DTYPE_COMBOS:
        for mode in ["EP", "TP"]:
            key = f"{mode}_{combo_name}"
            res = all_results.get(key)
            row = f"{mode + ' ' + combo_name:>18}"
            if res is None:
                row += "  FAILED"
            else:
                for ntok in TOKEN_COUNTS:
                    if ntok in res:
                        row += f"  {res[ntok]:7.2f}"
                    else:
                        row += f"  {'—':>7}"
            print(row)

    print()


if __name__ == "__main__":
    main()
