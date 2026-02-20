#!/usr/bin/env python3
"""Profile fused_moe with XProf: baseline vs quantize_activation."""

import os, time, json, glob
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=rematerialization"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func
from xprof.convert import _pywrap_profiler_plugin as xp

HIDDEN, INTER, EXPERTS, TOPK, EP = 7168, 2048, 256, 8, 8
WEIGHT_DTYPE = jnp.float8_e4m3fn
ACT_DTYPE = jnp.bfloat16
NTOK = 1024  # peak speedup case
WARMUP = 10
PROFILE_ITERS = 10


def profile_config(config_name, q_act, tokens, gating, w1, w2, w1_scale, w2_scale, mesh, logdir):
    """Warmup then capture XProf trace."""
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

    # Profile
    with jax.profiler.trace(logdir):
        for _ in range(PROFILE_ITERS):
            run_once().block_until_ready()

    print(f"  Profile saved to {logdir}")


def analyze_profile(config_name, logdir):
    """Extract per-op breakdown from XProf trace."""
    xplane = glob.glob(f'{logdir}/plugins/profile/*/*.xplane.pb')[0]

    # framework_op_stats — per-op breakdown
    result = xp.xspace_to_tools_data([xplane], 'framework_op_stats')
    data = json.loads(result[0].decode('utf-8'))

    # data[1] = exclude-IDLE table
    table = data[1]
    cols = table["cols"]
    rows = table["rows"]

    print(f"\n{'='*80}")
    print(f"  {config_name} — framework_op_stats (top ops by self_time)")
    print(f"{'='*80}")
    print(f"{'rank':>4}  {'self_time_us':>12}  {'pct':>6}  {'bw_GBps':>9}  {'flops_G':>9}  {'bound':>8}  op_name")
    print("-" * 100)

    # Collect all ops with self_time
    ops = []
    for row in rows:
        c = row["c"]
        rank = c[0]["v"] if c[0] else None
        op_type = c[2]["v"] if c[2] else ""
        op_name = c[3]["v"] if c[3] else ""
        total_time = c[5]["v"] if c[5] else 0
        self_time = c[7]["v"] if c[7] else 0
        flop_rate = c[13]["v"] if c[13] else 0
        mem_bw = c[15]["v"] if c[15] else 0
        bound = c[17]["v"] if c[17] else ""
        ops.append((self_time, rank, op_type, op_name, total_time, flop_rate, mem_bw, bound))

    # Sort by self_time descending
    ops.sort(key=lambda x: x[0], reverse=True)
    total_self = sum(o[0] for o in ops)

    for self_time, rank, op_type, op_name, total_time, flop_rate, mem_bw, bound in ops[:25]:
        pct = self_time / total_self * 100 if total_self > 0 else 0
        print(f"{rank:>4}  {self_time:>12.1f}  {pct:>5.1f}%  {mem_bw:>9.1f}  {flop_rate:>9.1f}  {bound:>8}  {op_name}")

    print(f"\n  Total self_time: {total_self:.1f} us")

    # op_profile — IDLE time
    result2 = xp.xspace_to_tools_data([xplane], 'op_profile')
    data2 = json.loads(result2[0].decode('utf-8'))
    total_raw = data2["byProgram"]["metrics"]["rawTime"]
    total_ms = total_raw / 1e9
    children = data2["byProgram"].get("children", [])
    idle_ms = 0
    active_ms = 0
    for child in children:
        child_ms = child["metrics"]["rawTime"] / 1e9
        if child.get("name") == "IDLE":
            idle_ms = child_ms
        else:
            active_ms = child_ms

    print(f"  op_profile: total={total_ms:.3f}ms, active={active_ms:.3f}ms, IDLE={idle_ms:.3f}ms")

    # Look for quantize-related ops
    print(f"\n  Quantize-related ops:")
    for self_time, rank, op_type, op_name, total_time, flop_rate, mem_bw, bound in ops:
        name_lower = op_name.lower()
        if any(kw in name_lower for kw in ["quantize", "abs", "max", "convert", "cast", "scale", "clamp", "float8", "fp8"]):
            pct = self_time / total_self * 100 if total_self > 0 else 0
            print(f"    {self_time:>10.1f}us ({pct:>5.1f}%)  {bound:>8}  {op_name}")

    return ops, total_self


def main():
    print(f"JAX: {jax.__version__}, Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    devices = np.array(jax.devices()[:EP]).reshape(1, EP)
    mesh = Mesh(devices, ("data", "model"))

    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    experts_per_dev = EXPERTS // EP
    w_s = NamedSharding(mesh, P("model", None, None))

    w1 = jax.device_put(
        (jax.random.normal(k1, (EXPERTS, HIDDEN, INTER * 2), dtype=jnp.bfloat16) / 100).astype(WEIGHT_DTYPE), w_s)
    w2 = jax.device_put(
        (jax.random.normal(k2, (EXPERTS, INTER, HIDDEN), dtype=jnp.bfloat16) / 100).astype(WEIGHT_DTYPE), w_s)

    w1_scale = jax.device_put(
        jnp.ones((EXPERTS, 1, 1, INTER * 2), dtype=jnp.float32),
        NamedSharding(mesh, P("model", None, None, None)))
    w2_scale = jax.device_put(
        jnp.ones((EXPERTS, 1, 1, HIDDEN), dtype=jnp.float32),
        NamedSharding(mesh, P("model", None, None, None)))

    tok_s = NamedSharding(mesh, P("model", None))
    tokens = jax.device_put(
        (jax.random.normal(k3, (NTOK, HIDDEN), dtype=jnp.bfloat16) / 10).astype(ACT_DTYPE), tok_s)
    gating = jax.device_put(
        jax.random.normal(k4, (NTOK, EXPERTS), dtype=jnp.bfloat16), tok_s)

    base_dir = "/mnt/pd/xprof"

    configs = [
        ("baseline (bf16×fp8)", False, f"{base_dir}/moe_baseline"),
        ("quantize_activation (fp8×fp8)", True, f"{base_dir}/moe_qact"),
    ]

    for config_name, q_act, logdir in configs:
        print(f"\nProfiling: {config_name} ...")
        profile_config(config_name, q_act, tokens, gating, w1, w2, w1_scale, w2_scale, mesh, logdir)

    print("\n\n" + "=" * 80)
    print("  ANALYSIS")
    print("=" * 80)

    for config_name, q_act, logdir in configs:
        analyze_profile(config_name, logdir)

    print("\nDONE")


if __name__ == "__main__":
    main()
