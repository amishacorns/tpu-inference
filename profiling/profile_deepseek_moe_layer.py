#!/usr/bin/env python3
"""Profile the real DeepSeek-R1 MoE layer with XProf.

Runs the JIT-compiled MoE layer (routed GMM + shared expert Pallas kernel),
captures XProf traces, then parses them with framework_op_stats and op_profile
to get a per-op breakdown (router, permute, GMM, all-to-all, shared expert, etc).
"""

import os, time, sys
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=rematerialization"
os.environ["ENABLE_QUANTIZED_MATMUL_KERNEL"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Monkey-patch: nnx.with_sharding_constraint was removed in newer Flax
if not hasattr(nnx, 'with_sharding_constraint'):
    def _with_sharding_constraint(x, shardings):
        if isinstance(shardings, tuple):
            shardings = P(*shardings)
        return jax.lax.with_sharding_constraint(x, shardings)
    nnx.with_sharding_constraint = _with_sharding_constraint

from tpu_inference.layers.jax.quantization.fp8 import Fp8Config
from tpu_inference.layers.jax.quantization.unquantized import UnquantizedConfig
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.models.jax.deepseek_v3 import (
    DeepSeekV3Router, DeepseekV3MLP, DeepseekV3MoE)

# ── DeepSeek-R1 config ──
HIDDEN = 7168
INTER_MOE = 2048
INTER_SHARED = 18432
EXPERTS = 256
TOPK = 8
N_GROUPS = 8
TOPK_GROUPS = 4
ROUTED_SCALING = 2.5
HIDDEN_ACT = "silu"
EP = 8

DTYPE = jnp.bfloat16
WARMUP = 10
PROFILE_ITERS = 10

# Token counts to profile
PROFILE_TOKENS = [256, 1024, 4096, 8192]
PROFILE_DIR = "/mnt/pd/xprof/moe_layer"

# Routed experts: channelwise fp8
FP8_QUANT_CFG = Fp8Config({
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [1, 7168],
})

# Shared expert: fp8 channelwise with Pallas blockwise kernel
SHARED_QUANT_CFG = Fp8Config({
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [1, 7168],
})


def bypass_moe_weight_counter(layer: JaxMoE):
    E = layer.num_local_experts
    for attr_name in list(vars(layer).keys()):
        attr = getattr(layer, attr_name, None)
        if isinstance(attr, nnx.Param):
            try:
                attr.set_metadata('_cnt_moe_weights_loaded', E)
            except Exception:
                pass


def analyze_profile(logdir, ntok):
    """Parse XProf trace using the Python API (no server needed)."""
    import json, glob
    from xprof.convert import _pywrap_profiler_plugin as xp

    xplane_files = glob.glob(f'{logdir}/plugins/profile/*/*.xplane.pb')
    if not xplane_files:
        print(f"  WARNING: no xplane.pb found in {logdir}")
        return
    xplane = xplane_files[0]

    m = ntok * TOPK
    print(f"\n{'='*90}")
    print(f"  N={ntok} tokens  (M={m} after top-{TOPK})  Profile: {logdir}")
    print(f"{'='*90}")

    # ── op_profile: get total active vs IDLE time ──
    try:
        result = xp.xspace_to_tools_data([xplane], 'op_profile')
        op_data = json.loads(result[0].decode('utf-8'))
        total_ps = op_data["byProgram"]["metrics"]["rawTime"]
        total_ms = total_ps / 1e9

        children = op_data["byProgram"].get("children", [])
        idle_ms = 0
        active_ms = total_ms
        for child in children:
            if child.get("name") == "IDLE":
                idle_ms = child["metrics"]["rawTime"] / 1e9
                active_ms = total_ms - idle_ms
                break

        print(f"\n  op_profile summary (per iteration, {PROFILE_ITERS} iters averaged):")
        iter_total = total_ms / PROFILE_ITERS
        iter_active = active_ms / PROFILE_ITERS
        iter_idle = idle_ms / PROFILE_ITERS
        print(f"    Total device time:  {iter_total:.3f} ms")
        print(f"    Active:             {iter_active:.3f} ms ({100*iter_active/iter_total:.1f}%)")
        print(f"    IDLE:               {iter_idle:.3f} ms ({100*iter_idle/iter_total:.1f}%)")
    except Exception as e:
        print(f"  op_profile failed: {e}")
        iter_active = None

    # ── framework_op_stats: per-op breakdown ──
    try:
        result = xp.xspace_to_tools_data([xplane], 'framework_op_stats')
        data = json.loads(result[0].decode('utf-8'))

        # data[1] is the exclude-IDLE table
        table = data[1] if len(data) > 1 else data[0]
        rows = table.get("rows", [])

        if not rows:
            print(f"  framework_op_stats: no rows found")
            return

        # Parse rows: [0]=rank, [2]=type, [3]=op_name, [5]=total_time_us,
        #   [7]=total_self_time_us, [13]=flop_rate_GFLOPs, [15]=memory_bw_GBps,
        #   [17]=bound_by
        ops = []
        for row in rows:
            c = row["c"]
            op_name = c[3]["v"] if c[3] else "?"
            op_type = c[2]["v"] if c[2] else "?"
            total_self_us = float(c[7]["v"]) if c[7] and c[7]["v"] is not None else 0
            total_time_us = float(c[5]["v"]) if c[5] and c[5]["v"] is not None else 0
            flop_rate = float(c[13]["v"]) if c[13] and c[13]["v"] is not None else 0
            mem_bw = float(c[15]["v"]) if c[15] and c[15]["v"] is not None else 0
            bound = c[17]["v"] if c[17] and c[17]["v"] is not None else "?"
            ops.append({
                'name': op_name, 'type': op_type,
                'self_us': total_self_us / PROFILE_ITERS,
                'total_us': total_time_us / PROFILE_ITERS,
                'flop_rate_gflops': flop_rate,
                'mem_bw_gbps': mem_bw,
                'bound': bound,
            })

        # Sort by self_time descending
        ops.sort(key=lambda x: -x['self_us'])
        total_self_us = sum(o['self_us'] for o in ops)

        print(f"\n  framework_op_stats per-op breakdown (per iteration):")
        print(f"  Total self time: {total_self_us:.0f} us = {total_self_us/1000:.3f} ms")
        print()
        print(f"  {'#':>3} {'self_us':>9} {'%':>6} {'cum%':>6} {'GF/s':>8} {'GB/s':>8} {'bound':>8}  {'op_name'}")
        print(f"  {'-'*120}")

        cum_pct = 0
        # Categorize ops for summary
        categories = {
            'GMM (routed matmul)': 0,
            'Shared expert matmul': 0,
            'All-to-all / collective': 0,
            'Permute / gather-scatter': 0,
            'Router (sigmoid/topk)': 0,
            'Elementwise (act/mul/add)': 0,
            'Other': 0,
        }

        for i, op in enumerate(ops):
            if op['self_us'] < 0.5:
                continue
            pct = 100 * op['self_us'] / total_self_us if total_self_us > 0 else 0
            cum_pct += pct
            name = op['name']

            # Categorize
            name_lower = name.lower()
            if 'gmm' in name_lower or ('dot' in name_lower and 'expert' in name_lower):
                cat = 'GMM (routed matmul)'
            elif 'quantized_matmul' in name_lower or 'blockwise' in name_lower or 'pallas' in name_lower:
                cat = 'Shared expert matmul'
            elif 'all-to-all' in name_lower or 'collective' in name_lower or 'all_to_all' in name_lower:
                cat = 'All-to-all / collective'
            elif 'gather' in name_lower or 'scatter' in name_lower or 'permut' in name_lower or 'sort' in name_lower or 'dynamic-slice' in name_lower or 'dynamic-update-slice' in name_lower:
                cat = 'Permute / gather-scatter'
            elif 'sigmoid' in name_lower or 'topk' in name_lower or 'top-k' in name_lower or 'router' in name_lower:
                cat = 'Router (sigmoid/topk)'
            elif any(x in name_lower for x in ['multiply', 'add', 'subtract', 'silu', 'swish', 'tanh', 'convert', 'broadcast', 'reduce', 'reshape', 'transpose', 'copy', 'bitcast']):
                cat = 'Elementwise (act/mul/add)'
            else:
                cat = 'Other'
            categories[cat] += op['self_us']

            # Truncate long names
            display_name = name[:90] + '...' if len(name) > 90 else name
            print(f"  {i+1:>3} {op['self_us']:>9.1f} {pct:>5.1f}% {cum_pct:>5.1f}% "
                  f"{op['flop_rate_gflops']:>8.0f} {op['mem_bw_gbps']:>8.0f} {op['bound']:>8}  {display_name}")

            if cum_pct >= 99.0 and i > 20:
                remaining = len([o for o in ops[i+1:] if o['self_us'] >= 0.5])
                if remaining > 0:
                    print(f"  ... {remaining} more ops below 1% cumulative ...")
                break

        # Category summary
        print(f"\n  {'Category Summary':}")
        print(f"  {'-'*70}")
        for cat, us in sorted(categories.items(), key=lambda x: -x[1]):
            if us < 0.5:
                continue
            pct = 100 * us / total_self_us if total_self_us > 0 else 0
            bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
            print(f"  {cat:<35s} {us:>8.0f} us  {us/1000:>7.3f} ms  {pct:>5.1f}%  {bar}")

        print(f"  {'TOTAL':<35s} {total_self_us:>8.0f} us  {total_self_us/1000:>7.3f} ms  100.0%")

    except Exception as e:
        import traceback
        print(f"  framework_op_stats failed: {e}")
        traceback.print_exc()


def main():
    print(f"JAX: {jax.__version__}, Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    devices = np.array(jax.devices()[:EP]).reshape(1, EP)
    mesh = Mesh(devices, ("data", "model"))

    with jax.set_mesh(mesh):
        rngs = nnx.Rngs(42)

        # ── Router ──
        router = DeepSeekV3Router(
            hidden_size=HIDDEN, num_experts=EXPERTS,
            num_experts_per_tok=TOPK, n_groups=N_GROUPS,
            topk_groups=TOPK_GROUPS, norm_topk_prob=True,
            rngs=rngs, routed_scaling_factor=ROUTED_SCALING,
            dtype=DTYPE, moe_backend=MoEBackend.GMM_EP,
            activation_ffw_td=("data", None),
            ed_sharding=(None, None), e_sharding=(None,),
            quant_config=UnquantizedConfig({}),
        )

        # ── Routed experts ──
        experts = JaxMoE(
            dtype=DTYPE, num_local_experts=EXPERTS,
            apply_expert_weight_before_computation=False,
            expert_axis_name="model", num_expert_parallelism=EP,
            hidden_size=HIDDEN, intermediate_size_moe=INTER_MOE,
            num_experts_per_tok=TOPK, mesh=mesh,
            hidden_act=HIDDEN_ACT, rngs=rngs,
            quant_config=FP8_QUANT_CFG,
            activation_ffw_td=("data", "model"),
            activation_ffw_ted=("data", None, "model"),
            edf_sharding=(None, "model", None),
            efd_sharding=(None, None, "model"),
            moe_backend=MoEBackend.GMM_EP, router=router,
        )
        experts.scoring_func = "sigmoid"

        print("Bypassing weight-loading counter...")
        bypass_moe_weight_counter(experts)
        print("Processing weights (fuse + requant)...")
        experts.quant_method.process_weights_after_loading(experts)

        # ── Shared expert ──
        shared_experts = DeepseekV3MLP(
            dtype=DTYPE, hidden_act=HIDDEN_ACT,
            hidden_size=HIDDEN, intermediate_size=INTER_SHARED,
            rngs=rngs, activation_ffw_td=("data", None),
            df_sharding=(None, "model"), fd_sharding=("model", None),
            quant_config=SHARED_QUANT_CFG,
        )
        print("Processing shared expert weights...")
        for proj in [shared_experts.gating_proj, shared_experts.up_proj, shared_experts.down_proj]:
            proj.quant_method.process_weights_after_loading(proj)

        # ── Assemble MoE layer ──
        moe_layer = DeepseekV3MoE(
            experts=experts, shared_experts=shared_experts,
            routed_scaling_factor=ROUTED_SCALING,
        )

        # ── Shard onto mesh ──
        print("Sharding onto mesh...")
        replicated = NamedSharding(mesh, P())
        gd, sd = nnx.split(moe_layer)
        sd = jax.tree.map(lambda x: jax.device_put(x, replicated), sd)
        nnx.update(moe_layer, sd)

        @nnx.jit
        def step(layer, x):
            return layer(x)

        tok_s = NamedSharding(mesh, P("model", None))
        key = jax.random.PRNGKey(0)

        # ── Profile each token count ──
        for ntok in PROFILE_TOKENS:
            k1, key = jax.random.split(key)
            tokens = jax.device_put(
                jax.random.normal(k1, (ntok, HIDDEN), dtype=DTYPE) / 10,
                tok_s)

            # Warmup (includes compilation)
            print(f"\nN={ntok}: warming up ({WARMUP} iters)...")
            for i in range(WARMUP):
                step(moe_layer, tokens).block_until_ready()
                if i == 0:
                    print(f"  compiled + first warmup done")

            # Quick timing for reference
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                step(moe_layer, tokens).block_until_ready()
                times.append((time.perf_counter() - t0) * 1000)
            med = np.median(times)
            print(f"  wall-clock median: {med:.2f} ms")

            # XProf trace
            logdir = f"{PROFILE_DIR}_n{ntok}"
            print(f"  Tracing {PROFILE_ITERS} iters → {logdir}")
            jax.profiler.start_trace(logdir)
            for _ in range(PROFILE_ITERS):
                step(moe_layer, tokens).block_until_ready()
            jax.profiler.stop_trace()
            print(f"  Trace saved.")

            # Analyze immediately
            analyze_profile(logdir, ntok)

        print("\n\nDONE")


if __name__ == "__main__":
    main()
