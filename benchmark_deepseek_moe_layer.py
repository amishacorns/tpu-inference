#!/usr/bin/env python3
"""Benchmark the real DeepSeek-R1 MoE layer (DeepseekV3MoE).

Uses the ACTUAL DeepseekV3MoE module: JaxMoE (routed) + DeepseekV3MLP (shared).
Routed experts use Fp8Config with channelwise quantization
(weight_block_size=[1, 7168]).  Shared expert also uses Fp8Config with the
Pallas blockwise kernel (ENABLE_QUANTIZED_MATMUL_KERNEL=1).

The counter check in Fp8FusedMoEMethod.process_weights_after_loading is bypassed
by manually setting _cnt_moe_weights_loaded on each param.
"""

import os, time
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

TOKEN_COUNTS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
WARMUP = 10
ITERS = 30
DTYPE = jnp.bfloat16

# Routed experts: channelwise fp8 (matches production checkpoint)
FP8_QUANT_CFG = Fp8Config({
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [1, 7168],
})

# Shared expert: fp8 channelwise with Pallas blockwise kernel.
# ENABLE_QUANTIZED_MATMUL_KERNEL=1 routes to the fast Pallas kernel
# instead of xla_quantized_matmul (which has per-shape compilation).
SHARED_QUANT_CFG = Fp8Config({
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [1, 7168],
})


def bypass_moe_weight_counter(layer: JaxMoE):
    """Set _cnt_moe_weights_loaded on every weight/scale param so
    process_weights_after_loading thinks all experts are loaded."""
    E = layer.num_local_experts
    for attr_name in list(vars(layer).keys()):
        attr = getattr(layer, attr_name, None)
        if isinstance(attr, nnx.Param):
            try:
                attr.set_metadata('_cnt_moe_weights_loaded', E)
            except Exception:
                pass


def main():
    print(f"JAX: {jax.__version__}, Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    devices = np.array(jax.devices()[:EP]).reshape(1, EP)
    mesh = Mesh(devices, ("data", "model"))

    with jax.set_mesh(mesh):
        rngs = nnx.Rngs(42)

        # ── Router (real DeepSeek sigmoid group-topk, bf16) ──
        router = DeepSeekV3Router(
            hidden_size=HIDDEN,
            num_experts=EXPERTS,
            num_experts_per_tok=TOPK,
            n_groups=N_GROUPS,
            topk_groups=TOPK_GROUPS,
            norm_topk_prob=True,
            rngs=rngs,
            routed_scaling_factor=ROUTED_SCALING,
            dtype=DTYPE,
            moe_backend=MoEBackend.GMM_EP,
            activation_ffw_td=("data", None),
            ed_sharding=(None, None),
            e_sharding=(None,),
            quant_config=UnquantizedConfig({}),
        )

        # ── Routed experts (real JaxMoE with Fp8Config) ──
        experts = JaxMoE(
            dtype=DTYPE,
            num_local_experts=EXPERTS,
            apply_expert_weight_before_computation=False,
            expert_axis_name="model",
            num_expert_parallelism=EP,
            hidden_size=HIDDEN,
            intermediate_size_moe=INTER_MOE,
            num_experts_per_tok=TOPK,
            mesh=mesh,
            hidden_act=HIDDEN_ACT,
            rngs=rngs,
            quant_config=FP8_QUANT_CFG,
            activation_ffw_td=("data", "model"),
            activation_ffw_ted=("data", None, "model"),
            edf_sharding=(None, "model", None),
            efd_sharding=(None, None, "model"),
            moe_backend=MoEBackend.GMM_EP,
            router=router,
        )
        experts.scoring_func = "sigmoid"

        # Bypass the weight-loading counter so process_weights_after_loading works
        print("Bypassing weight-loading counter...")
        bypass_moe_weight_counter(experts)

        # Process weights (fuses gate+up, dequant→requant, transposes)
        print("Processing weights (fuse + requant)...")
        experts.quant_method.process_weights_after_loading(experts)

        # ── Shared expert (real DeepseekV3MLP, bf16 unquantized) ──
        # See comment at SHARED_QUANT_CFG for why we use bf16 here.
        shared_experts = DeepseekV3MLP(
            dtype=DTYPE,
            hidden_act=HIDDEN_ACT,
            hidden_size=HIDDEN,
            intermediate_size=INTER_SHARED,
            rngs=rngs,
            activation_ffw_td=("data", None),
            df_sharding=(None, "model"),
            fd_sharding=("model", None),
            quant_config=SHARED_QUANT_CFG,
        )

        # Process shared expert weights (requant + reshape scales for Pallas kernel)
        print("Processing shared expert weights...")
        for proj in [shared_experts.gating_proj, shared_experts.up_proj, shared_experts.down_proj]:
            proj.quant_method.process_weights_after_loading(proj)

        # ── Assemble the real DeepseekV3MoE ──
        moe_layer = DeepseekV3MoE(
            experts=experts,
            shared_experts=shared_experts,
            routed_scaling_factor=ROUTED_SCALING,
        )

        # ── Shard everything onto mesh ──
        print("Sharding onto mesh...")
        replicated = NamedSharding(mesh, P())
        gd, sd = nnx.split(moe_layer)
        sd = jax.tree.map(lambda x: jax.device_put(x, replicated), sd)
        nnx.update(moe_layer, sd)

        # Print weight info
        w13 = experts.kernel_gating_upproj_EDF.value
        w2 = experts.kernel_down_proj_EFD.value
        print(f"Routed: w13={w13.shape} {w13.dtype}, w2={w2.shape} {w2.dtype}")
        sg = shared_experts.gating_proj.weight.value
        print(f"Shared: gate={sg.shape} {sg.dtype} (gate+up: {HIDDEN}×{INTER_SHARED}, down: {INTER_SHARED}×{HIDDEN})")

        tok_s = NamedSharding(mesh, P("model", None))

        print(f"\n{'N':>6} {'M':>7}  {'median_ms':>10}  {'min_ms':>8}  {'max_ms':>8}")
        print("-" * 50)

        key = jax.random.PRNGKey(0)
        for ntok in TOKEN_COUNTS:
            k1, key = jax.random.split(key)
            tokens = jax.device_put(
                jax.random.normal(k1, (ntok, HIDDEN), dtype=DTYPE) / 10,
                tok_s)

            for _ in range(WARMUP):
                moe_layer(tokens).block_until_ready()

            times = []
            for _ in range(ITERS):
                t0 = time.perf_counter()
                moe_layer(tokens).block_until_ready()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            med = np.median(times)
            mn = np.min(times)
            mx = np.max(times)
            print(f"{ntok:>6} {ntok*TOPK:>7}  {med:>10.2f}  {mn:>8.2f}  {mx:>8.2f}")


if __name__ == "__main__":
    main()