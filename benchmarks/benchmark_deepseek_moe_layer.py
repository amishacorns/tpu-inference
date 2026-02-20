#!/usr/bin/env python3
"""Benchmark the real DeepSeek-R1 MoE layer (DeepseekV3MoE).

Uses the ACTUAL DeepseekV3MoE module: JaxMoE (routed) + DeepseekV3MLP (shared).
ALL weights are fp4 subchannel qbs=256:

  Routed experts: fp4 via gmm_v2 (bf16 × fp4, subchannel auto-detected)
  Shared expert:  fp4 via native jnp.matmul(bf16, fp4), TP-sharded via
                  shard_map. No Pallas kernel, no activation quantization.

Routed experts initially use Fp8Config for weight processing (fuse gate+up,
transpose), then weights are replaced with fp4 + subchannel scales.
Shared expert weights are kept as fp4 and passed directly to jnp.matmul
which XLA handles natively on TPU v7x MXU.

The ENTIRE forward (routed + shared) is wrapped in @nnx.jit so XLA traces
everything together, eliminating Python dispatch overhead between the two.
"""

import os, time
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=rematerialization"

import jax
import jax.numpy as jnp
import jax.lax as lax
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

# Subchannel quantization: 0 = per-channel fp8 (original), >0 = subchannel fp4
QUANT_BLOCK_SIZE = 256

# Routed experts: channelwise fp8 (matches production checkpoint)
FP8_QUANT_CFG = Fp8Config({
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
            activation_ffw_td=P("data", None),
            ed_sharding=P(None, None),
            e_sharding=P(None),
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

        # ── Convert routed experts to fp4 subchannel ──
        if QUANT_BLOCK_SIZE > 0:
            print(f"Converting routed experts to fp4 subchannel (qbs={QUANT_BLOCK_SIZE})...")
            key_conv = jax.random.PRNGKey(99)

            w13_shape = experts.kernel_gating_upproj_EDF.value.shape  # [E, K=7168, N=4096]
            w2_shape = experts.kernel_down_proj_EFD.value.shape        # [E, K=2048, N=7168]

            k1c, k2c = jax.random.split(key_conv)
            w13_fp4 = (jax.random.normal(k1c, w13_shape, dtype=jnp.bfloat16) * 0.01
                       ).astype(jnp.float4_e2m1fn)
            w2_fp4 = (jax.random.normal(k2c, w2_shape, dtype=jnp.bfloat16) * 0.01
                      ).astype(jnp.float4_e2m1fn)

            E, K1, N1 = w13_shape
            _, K2, N2 = w2_shape
            w13_scale = jnp.ones((E, K1 // QUANT_BLOCK_SIZE, 1, N1), dtype=jnp.float32)
            w2_scale = jnp.ones((E, K2 // QUANT_BLOCK_SIZE, 1, N2), dtype=jnp.float32)

            experts.kernel_gating_upproj_EDF = nnx.Param(w13_fp4)
            experts.kernel_down_proj_EFD = nnx.Param(w2_fp4)
            scale_name = experts.quant_method.weight_scale_name
            setattr(experts, f"kernel_gating_upproj_EDF_{scale_name}",
                    nnx.Param(w13_scale))
            setattr(experts, f"kernel_down_proj_EFD_{scale_name}",
                    nnx.Param(w2_scale))
            print(f"  w13: {w13_fp4.shape} {w13_fp4.dtype}, scale: {w13_scale.shape}")
            print(f"  w2:  {w2_fp4.shape} {w2_fp4.dtype}, scale: {w2_scale.shape}")

        # ── Shared expert (native fp4 via jnp.matmul) ──
        # TP-sharded: column-parallel gate/up (F split), row-parallel down (psum).
        # Uses native jnp.matmul(bf16, fp4) — XLA handles bf16×fp4 on MXU directly.
        # No Pallas kernel, no activation quantization, no subchannel overhead.
        D_sh, F_sh = HIDDEN, INTER_SHARED
        F_per = F_sh // EP
        print(f"Creating fp4 shared expert (native matmul, TP={EP})...")

        k_sh = jax.random.PRNGKey(77)
        sh_weights = {}
        for proj_name, w_shape in [
            ("gate", (F_sh, D_sh)),
            ("up",   (F_sh, D_sh)),
            ("down", (D_sh, F_sh)),
        ]:
            k_sh, k_sub = jax.random.split(k_sh)
            w_fp4 = (jax.random.normal(k_sub, w_shape, dtype=jnp.bfloat16) * 0.01
                     ).astype(jnp.float4_e2m1fn)
            sh_weights[proj_name] = w_fp4
            print(f"  {proj_name}: {w_fp4.shape} {w_fp4.dtype}")

        # Shard shared expert weights for TP
        col_w_s = NamedSharding(mesh, P("model", None))  # gate/up [F, D]
        row_w_s = NamedSharding(mesh, P(None, "model"))   # down [D, F]

        sh_gate_w = jax.device_put(sh_weights["gate"], col_w_s)
        sh_up_w   = jax.device_put(sh_weights["up"], col_w_s)
        sh_down_w = jax.device_put(sh_weights["down"], row_w_s)

        # Wrap shared expert as an nnx.Module so it traces with nnx.jit
        class SharedExpertMLP(nnx.Module):
            def __init__(self, gate_w, up_w, down_w, mesh):
                self.gate_w = nnx.Param(gate_w)
                self.up_w = nnx.Param(up_w)
                self.down_w = nnx.Param(down_w)
                self.mesh = mesh

            def __call__(self, x_TD):
                def _fwd(x, gw, uw, dw):
                    gate = jnp.matmul(x, gw.T, preferred_element_type=jnp.float32)
                    gate = gate.astype(jnp.bfloat16)
                    up = jnp.matmul(x, uw.T, preferred_element_type=jnp.float32)
                    up = up.astype(jnp.bfloat16)
                    fused = jax.nn.silu(gate) * up
                    down = jnp.matmul(fused, dw.T, preferred_element_type=jnp.float32)
                    down = down.astype(jnp.bfloat16)
                    down = lax.psum(down, axis_name="model")
                    return down
                return jax.shard_map(
                    _fwd, mesh=self.mesh,
                    in_specs=(P("model", None), P("model", None),
                              P("model", None), P(None, "model")),
                    out_specs=P("model", None),
                    check_vma=False,
                )(x_TD, self.gate_w.value, self.up_w.value, self.down_w.value)

        shared_expert = SharedExpertMLP(sh_gate_w, sh_up_w, sh_down_w, mesh)

        # ── Assemble DeepseekV3MoE with shared expert ──
        moe_layer = DeepseekV3MoE(
            experts=experts, shared_experts=shared_expert,
            routed_scaling_factor=ROUTED_SCALING,
        )

        # ── Shard routed experts onto mesh ──
        print("Sharding onto mesh...")
        replicated = NamedSharding(mesh, P())
        gd, sd = nnx.split(moe_layer)
        sd = jax.tree.map(lambda x: jax.device_put(x, replicated), sd)
        nnx.update(moe_layer, sd)

        # Re-place shared expert weights (split undoes device_put sharding)
        shared_expert.gate_w = nnx.Param(sh_gate_w)
        shared_expert.up_w = nnx.Param(sh_up_w)
        shared_expert.down_w = nnx.Param(sh_down_w)

        # Print weight info
        w13 = experts.kernel_gating_upproj_EDF.value
        w2 = experts.kernel_down_proj_EFD.value
        print(f"Routed: w13={w13.shape} {w13.dtype}, w2={w2.shape} {w2.dtype}")
        scale_name = experts.quant_method.weight_scale_name
        w13s = getattr(experts, f"kernel_gating_upproj_EDF_{scale_name}").value
        print(f"  w13_scale: {w13s.shape}, w2_scale: {getattr(experts, f'kernel_down_proj_EFD_{scale_name}').value.shape}")
        print(f"Shared: gate/up=[{F_sh},{D_sh}] fp4 → TP [{F_per},{D_sh}]/dev")
        print(f"  down=[{D_sh},{F_sh}] fp4 → TP [{D_sh},{F_per}]/dev + psum")

        tok_s = NamedSharding(mesh, P("model", None))

        # ── JIT the ENTIRE forward (routed + shared together) ──
        @nnx.jit
        def forward(moe, tokens):
            return moe(tokens)

        print(f"\nAll fp4 subchannel qbs={QUANT_BLOCK_SIZE}")
        print(f"Routed: {EXPERTS} experts, EP={EP}, topk={TOPK} (bf16×fp4 via GMM)")
        print(f"Shared: TP={EP}, gate/up [{F_sh},{D_sh}], down [{D_sh},{F_sh}] (bf16×fp4 native XLA)")
        print(f"Entire forward is @nnx.jit'd — XLA traces routed+shared together")
        print(f"\n{'N':>6} {'M':>7}  {'median_ms':>10}  {'min_ms':>8}  {'max_ms':>8}")
        print("-" * 50)

        key = jax.random.PRNGKey(0)
        for ntok in TOKEN_COUNTS:
            k1, key = jax.random.split(key)
            tokens = jax.device_put(
                jax.random.normal(k1, (ntok, HIDDEN), dtype=DTYPE) / 10,
                tok_s)

            for _ in range(WARMUP):
                forward(moe_layer, tokens).block_until_ready()

            times = []
            for _ in range(ITERS):
                t0 = time.perf_counter()
                forward(moe_layer, tokens).block_until_ready()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            med = np.median(times)
            mn = np.min(times)
            mx = np.max(times)
            print(f"{ntok:>6} {ntok*TOPK:>7}  {med:>10.2f}  {mn:>8.2f}  {mx:>8.2f}")


if __name__ == "__main__":
    main()