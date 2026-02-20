#!/usr/bin/env python3
"""Profile shared expert MLP across dtype configs on TPU v7x-8.

Configs:
  bf16×bf16  — pre-dequant fp4 weights to bf16, pure bf16 matmul
  bf16×fp4   — Pallas kernel: bf16 activations × fp4 weights, subchannel post-scale,
               NO activation quantization (same approach as GMM v2 for routed experts)
  fp8×fp4    — production Pallas blockwise kernel: fp8 activation quant × fp4 weights

TP=8. DeepSeek-R1 shared expert:
  gate/up: [N, D=7168] × [F=18432, D]^T  (col-parallel, F/8=2304 per device)
  down:    [N, F=18432] × [D=7168, F]^T  (row-parallel + psum, F/8=2304 per device)
"""

import os, time, functools
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=rematerialization"

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.quantized_matmul.blockwise_kernel import (
    quantized_matmul_kernel as blockwise_quantized_matmul_kernel)
from tpu_inference.kernels.quantized_matmul.util import (
    unfold_args, next_multiple)
from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import (
    TunedKey, TunedValue, TUNED_BLOCK_SIZES,
    get_tuned_block_sizes, get_device_vmem_limit)
from tpu_inference.layers.common.linear import sharded_quantized_matmul

# ── Config ──
HIDDEN = 7168       # D
INTER = 18432       # F
TP = 8
QBS = 256           # subchannel quant block size
MXU_SIZE = 256

TOKEN_COUNTS = [128, 256, 512, 1024, 2048, 4096, 8192]
WARMUP = 10
ITERS = 30
DTYPE = jnp.bfloat16

D = HIDDEN
F = INTER
F_per = F // TP     # 2304 per device


# ── Register tuned tile sizes ──
def _register_tuned():
    shapes = [(F_per, D), (D, F_per)]  # (n_out, n_in) inside shard_map
    safe = TunedValue(128, 256, 256, 1)
    for n_out, n_in in shapes:
        for nb in TOKEN_COUNTS:
            for x_q in ["float8_e4m3fn", "bfloat16"]:
                key = TunedKey(7, nb, n_out, n_in, x_q, "float4_e2m1fn")
                if key not in TUNED_BLOCK_SIZES:
                    TUNED_BLOCK_SIZES[key] = safe

_register_tuned()


# ═══════════════════════════════════════════════════════════════════════
# bf16×fp4 Pallas kernel  —  NO fp8 activation quantization
# ═══════════════════════════════════════════════════════════════════════
# Modeled on blockwise_quantized_matmul_kernel but with the activation
# quantization removed.  dot_general(bf16, fp4, preferred_element_type=f32)
# then multiply by weight scale only (no lhs_scale).

@functools.partial(jax.jit, static_argnames=["block_size", "tuned_value"])
def bf16xfp4_matmul_kernel(
    x: jax.Array,        # [bs, n_in] bf16
    w_q: jax.Array,      # [n_out, n_in] fp4
    w_scale: jax.Array,  # [n_in // block_size, 1, n_out] f32
    *,
    block_size: int,
    tuned_value: TunedValue | None = None,
) -> jax.Array:
    """Dense matmul: bf16 × fp4 with subchannel post-scale, no activation quant."""

    if tuned_value is None:
        tuned_value = get_tuned_block_sizes(
            n_batch=x.shape[0], n_out=w_q.shape[0], n_in=x.shape[1],
            x_q_dtype="bfloat16", w_q_dtype="float4_e2m1fn")

    batch_bs = tuned_value.batch_block_size
    out_bs = tuned_value.out_block_size
    in_bs = tuned_value.in_block_size
    n_lane_mul = tuned_value.n_lane_multiplier

    orig_nb, orig_ni = x.shape
    orig_no = w_q.shape[0]

    block_size = in_bs if block_size == orig_ni else block_size

    # Pad
    pad_nb = next_multiple(orig_nb, batch_bs)
    if orig_nb < pad_nb:
        x = jnp.pad(x, ((0, pad_nb - orig_nb), (0, 0)))
    pad_no = next_multiple(orig_no, out_bs)
    if orig_no < pad_no:
        w_q = jnp.pad(w_q, ((0, pad_no - orig_no), (0, 0)))
        w_scale = jnp.pad(w_scale, ((0, 0), (0, 0), (0, pad_no - orig_no)))
    pad_ni = next_multiple(orig_ni, in_bs)
    if orig_ni < pad_ni:
        x = jnp.pad(x, ((0, 0), (0, pad_ni - orig_ni)))
        w_q = jnp.pad(w_q, ((0, 0), (0, pad_ni - orig_ni)))

    if w_scale.dtype != jnp.float32:
        w_scale = w_scale.astype(jnp.float32)

    n_batch = pad_nb // batch_bs
    n_out = pad_no // out_bs
    n_in = pad_ni // in_bs
    save_acc = n_in > 1

    acc_dtype = jnp.bfloat16
    steps_k = in_bs // block_size
    compute_tile_n = MXU_SIZE * n_lane_mul
    steps_n = out_bs // compute_tile_n

    def kernel(lhs_ref, rhs_ref, w_scales_ref, out_ref, acc_scratch):
        pid_k = pl.program_id(2)
        is_first_step = pid_k == 0
        is_last_step = pid_k == (pad_ni // in_bs - 1)

        def accum(is_first_step, is_last_step):
            accumulators = [None] * steps_n

            for i in range(steps_k):
                k_s, k_e = i * block_size, (i + 1) * block_size
                lhs = lhs_ref[:, k_s:k_e]           # [batch_bs, block_size] bf16
                rhs_full = rhs_ref[:, k_s:k_e]      # [out_bs, block_size] fp4
                rhs_scale_full = w_scales_ref[i, :, :].astype(acc_dtype)

                for j in range(steps_n):
                    n_s = j * compute_tile_n
                    n_e = (j + 1) * compute_tile_n
                    rhs_slice = rhs_full[n_s:n_e, :]
                    rhs_scale_slice = rhs_scale_full[:, n_s:n_e]

                    # bf16 × fp4 → f32 (native MXU, no activation quant)
                    dot_res = jax.lax.dot_general(
                        lhs, rhs_slice,
                        (((1,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    # Weight scale only — no lhs_scale
                    res = dot_res.astype(acc_dtype) * rhs_scale_slice

                    if i == 0:
                        accumulators[j] = res
                    else:
                        accumulators[j] += res

            acc_block = jnp.concatenate(accumulators, axis=1)

            if not is_first_step:
                acc_block += acc_scratch[...]

            if is_last_step:
                out_ref[...] = acc_block.astype(out_ref.dtype)
            else:
                acc_scratch[...] = acc_block

        unfold_args((is_first_step, is_last_step), (), accum)

    kernel_call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((batch_bs, in_bs), lambda b, o, i: (b, i),
                             memory_space=pltpu.VMEM),
                pl.BlockSpec((out_bs, in_bs), lambda b, o, i: (o, i),
                             memory_space=pltpu.VMEM),
                pl.BlockSpec((steps_k, 1, out_bs), lambda _, o, i: (i, 0, o),
                             memory_space=pltpu.VMEM),
            ],
            out_specs=pl.BlockSpec((batch_bs, out_bs), lambda b, o, i: (b, o)),
            scratch_shapes=[
                pltpu.VMEM((batch_bs, out_bs), acc_dtype)
            ],
            grid=(n_batch, n_out, n_in),
        ),
        out_shape=jax.ShapeDtypeStruct((pad_nb, pad_no), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=get_device_vmem_limit(),
        ),
    )

    out = kernel_call(x, w_q, w_scale)
    return out[:orig_nb, :orig_no]


# ═══════════════════════════════════════════════════════════════════════
# Sharding helpers  —  TP via shard_map
# ═══════════════════════════════════════════════════════════════════════

def _sharded_bf16_matmul(x, w, weight_spec, mesh):
    """bf16 × bf16 dense matmul with TP sharding."""
    out_axis, in_axis = weight_spec
    x_sharding = P(None, in_axis)
    out_sharding = P(None, out_axis)

    def wrapper(x, w):
        out = jnp.matmul(x, w.T)
        if in_axis:
            out = lax.psum(out, axis_name=in_axis)
        return out

    return jax.shard_map(
        wrapper, mesh=mesh,
        in_specs=(x_sharding, weight_spec),
        out_specs=out_sharding,
        check_vma=False,
    )(x, w)


def _sharded_bf16xfp4_matmul(x, w_q, w_s, weight_spec, mesh):
    """bf16 × fp4 matmul with subchannel scale, no activation quant, TP sharding."""
    out_axis, in_axis = weight_spec
    x_sharding = P(None, in_axis)
    out_sharding = P(None, out_axis)
    num_blocks = w_s.shape[0]
    scale_sharding = P(
        in_axis if num_blocks > 1 else None,
        None, out_axis,
    )

    x = lax.with_sharding_constraint(x, NamedSharding(mesh, x_sharding))

    def wrapper(x, w_q, w_s):
        k_dim = x.shape[1]
        local_blocks = w_s.shape[0]
        bs = k_dim // local_blocks
        out = bf16xfp4_matmul_kernel(x, w_q, w_s, block_size=bs)
        if in_axis:
            out = lax.psum(out, axis_name=in_axis)
        return out

    return jax.shard_map(
        wrapper, mesh=mesh,
        in_specs=(x_sharding, weight_spec, scale_sharding),
        out_specs=out_sharding,
        check_vma=False,
    )(x, w_q, w_s)


# ═══════════════════════════════════════════════════════════════════════

def main():
    print(f"JAX: {jax.__version__}, Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    devices = np.array(jax.devices()[:TP]).reshape(1, TP)
    mesh = Mesh(devices, ("data", "model"))

    key = jax.random.PRNGKey(42)

    # ── Create fp4 weights + scales ──
    k1, k2, k3, key = jax.random.split(key, 4)
    gate_fp4 = (jax.random.normal(k1, (F, D), dtype=DTYPE) * 0.01).astype(jnp.float4_e2m1fn)
    up_fp4   = (jax.random.normal(k2, (F, D), dtype=DTYPE) * 0.01).astype(jnp.float4_e2m1fn)
    down_fp4 = (jax.random.normal(k3, (D, F), dtype=DTYPE) * 0.01).astype(jnp.float4_e2m1fn)

    gate_s = jnp.ones((D // QBS, 1, F), dtype=jnp.float32)
    up_s   = jnp.ones((D // QBS, 1, F), dtype=jnp.float32)
    down_s = jnp.ones((F // QBS, 1, D), dtype=jnp.float32)

    # ── Pre-dequant for bf16×bf16 (scale=1 → just cast) ──
    gate_bf16 = gate_fp4.astype(jnp.bfloat16)
    up_bf16   = up_fp4.astype(jnp.bfloat16)
    down_bf16 = down_fp4.astype(jnp.bfloat16)

    # ── Shard for TP ──
    col_w = NamedSharding(mesh, P("model", None))       # gate/up weight [F, D]
    col_s = NamedSharding(mesh, P(None, None, "model"))  # scale [D//QBS, 1, F]
    row_w = NamedSharding(mesh, P(None, "model"))        # down weight [D, F]
    row_s = NamedSharding(mesh, P("model", None, None))  # scale [F//QBS, 1, D]
    tok_rep = NamedSharding(mesh, P(None, None))         # tokens replicated

    gate_bf16_s = jax.device_put(gate_bf16, col_w)
    up_bf16_s   = jax.device_put(up_bf16, col_w)
    down_bf16_s = jax.device_put(down_bf16, row_w)

    gate_fp4_s = jax.device_put(gate_fp4, col_w)
    up_fp4_s   = jax.device_put(up_fp4, col_w)
    down_fp4_s = jax.device_put(down_fp4, row_w)
    gate_sc_s  = jax.device_put(gate_s, col_s)
    up_sc_s    = jax.device_put(up_s, col_s)
    down_sc_s  = jax.device_put(down_s, row_s)

    col_spec = P("model", None)
    row_spec = P(None, "model")

    print(f"Shared expert: D={D}, F={F}, TP={TP}, F/TP={F_per}")
    print(f"  gate/up: [{F},{D}] → [{F_per},{D}]/dev (col-parallel)")
    print(f"  down:    [{D},{F}] → [{D},{F_per}]/dev (row-parallel + psum)")

    # ──── Config 1: bf16×bf16 ────
    @jax.jit
    def fwd_bf16xbf16(tokens, gw, uw, dw):
        gate = _sharded_bf16_matmul(tokens, gw, col_spec, mesh)
        up   = _sharded_bf16_matmul(tokens, uw, col_spec, mesh)
        fused = jax.nn.silu(gate) * up
        down = _sharded_bf16_matmul(fused, dw, row_spec, mesh)
        return down

    # ──── Config 2: bf16×fp4 (Pallas, no activation quant) ────
    @jax.jit
    def fwd_bf16xfp4(tokens, gw, gs, uw, us, dw, ds):
        gate = _sharded_bf16xfp4_matmul(tokens, gw, gs, col_spec, mesh)
        up   = _sharded_bf16xfp4_matmul(tokens, uw, us, col_spec, mesh)
        fused = jax.nn.silu(gate) * up
        down = _sharded_bf16xfp4_matmul(fused, dw, ds, row_spec, mesh)
        return down

    # ──── Config 3: fp8×fp4 (Pallas blockwise, production kernel) ────
    @jax.jit
    def fwd_fp8xfp4(tokens, gw, gs, uw, us, dw, ds):
        gate = sharded_quantized_matmul(tokens, gw, gs, col_spec, mesh=mesh)
        up   = sharded_quantized_matmul(tokens, uw, us, col_spec, mesh=mesh)
        fused = jax.nn.silu(gate) * up
        down = sharded_quantized_matmul(fused, dw, ds, row_spec, mesh=mesh)
        return down

    # ── Benchmark ──
    configs = [
        ("bf16×bf16", lambda tok: fwd_bf16xbf16(tok, gate_bf16_s, up_bf16_s, down_bf16_s)),
        ("bf16×fp4",  lambda tok: fwd_bf16xfp4(tok, gate_fp4_s, gate_sc_s,
                                                up_fp4_s, up_sc_s,
                                                down_fp4_s, down_sc_s)),
        ("fp8×fp4",   lambda tok: fwd_fp8xfp4(tok, gate_fp4_s, gate_sc_s,
                                               up_fp4_s, up_sc_s,
                                               down_fp4_s, down_sc_s)),
    ]

    results = {name: {} for name, _ in configs}

    for cfg_name, fwd_fn in configs:
        print(f"\n{'='*60}")
        print(f"Benchmarking {cfg_name}...")
        print(f"{'='*60}")
        for ntok in TOKEN_COUNTS:
            k1, key = jax.random.split(key)
            tokens = jax.device_put(
                jax.random.normal(k1, (ntok, D), dtype=DTYPE) / 10, tok_rep)

            # Warmup (includes first-time compilation)
            print(f"  N={ntok:>5}: compiling...", end="", flush=True)
            for _ in range(WARMUP):
                fwd_fn(tokens).block_until_ready()
            print(" timing...", end="", flush=True)

            times = []
            for _ in range(ITERS):
                t0 = time.perf_counter()
                fwd_fn(tokens).block_until_ready()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            med = np.median(times)
            results[cfg_name][ntok] = med
            print(f" {med:.2f} ms")

    # ── Summary table ──
    hdr = "".join(f"{n:>8}" for n in TOKEN_COUNTS)
    print(f"\n{'='*80}")
    print(f"SHARED EXPERT MLP — median ms, TP={TP}, QBS={QBS}")
    print(f"  gate/up: [{F_per},{D}]/dev  down: [{D},{F_per}]/dev + psum")
    print(f"{'='*80}")
    print(f"{'Config':>16} {hdr}")
    print(f"{'-'*16} " + "-" * (8 * len(TOKEN_COUNTS)))
    for cfg_name, _ in configs:
        row = "".join(f"{results[cfg_name].get(n, float('nan')):>8.2f}"
                      for n in TOKEN_COUNTS)
        print(f"{cfg_name:>16} {row}")

    # Speedup vs bf16×bf16
    baseline = results["bf16×bf16"]
    print(f"\n{'Speedup vs bf16×bf16':>24}")
    for cfg_name, _ in configs[1:]:
        row = "".join(
            f"{baseline.get(n, 1) / results[cfg_name].get(n, 1):>8.2f}x"
            for n in TOKEN_COUNTS)
        print(f"{cfg_name:>16} {row}")
    print()


if __name__ == "__main__":
    main()
