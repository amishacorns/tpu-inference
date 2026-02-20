"""Profile raw GMM kernel in isolation — no EP, no gating, no masking.

Measures just the Pallas GMM matmul (gmm_v2) with realistic DeepSeek-R1 shapes
and group distributions.  Supports subchannel scaling (fp4 weights + per-block
scales) and per-channel baseline.  Produces roofline analysis + XProf per-op
breakdown.
"""

import os, time, sys
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=rematerialization"

import jax
import jax.numpy as jnp
import numpy as np
from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2

# ── DeepSeek-R1 config ──
HIDDEN = 7168
INTER = 2048
EXPERTS = 256
TOPK = 8
EP = 8
LOCAL_EXPERTS = EXPERTS // EP  # 32

# ── Dtype config ──
ACT_DTYPE = jnp.float8_e4m3fn
ACT_BYTES = 1

WEIGHT_DTYPE = jnp.float4_e2m1fn
WEIGHT_BYTES = 0.5

# ── Subchannel config ──
QUANT_BLOCK_SIZE = 256  # 0 = per-channel, >0 = subchannel block-wise scaling

# ── Profiling config ──
WARMUP = 5
ITERS = 20
PROFILE_ITERS = 5
PROFILE_DIR = "/mnt/pd/xprof/gmm_v2_subchannel"
TOKEN_COUNTS = [512, 1024, 2048, 4096, 8192, 16384]

# ── TPU v7x specs ──
MXU_TFLOPS_FP8 = 1840.0    # fp8 peak (2x bf16)
MXU_TFLOPS_BF16 = 920.0    # bf16 peak (reference)
HBM_BW_GB = 3300.0         # GB/s


def uniform_group_sizes(m, num_groups):
    """Create uniform group sizes (m tokens spread evenly across groups)."""
    base = m // num_groups
    remainder = m % num_groups
    sizes = np.full(num_groups, base, dtype=np.int32)
    sizes[:remainder] += 1
    return jnp.array(sizes, dtype=jnp.int32)


def compute_gmm_flops(m, k, n):
    """Standard matmul FLOPs: 2*M*K*N."""
    return 2 * m * k * n


def compute_gmm_bytes(m, k, n, num_groups, act_bytes, weight_bytes):
    """HBM bytes for one GMM call (theoretical minimum).
    
    LHS (activations): M * K * act_bytes  (read)
    RHS (weights):     G * K * N * weight_bytes  (read)
    Output:            M * N * act_bytes  (write)
    """
    lhs_bytes = m * k * act_bytes
    rhs_bytes = num_groups * k * n * weight_bytes
    out_bytes = m * n * act_bytes  # output in same dtype as LHS
    return lhs_bytes + rhs_bytes + out_bytes


def profile_one_gmm(name, m, k, n, lhs, rhs, rhs_scale, group_sizes):
    """Profile a single GMM kernel (gmm_v2). Returns dict of results."""

    def run():
        return gmm_v2(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            rhs_scale=rhs_scale,
            preferred_element_type=jnp.bfloat16,
        )

    # Warmup
    for _ in range(WARMUP):
        run().block_until_ready()

    # Timed iterations
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        run().block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)

    med_ms = float(np.median(times))
    flops = compute_gmm_flops(m, k, n)
    hbm_bytes = compute_gmm_bytes(m, k, n, LOCAL_EXPERTS, ACT_BYTES, WEIGHT_BYTES)

    tflops = flops / (med_ms / 1000) / 1e12
    bw_gbs = hbm_bytes / (med_ms / 1000) / 1e9
    mxu_pct_bf16 = 100 * tflops / MXU_TFLOPS_BF16
    mxu_pct_fp8 = 100 * tflops / MXU_TFLOPS_FP8
    hbm_pct = 100 * bw_gbs / HBM_BW_GB
    ai = flops / hbm_bytes
    ridge = MXU_TFLOPS_FP8 * 1e12 / (HBM_BW_GB * 1e9)

    if ai >= ridge:
        roofline = MXU_TFLOPS_FP8
        bound = "compute"
    else:
        roofline = ai * HBM_BW_GB * 1e9 / 1e12
        bound = "memory"

    gap = roofline / tflops if tflops > 0 else float('inf')

    return {
        'name': name, 'm': m, 'k': k, 'n': n,
        'med_ms': med_ms,
        'tflops': tflops, 'mxu_pct_bf16': mxu_pct_bf16,
        'mxu_pct_fp8': mxu_pct_fp8,
        'bw_gbs': bw_gbs, 'hbm_pct': hbm_pct,
        'ai': ai, 'roofline': roofline, 'bound': bound, 'gap': gap,
        'run_fn': run,
    }


def main():
    qbs_str = f"subchannel qbs={QUANT_BLOCK_SIZE}" if QUANT_BLOCK_SIZE else "per-channel"
    print(f"TPU v7x: {MXU_TFLOPS_BF16} TF/s bf16, {MXU_TFLOPS_FP8} TF/s fp8, {HBM_BW_GB} GB/s HBM")
    print(f"DeepSeek-R1: H={HIDDEN} I={INTER} E={EXPERTS} topk={TOPK} EP={EP}")
    print(f"Local experts: {LOCAL_EXPERTS}")
    print(f"Act: {ACT_DTYPE} ({ACT_BYTES}B), Weight: {WEIGHT_DTYPE} ({WEIGHT_BYTES}B)")
    print(f"Scale mode: {qbs_str}")
    print(f"Warmup={WARMUP}, Iters={ITERS}")
    print()

    key = jax.random.key(42)

    # Create weights (one set, reused across token counts)
    k1, k2, key = jax.random.split(key, 3)
    # GMM1: [local_experts, H, 2*I]
    w1 = (jax.random.normal(k1, (LOCAL_EXPERTS, HIDDEN, INTER*2),
                            dtype=jnp.bfloat16) / 100).astype(WEIGHT_DTYPE)
    # GMM2: [local_experts, I, H]
    w2 = (jax.random.normal(k2, (LOCAL_EXPERTS, INTER, HIDDEN),
                            dtype=jnp.bfloat16) / 100).astype(WEIGHT_DTYPE)

    # Create scales
    if QUANT_BLOCK_SIZE > 0:
        w1_num_blocks = HIDDEN // QUANT_BLOCK_SIZE
        w2_num_blocks = INTER // QUANT_BLOCK_SIZE
        w1_scale = jnp.ones((LOCAL_EXPERTS, w1_num_blocks, 1, INTER*2),
                            dtype=jnp.float32)
        w2_scale = jnp.ones((LOCAL_EXPERTS, w2_num_blocks, 1, HIDDEN),
                            dtype=jnp.float32)
    else:
        # Per-channel: [E, 1, 1, N]
        w1_scale = jnp.ones((LOCAL_EXPERTS, 1, 1, INTER*2), dtype=jnp.float32)
        w2_scale = jnp.ones((LOCAL_EXPERTS, 1, 1, HIDDEN), dtype=jnp.float32)

    print(f"w1: {w1.shape} {w1.dtype}, scale: {w1_scale.shape}")
    print(f"w2: {w2.shape} {w2.dtype}, scale: {w2_scale.shape}")
    print()

    # ── Full-M profiling ──
    all_results = []

    hdr = (f"{'tok':>6} {'M':>7} "
           f"{'GMM1 ms':>8} {'GMM1 TF/s':>10} {'GMM1 MXU%':>9} "
           f"{'GMM2 ms':>8} {'GMM2 TF/s':>10} {'GMM2 MXU%':>9} "
           f"{'Total ms':>9}")
    print(hdr)
    print("-" * len(hdr))

    for ntok in TOKEN_COUNTS:
        m = ntok * TOPK
        group_sizes = uniform_group_sizes(m, LOCAL_EXPERTS)

        ikey, key = jax.random.split(key)
        k1i, k2i = jax.random.split(ikey)
        lhs1 = (jax.random.normal(k1i, (m, HIDDEN), dtype=jnp.bfloat16) / 10
                ).astype(ACT_DTYPE)
        lhs2 = (jax.random.normal(k2i, (m, INTER), dtype=jnp.bfloat16) / 10
                ).astype(ACT_DTYPE)

        try:
            r1 = profile_one_gmm(f"GMM1-N{ntok}", m, HIDDEN, INTER*2,
                                 lhs1, w1, w1_scale, group_sizes)
        except Exception as e:
            print(f"{ntok:>6} {m:>7}  GMM1 FAILED: {e}")
            continue

        try:
            r2 = profile_one_gmm(f"GMM2-N{ntok}", m, INTER, HIDDEN,
                                 lhs2, w2, w2_scale, group_sizes)
        except Exception as e:
            print(f"{ntok:>6} {m:>7}  GMM2 FAILED: {e}")
            continue

        total_ms = r1['med_ms'] + r2['med_ms']
        print(f"{ntok:>6} {m:>7} "
              f"{r1['med_ms']:>8.3f} {r1['tflops']:>10.1f} {r1['mxu_pct_fp8']:>8.1f}% "
              f"{r2['med_ms']:>8.3f} {r2['tflops']:>10.1f} {r2['mxu_pct_fp8']:>8.1f}% "
              f"{total_ms:>9.3f}")

        all_results.append((ntok, m, r1, r2))

    # ── XProf traces ──
    profile_tokens = [1024, 4096, 16384]
    print(f"\n{'='*60}")
    print(f"XProf profiling GMM kernels at {profile_tokens}")
    print(f"{'='*60}")

    for ntok in profile_tokens:
        m = ntok * TOPK
        group_sizes = uniform_group_sizes(m, LOCAL_EXPERTS)

        ikey, key = jax.random.split(key)
        k1i, k2i = jax.random.split(ikey)
        lhs1 = (jax.random.normal(k1i, (m, HIDDEN), dtype=jnp.bfloat16) / 10
                ).astype(ACT_DTYPE)
        lhs2 = (jax.random.normal(k2i, (m, INTER), dtype=jnp.bfloat16) / 10
                ).astype(ACT_DTYPE)

        def run_gmm1():
            return gmm_v2(lhs=lhs1, rhs=w1, group_sizes=group_sizes,
                          rhs_scale=w1_scale,
                          preferred_element_type=jnp.bfloat16)

        def run_gmm2():
            return gmm_v2(lhs=lhs2, rhs=w2, group_sizes=group_sizes,
                          rhs_scale=w2_scale,
                          preferred_element_type=jnp.bfloat16)

        # Warmup
        for _ in range(3):
            run_gmm1().block_until_ready()
            run_gmm2().block_until_ready()

        profile_dir = f"{PROFILE_DIR}_n{ntok}"
        print(f"  Tracing N={ntok} (M={m}) → {profile_dir}")
        jax.profiler.start_trace(profile_dir)
        for _ in range(PROFILE_ITERS):
            run_gmm1().block_until_ready()
            run_gmm2().block_until_ready()
        jax.profiler.stop_trace()
        print(f"  Done → {profile_dir}")

    # ── Parse XProf traces ──
    print(f"\n{'='*60}")
    print("XProf per-op breakdown (GMM kernel only)")
    print(f"{'='*60}")

    import json, glob
    from xprof.convert import _pywrap_profiler_plugin as xp

    for ntok in profile_tokens:
        m = ntok * TOPK
        profile_dir = f"{PROFILE_DIR}_n{ntok}"
        xplane_files = glob.glob(f"{profile_dir}/plugins/profile/*/*.xplane.pb")
        if not xplane_files:
            print(f"\n  N={ntok}: no xplane found")
            continue

        xplane_files.sort()
        result = xp.xspace_to_tools_data([xplane_files[-1]], 'framework_op_stats')
        data = json.loads(result[0].decode('utf-8'))

        # data[1] has exclude-IDLE per-op breakdown
        if len(data) < 2:
            print(f"\n  N={ntok}: no op breakdown")
            continue

        cols = data[1].get("cols", [])
        rows = data[1].get("rows", [])
        if not rows:
            print(f"\n  N={ntok}: no rows in op breakdown")
            continue

        total_self_us = sum(r["c"][7]["v"] for r in rows)
        print(f"\n  N={ntok} (M={m})  total_self_time={total_self_us:.0f}us = {total_self_us/1000:.3f}ms")
        print(f"  {'Op':<65s} {'self_us':>8} {'%':>6}  {'GFLOPs':>8}  {'BW GB/s':>8}  {'bound':>6}")
        print(f"  {'-'*110}")

        for r in rows:
            c = r["c"]
            op_type = c[2]["v"]
            op_name = c[3]["v"]
            self_us = c[7]["v"]
            if self_us < 1.0:
                continue
            pct = 100 * self_us / total_self_us if total_self_us > 0 else 0
            flop_rate = c[13]["v"] if c[13]["v"] else 0
            mem_bw = c[15]["v"] if c[15]["v"] else 0
            bound = c[17]["v"] if len(c) > 17 and c[17]["v"] else ""
            label = op_name if len(op_name) < 65 else op_name[:62] + "..."
            print(f"  {label:<65s} {self_us:>8.1f} {pct:>5.1f}%  {flop_rate:>8.1f}  {mem_bw:>8.1f}  {bound:>6}")

    print("\nDONE")


if __name__ == "__main__":
    main()
