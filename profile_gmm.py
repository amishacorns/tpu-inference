"""Profile raw GMM kernel in isolation — no EP, no gating, no masking.

Measures just the Pallas GMM matmul with realistic DeepSeek-R1 shapes
and group distributions. Produces roofline analysis + XProf per-op breakdown.
"""

import os, time, sys
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=rematerialization"

import jax
import jax.numpy as jnp
import numpy as np
from tpu_inference.kernels.megablox.gmm import gmm, make_group_metadata
from tpu_inference.kernels.megablox.tuned_block_sizes import get_tuned_block_sizes
from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import get_device_vmem_limit

# ── DeepSeek-R1 config ──
HIDDEN = 7168
INTER = 2048
EXPERTS = 256
TOPK = 8
EP = 8
LOCAL_EXPERTS = EXPERTS // EP  # 32

# ── Dtype config ──
ACT_DTYPE = jnp.float8_e4m3fn
ACT_DTYPE_STR = jnp.dtype(ACT_DTYPE).name
ACT_BYTES = 1

WEIGHT_DTYPE = jnp.float4_e2m1fn
WEIGHT_DTYPE_STR = jnp.dtype(WEIGHT_DTYPE).name
WEIGHT_BYTES = 0.5

# ── Profiling config ──
WARMUP = 5
ITERS = 20
PROFILE_ITERS = 5
PROFILE_DIR = "/mnt/pd/xprof/gmm_kernel_fp8xfp4"
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


def get_tiling(m, k, n):
    """Look up tuned tiling from LUT."""
    return get_tuned_block_sizes(
        m=m, k=k, n=n,
        num_total_groups=EXPERTS,
        num_current_groups=LOCAL_EXPERTS,
        lhs_dtype=ACT_DTYPE_STR,
        rhs_dtype=WEIGHT_DTYPE_STR,
        quant_block_size=k,
    )


def profile_one_gmm(name, m, k, n, lhs, rhs, group_sizes, tiling,
                    num_nonzero_groups=None):
    """Profile a single GMM kernel. Returns dict of results."""
    if num_nonzero_groups is None:
        num_nonzero_groups = LOCAL_EXPERTS
    group_offset = jnp.array(0, dtype=jnp.int32)
    vmem_limit = get_device_vmem_limit()

    # Pre-compute group metadata
    tm = tiling[0]
    group_meta, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes,
        m=m,
        tm=tm,
        start_group=group_offset,
        num_nonzero_groups=jnp.int32(num_nonzero_groups),
        visit_empty_groups=False,
    )

    def run():
        return gmm(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            preferred_element_type=ACT_DTYPE,
            rhs_scale=None,
            rhs_bias=None,
            tiling=tiling,
            group_offset=group_offset,
            vmem_limit_bytes=vmem_limit,
            group_metadata=group_meta,
            num_active_tiles=num_active_tiles,
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
        'tiling': tiling, 'med_ms': med_ms,
        'tflops': tflops, 'mxu_pct_bf16': mxu_pct_bf16,
        'mxu_pct_fp8': mxu_pct_fp8,
        'bw_gbs': bw_gbs, 'hbm_pct': hbm_pct,
        'ai': ai, 'roofline': roofline, 'bound': bound, 'gap': gap,
        'run_fn': run,
    }


def main():
    print(f"TPU v7x: {MXU_TFLOPS_BF16} TF/s bf16, {MXU_TFLOPS_FP8} TF/s fp8, {HBM_BW_GB} GB/s HBM")
    print(f"DeepSeek-R1: H={HIDDEN} I={INTER} E={EXPERTS} topk={TOPK} EP={EP}")
    print(f"Local experts: {LOCAL_EXPERTS}")
    print(f"Act: {ACT_DTYPE} ({ACT_BYTES}B), Weight: {WEIGHT_DTYPE} ({WEIGHT_BYTES}B)")
    print(f"VMEM limit: {get_device_vmem_limit()/1e6:.0f} MB")
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

    print(f"w1: {w1.shape} {w1.dtype}")
    print(f"w2: {w2.shape} {w2.dtype}")
    print()

    # ── Full-M profiling ──
    all_results = []

    hdr = (f"{'tok':>6} {'M':>7} "
           f"{'GMM1 ms':>8} {'GMM1 TF/s':>10} {'GMM1 MXU%':>9} "
           f"{'GMM2 ms':>8} {'GMM2 TF/s':>10} {'GMM2 MXU%':>9} "
           f"{'Total ms':>9} {'Tiling1':>20} {'Tiling2':>20}")
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

        tiling1 = get_tiling(m, HIDDEN, INTER*2)
        tiling2 = get_tiling(m, INTER, HIDDEN)

        try:
            r1 = profile_one_gmm(f"GMM1-N{ntok}", m, HIDDEN, INTER*2,
                                 lhs1, w1, group_sizes, tiling1)
        except Exception as e:
            print(f"{ntok:>6} {m:>7}  GMM1 FAILED: {e}")
            continue

        try:
            r2 = profile_one_gmm(f"GMM2-N{ntok}", m, INTER, HIDDEN,
                                 lhs2, w2, group_sizes, tiling2)
        except Exception as e:
            print(f"{ntok:>6} {m:>7}  GMM2 FAILED: {e}")
            continue

        total_ms = r1['med_ms'] + r2['med_ms']
        t1_str = f"({tiling1[0]},{tiling1[1]},{tiling1[2]})"
        t2_str = f"({tiling2[0]},{tiling2[1]},{tiling2[2]})"
        print(f"{ntok:>6} {m:>7} "
              f"{r1['med_ms']:>8.3f} {r1['tflops']:>10.1f} {r1['mxu_pct_fp8']:>8.1f}% "
              f"{r2['med_ms']:>8.3f} {r2['tflops']:>10.1f} {r2['mxu_pct_fp8']:>8.1f}% "
              f"{total_ms:>9.3f} {t1_str:>20} {t2_str:>20}")

        all_results.append((ntok, m, r1, r2))

    # ── XProf traces (full M) ──
    profile_tokens = [1024, 4096, 16384]
    print(f"\n{'='*60}")
    print(f"XProf profiling GMM kernels (full M) at {profile_tokens}")
    print(f"{'='*60}")

    for ntok in profile_tokens:
        m = ntok * TOPK
        group_sizes = uniform_group_sizes(m, LOCAL_EXPERTS)
        group_offset = jnp.array(0, dtype=jnp.int32)

        ikey, key = jax.random.split(key)
        k1i, k2i = jax.random.split(ikey)
        lhs1 = (jax.random.normal(k1i, (m, HIDDEN), dtype=jnp.bfloat16) / 10
                ).astype(ACT_DTYPE)
        lhs2 = (jax.random.normal(k2i, (m, INTER), dtype=jnp.bfloat16) / 10
                ).astype(ACT_DTYPE)

        tiling1 = get_tiling(m, HIDDEN, INTER*2)
        tiling2 = get_tiling(m, INTER, HIDDEN)

        tm1 = tiling1[0]
        gm1, nat1 = make_group_metadata(
            group_sizes=group_sizes, m=m, tm=tm1,
            start_group=group_offset,
            num_nonzero_groups=jnp.int32(LOCAL_EXPERTS),
            visit_empty_groups=False)

        tm2 = tiling2[0]
        gm2, nat2 = make_group_metadata(
            group_sizes=group_sizes, m=m, tm=tm2,
            start_group=group_offset,
            num_nonzero_groups=jnp.int32(LOCAL_EXPERTS),
            visit_empty_groups=False)

        vmem_limit = get_device_vmem_limit()

        def run_gmm1():
            return gmm(lhs=lhs1, rhs=w1, group_sizes=group_sizes,
                       preferred_element_type=ACT_DTYPE, tiling=tiling1,
                       group_offset=group_offset, vmem_limit_bytes=vmem_limit,
                       group_metadata=gm1, num_active_tiles=nat1)

        def run_gmm2():
            return gmm(lhs=lhs2, rhs=w2, group_sizes=group_sizes,
                       preferred_element_type=ACT_DTYPE, tiling=tiling2,
                       group_offset=group_offset, vmem_limit_bytes=vmem_limit,
                       group_metadata=gm2, num_active_tiles=nat2)

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

    import gzip, json, glob
    from collections import defaultdict

    for ntok in profile_tokens:
        m = ntok * TOPK
        profile_dir = f"{PROFILE_DIR}_n{ntok}"
        gz_files = glob.glob(f"{profile_dir}/plugins/profile/*/t1v-*.trace.json.gz")
        if not gz_files:
            print(f"\n  N={ntok}: no trace found")
            continue

        gz_files.sort()
        with gzip.open(gz_files[-1], 'rt') as g:
            data = json.load(g)
        events = data['traceEvents']

        ops = [e for e in events if e.get('ph') == 'X' and e.get('tid') == 3]

        by_name = defaultdict(lambda: {'count': 0, 'total_us': 0})
        for e in ops:
            by_name[e.get('name', '?')]['count'] += 1
            by_name[e.get('name', '?')]['total_us'] += e.get('dur', 0)

        if not by_name:
            print(f"\n  N={ntok}: no ops found")
            continue

        divisor = list(by_name.values())[0]['count']
        sorted_ops = sorted(by_name.items(), key=lambda x: -x[1]['total_us'])
        total_us = sum(info['total_us'] for _, info in sorted_ops) / divisor

        print(f"\n  N={ntok} (M={m})  total={total_us:.0f}us = {total_us/1000:.3f}ms")
        print(f"  {'Op':<65s} {'us':>8} {'%':>6}")
        print(f"  {'-'*82}")

        gmm_total = 0
        for name, info in sorted_ops:
            avg_us = info['total_us'] / divisor
            pct = 100 * avg_us / total_us if total_us > 0 else 0
            if avg_us >= 0.5:
                marker = " ★" if 'gmm-' in name else ""
                print(f"  {name:<65s} {avg_us:>8.1f} {pct:>5.1f}%{marker}")
            if 'gmm-' in name:
                gmm_total += avg_us
                parts = name.split('-')
                m_val = int([p for p in parts if p.startswith('m_')][0][2:])
                k_val = int([p for p in parts if p.startswith('k_')][0][2:])
                n_val = int([p for p in parts if p.startswith('n_')][0][2:])
                flops = 2 * m_val * k_val * n_val
                tflops = flops / (avg_us * 1e-6) / 1e12
                print(f"  {'  → efficiency:':65s} {tflops:.0f} TF/s = "
                      f"{100*tflops/MXU_TFLOPS_BF16:.0f}% bf16, "
                      f"{100*tflops/MXU_TFLOPS_FP8:.0f}% fp8")

        non_gmm = total_us - gmm_total
        print(f"  {'-'*82}")
        print(f"  {'GMM compute:':<65s} {gmm_total:>8.1f} {100*gmm_total/total_us:>5.1f}%")
        print(f"  {'Non-GMM overhead:':<65s} {non_gmm:>8.1f} {100*non_gmm/total_us:>5.1f}%")

    print("\nDONE")


if __name__ == "__main__":
    main()
