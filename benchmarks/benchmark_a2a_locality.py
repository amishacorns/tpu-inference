"""
Benchmark: Does routing more A2A traffic to same-chip experts speed up the A2A?

Model: In dispatch+collect MoE with EP=8, each device has tokens that need to
reach experts on other devices. We model this as:
  - Tokens for same-chip partner: handled via fast on-package ppermute
  - Tokens for cross-chip devices: handled via 6-device all_to_all (ICI)

We sweep the fraction of tokens that stay on-chip (same-chip locality).
With DeepSeek's group routing, you can bias more experts onto same-chip,
increasing this fraction.

Baseline: uniform 8-way all_to_all (no locality awareness)
Two-phase: ppermute (same-chip) + smaller all_to_all (cross-chip 6 devices)

TPU v7x-8 layout:
  dev0,1 = chip A [0,0,0]
  dev2,3 = chip B [1,0,0]
  dev4,5 = chip C [0,1,0]
  dev6,7 = chip D [1,1,0]
"""

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import time

devices = jax.devices()
mesh = Mesh(devices, axis_names=('x',))

H = 7168  # DeepSeek hidden dim
WARMUP = 5
ITERS = 30

# Chip pairs: same-chip partner for each device
SAME_CHIP_PARTNER = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6}
SAME_CHIP_PERM = [(i, SAME_CHIP_PARTNER[i]) for i in range(8)]


def bench(fn, data_args, iters=ITERS):
    """Benchmark a jitted function."""
    for _ in range(WARMUP):
        out = fn(*data_args)
        if isinstance(out, tuple):
            out[0].block_until_ready()
        else:
            out.block_until_ready()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*data_args)
        if isinstance(out, tuple):
            out[0].block_until_ready()
        else:
            out.block_until_ready()
    return (time.perf_counter() - start) / iters * 1000  # ms


def make_sharded(shape, spec):
    """Create a sharded bf16 tensor."""
    data = jnp.ones(shape, dtype=jnp.bfloat16)
    return jax.device_put(data, NamedSharding(mesh, spec))


def run_benchmark(total_tokens_per_device):
    """
    Model the dispatch A2A for a given number of tokens per device.
    
    Each device has total_tokens_per_device tokens to send out.
    With topk=8 expert selections, each token goes to one expert/device.
    In uniform routing: ~1/8 of tokens go to each device.
    With locality: more tokens go to same-chip partner, fewer cross-chip.
    
    Sweep: what fraction of a device's outgoing tokens go to same-chip partner?
      - Uniform: 1/7 ≈ 14% (partner gets same share as any other device)
      - Max locality: up to ~85% (most tokens go to same-chip partner)
    
    Two-phase approach:
      Phase 1: ppermute same-chip tokens to partner (on-package, fast)
      Phase 2: all_to_all of remaining cross-chip tokens across all 8 devices
               (each device's cross-chip chunk is smaller)
    """
    
    total_bytes_mb = total_tokens_per_device * H * 2 / (1024*1024)
    
    print(f"\n{'='*80}")
    print(f"Tokens/device: {total_tokens_per_device} | "
          f"Hidden: {H} | Payload: {total_bytes_mb:.1f} MB | bf16")
    print(f"{'='*80}\n")

    # ── Baseline: flat 8-way all_to_all ──
    # Shape [8, ntok, H] sharded on axis 0. all_to_all splits on dim 1.
    # Need ntok divisible by 8.
    ntok_padded = ((total_tokens_per_device + 7) // 8) * 8
    baseline_data = make_sharded((8, ntok_padded, H), P('x', None, None))

    def uniform_a2a(x):
        return jax.lax.all_to_all(x, 'x', split_axis=1, concat_axis=1, tiled=True)

    baseline_fn = jax.jit(shard_map(
        uniform_a2a, mesh,
        in_specs=(P('x', None, None),),
        out_specs=P('x', None, None),
        check_rep=False,
    ))
    baseline_ms = bench(baseline_fn, (baseline_data,))

    # ── Sweep locality fraction ──
    # same_frac = fraction of total outgoing tokens that go to same-chip partner
    # The remaining (1-same_frac) is split across 6 cross-chip devices + self-loop
    # For the cross-chip all_to_all, we size it to carry the cross-chip portion.
    
    results = []
    
    # Sweep: 14% (uniform) through 86% (extreme locality)
    fractions = [1/7, 2/7, 3/7, 4/7, 5/7, 6/7]
    
    for same_frac in fractions:
        same_chip_tokens = max(8, int(total_tokens_per_device * same_frac))
        cross_chip_tokens = total_tokens_per_device - same_chip_tokens
        # The cross-chip tokens get distributed via a full 8-way all_to_all
        # (simpler than trying to do 6-device sub-group, and more realistic
        #  since XLA optimizes full-mesh collectives better)
        cross_chip_padded = max(8, ((cross_chip_tokens + 7) // 8) * 8)
        
        # Phase 1: Same-chip ppermute
        same_data = make_sharded((8, same_chip_tokens, H), P('x', None, None))
        
        def same_chip_exchange(x):
            return jax.lax.ppermute(x, 'x', perm=SAME_CHIP_PERM)
        
        phase1_fn = jax.jit(shard_map(
            same_chip_exchange, mesh,
            in_specs=(P('x', None, None),),
            out_specs=P('x', None, None),
            check_rep=False,
        ))
        phase1_ms = bench(phase1_fn, (same_data,))
        
        # Phase 2: Cross-chip all_to_all (smaller data)
        cross_data = make_sharded((8, cross_chip_padded, H), P('x', None, None))
        
        def cross_chip_a2a(x):
            return jax.lax.all_to_all(x, 'x', split_axis=1, concat_axis=1, tiled=True)
        
        phase2_fn = jax.jit(shard_map(
            cross_chip_a2a, mesh,
            in_specs=(P('x', None, None),),
            out_specs=P('x', None, None),
            check_rep=False,
        ))
        phase2_ms = bench(phase2_fn, (cross_data,))
        
        two_phase_ms = phase1_ms + phase2_ms
        
        results.append({
            'same_frac': same_frac,
            'same_tok': same_chip_tokens,
            'cross_tok': cross_chip_padded,
            'same_MB': same_chip_tokens * H * 2 / (1024*1024),
            'cross_MB': cross_chip_padded * H * 2 / (1024*1024),
            'phase1_ms': phase1_ms,
            'phase2_ms': phase2_ms,
            'two_phase_ms': two_phase_ms,
        })

        del same_data, cross_data

    # ── Print results ──
    print(f"Baseline uniform 8-way all_to_all: {baseline_ms:.3f} ms")
    print(f"  ({ntok_padded} tokens × {H} hidden = {ntok_padded * H * 2 / (1024*1024):.1f} MB per device)")
    print()
    
    header = (f"{'local%':>7} {'same_tok':>8} {'cross_tok':>9} "
              f"{'same_MB':>8} {'cross_MB':>9} "
              f"{'phase1':>8} {'phase2':>8} {'total':>8} {'baseline':>9} {'speedup':>8}")
    print(header)
    print("-" * len(header))
    
    for r in results:
        pct = r['same_frac'] * 100
        speedup = baseline_ms / r['two_phase_ms']
        print(f"{pct:>6.0f}% {r['same_tok']:>8d} {r['cross_tok']:>9d} "
              f"{r['same_MB']:>7.1f}M {r['cross_MB']:>8.1f}M "
              f"{r['phase1_ms']:>7.3f}  {r['phase2_ms']:>7.3f}  {r['two_phase_ms']:>7.3f}  "
              f"{baseline_ms:>8.3f}  {speedup:>7.2f}x")
    
    print()
    return baseline_ms, results


if __name__ == "__main__":
    print("TPU v7x-8: 4 chips × 2 cores, 2×2 ICI mesh")
    print(f"Same-chip pairs: {SAME_CHIP_PERM}")
    print()
    print("Model: two-phase A2A (same-chip ppermute + cross-chip all_to_all)")
    print("       vs baseline uniform 8-way all_to_all")
    print()

    for ntok in [64, 256, 1024, 4096]:
        run_benchmark(ntok)
