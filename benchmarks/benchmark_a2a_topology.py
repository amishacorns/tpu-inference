#!/usr/bin/env python3
"""Benchmark all-to-all collective with restricted topology on TPU.

Tests whether restricting all-to-all to a subset of devices (e.g., 4 of 8)
is faster than full 8-device all-to-all. This is the key question for
grouped expert routing: if we pick top-4 expert groups (= 4 devices),
can we save communication time?

TPU v7x-8 topology (empirically determined):
  4 physical chips in a 2x2 ICI mesh, 2 cores per chip = 8 devices.
  Chip coords:
    (0,0) → dev 0,1    (1,0) → dev 2,3
    (0,1) → dev 4,5    (1,1) → dev 6,7

  ICI links: 2 per chip (x and y neighbors). Max hops = 2 (diagonal).
    (0,0)---(1,0)
      |       |
    (0,1)---(1,1)

  Intra-chip (core 0 ↔ core 1): essentially free (shared HBM).
  1-hop ICI: chip neighbors (x or y direction).
  2-hop ICI: diagonal chips (e.g., (0,0)↔(1,1)).

Tests with topology-aware sub-groups:
  1. all_to_all / psum — full 8 devices (all 4 chips)
  2. all_to_all / psum — 4 devices on same chip-pair (1-hop, e.g., [0,1,2,3])
  3. all_to_all / psum — 2 devices on same chip (0-hop, e.g., [0,1])
  4. Topology-aware groupings: same-row vs same-col vs diagonal
  5. ppermute (nearest-neighbor baseline)
  6. ragged_all_to_all (variable-size MoE dispatch)

Data sizes match MoE token dispatch payloads:
  N_tokens × hidden_dim(7168) × dtype_size
"""

import time
import functools
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HIDDEN_DIM = 7168          # DeepSeek hidden size
WARMUP = 10
ITERS = 50
TOKEN_COUNTS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
DTYPES = [jnp.bfloat16, jnp.float8_e4m3fn]

# Topology-aware sub-groups for v7x-8 (4 chips × 2 cores)
# Same-chip pairs (0 ICI hops between cores):
SAME_CHIP_GROUPS = [[0, 1], [2, 3], [4, 5], [6, 7]]
# Same-row chip pairs (1 ICI hop: x-neighbors):
SAME_ROW_GROUPS = [[0, 1, 2, 3], [4, 5, 6, 7]]  # row y=0, row y=1
# Same-col chip pairs (1 ICI hop: y-neighbors):
SAME_COL_GROUPS = [[0, 1, 4, 5], [2, 3, 6, 7]]  # col x=0, col x=1
# Cross-diagonal (2 ICI hops for the cross-chip pair):
CROSS_DIAG_GROUPS = [[0, 1, 6, 7], [2, 3, 4, 5]]  # (0,0)+(1,1), (1,0)+(0,1)


def make_axis_index_groups(n_devices, group_size):
    """Create axis_index_groups for sub-group collectives.
    E.g., n_devices=8, group_size=4 → [[0,1,2,3], [4,5,6,7]]
    """
    assert n_devices % group_size == 0
    groups = []
    for start in range(0, n_devices, group_size):
        groups.append(list(range(start, start + group_size)))
    return groups


def benchmark_all_to_all(mesh, n_tokens, dtype, group_size=None,
                        axis_index_groups_override=None):
    """Benchmark jax.lax.all_to_all with optional sub-grouping.

    Each device starts with (n_tokens, HIDDEN_DIM) and splits it into
    `num_peers` chunks, sending one chunk to each peer device.
    After all_to_all, each device has (n_tokens, HIDDEN_DIM) — the
    chunks it received from all peers, concatenated along axis 0.
    """
    n_devices = len(jax.devices())
    if axis_index_groups_override is not None:
        axis_groups = axis_index_groups_override
        peers = len(axis_groups[0])
    elif group_size:
        peers = group_size
        axis_groups = make_axis_index_groups(n_devices, peers)
    else:
        peers = n_devices
        axis_groups = None

    # Shape: each device has (n_tokens, HIDDEN_DIM), split along axis 0
    # into `peers` chunks for the all-to-all exchange.
    # Total data per device = n_tokens * HIDDEN_DIM * dtype_size
    # We need n_tokens divisible by peers.
    padded_tokens = ((n_tokens + peers - 1) // peers) * peers

    # Create sharded input: (n_devices, padded_tokens, HIDDEN_DIM) sharded on axis 0
    total_shape = (n_devices, padded_tokens, HIDDEN_DIM)
    sharding = NamedSharding(mesh, P("devices", None, None))

    key = jax.random.PRNGKey(42)
    data = jax.device_put(
        jax.random.normal(key, total_shape, dtype=jnp.bfloat16).astype(dtype),
        sharding,
    )

    @jax.jit
    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P("devices", None, None),),
        out_specs=P("devices", None, None),
        check_vma=False,
    )
    def do_all_to_all(x):
        # x shape per device: (1, padded_tokens, HIDDEN_DIM)
        # Reshape to (peers, padded_tokens // peers, HIDDEN_DIM) for exchange
        x = x.reshape(peers, padded_tokens // peers, HIDDEN_DIM)
        y = jax.lax.all_to_all(
            x,
            axis_name="devices",
            split_axis=0,
            concat_axis=0,
            axis_index_groups=axis_groups,
        )
        return y.reshape(1, padded_tokens, HIDDEN_DIM)

    # Warmup
    for _ in range(WARMUP):
        out = do_all_to_all(data)
        out.block_until_ready()

    # Timed
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = do_all_to_all(data)
        out.block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return np.median(times), np.min(times), np.max(times)


def benchmark_psum(mesh, n_tokens, dtype, group_size=None,
                   axis_index_groups_override=None):
    """Benchmark jax.lax.psum (all-reduce) with optional sub-grouping.

    This is how the current GMM EP path works: each device computes
    partial results for all tokens, then psum reduces across the EP axis.
    """
    n_devices = len(jax.devices())
    if axis_index_groups_override is not None:
        axis_groups = axis_index_groups_override
    elif group_size:
        axis_groups = make_axis_index_groups(n_devices, group_size)
    else:
        axis_groups = None

    total_shape = (n_devices, n_tokens, HIDDEN_DIM)
    sharding = NamedSharding(mesh, P("devices", None, None))

    key = jax.random.PRNGKey(42)
    data = jax.device_put(
        jax.random.normal(key, total_shape, dtype=jnp.bfloat16).astype(dtype),
        sharding,
    )

    @jax.jit
    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P("devices", None, None),),
        out_specs=P("devices", None, None),
        check_vma=False,
    )
    def do_psum(x):
        return jax.lax.psum(x, axis_name="devices", axis_index_groups=axis_groups)

    # Warmup
    for _ in range(WARMUP):
        out = do_psum(data)
        out.block_until_ready()

    # Timed
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = do_psum(data)
        out.block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return np.median(times), np.min(times), np.max(times)


def benchmark_ppermute_ring(mesh, n_tokens, dtype):
    """Benchmark ppermute in a ring pattern (nearest-neighbor baseline).

    Each device sends to its right neighbor — this is the cheapest
    possible collective (single hop).
    """
    n_devices = len(jax.devices())

    total_shape = (n_devices, n_tokens, HIDDEN_DIM)
    sharding = NamedSharding(mesh, P("devices", None, None))

    key = jax.random.PRNGKey(42)
    data = jax.device_put(
        jax.random.normal(key, total_shape, dtype=jnp.bfloat16).astype(dtype),
        sharding,
    )

    perm = [(i, (i + 1) % n_devices) for i in range(n_devices)]

    @jax.jit
    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P("devices", None, None),),
        out_specs=P("devices", None, None),
        check_vma=False,
    )
    def do_ppermute(x):
        return jax.lax.ppermute(x, axis_name="devices", perm=perm)

    # Warmup
    for _ in range(WARMUP):
        out = do_ppermute(data)
        out.block_until_ready()

    # Timed
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = do_ppermute(data)
        out.block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return np.median(times), np.min(times), np.max(times)


def benchmark_ragged_all_to_all(mesh, n_tokens, dtype, group_size=None):
    """Benchmark jax.lax.ragged_all_to_all — variable-size exchange.

    This simulates the actual MoE dispatch pattern where each device
    sends a different number of tokens to each expert device based on
    the routing decisions. We simulate a roughly uniform distribution.
    """
    n_devices = len(jax.devices())
    peers = group_size if group_size else n_devices

    # For ragged_all_to_all we need: input tensor, output tensor shape,
    # input_offsets, send_sizes, output_offsets, recv_sizes per device pair.
    # Simulate roughly uniform dispatch: each device sends ~n_tokens/peers tokens
    # to each peer.
    tokens_per_peer = n_tokens // peers
    remainder = n_tokens - tokens_per_peer * peers

    # Build per-device send/recv sizes and offsets
    # Each device sends tokens_per_peer to each peer (+ remainder to first peer)
    send_sizes_per_device = np.full((peers,), tokens_per_peer, dtype=np.int32)
    send_sizes_per_device[0] += remainder
    input_offsets = np.cumsum([0] + list(send_sizes_per_device[:-1])).astype(np.int32)

    # Since all devices send the same pattern, recv matches send
    recv_sizes_per_device = send_sizes_per_device.copy()
    output_offsets = input_offsets.copy()

    # Total recv = sum of what all peers send to this device
    total_recv = int(recv_sizes_per_device.sum())

    total_shape = (n_devices, n_tokens, HIDDEN_DIM)
    out_shape = (n_devices, total_recv, HIDDEN_DIM)
    sharding = NamedSharding(mesh, P("devices", None, None))
    rep_sharding = NamedSharding(mesh, P("devices", None))

    key = jax.random.PRNGKey(42)
    data = jax.device_put(
        jax.random.normal(key, total_shape, dtype=jnp.bfloat16).astype(dtype),
        sharding,
    )

    # Replicate metadata per device
    send_s = jax.device_put(jnp.broadcast_to(jnp.array(send_sizes_per_device), (n_devices, peers)), rep_sharding)
    recv_s = jax.device_put(jnp.broadcast_to(jnp.array(recv_sizes_per_device), (n_devices, peers)), rep_sharding)
    in_off = jax.device_put(jnp.broadcast_to(jnp.array(input_offsets), (n_devices, peers)), rep_sharding)
    out_off = jax.device_put(jnp.broadcast_to(jnp.array(output_offsets), (n_devices, peers)), rep_sharding)

    # Estimate max output tokens
    output_est = jax.device_put(
        jax.random.normal(key, out_shape, dtype=jnp.bfloat16).astype(dtype),
        sharding,
    )

    axis_groups = make_axis_index_groups(n_devices, peers) if group_size else None

    @jax.jit
    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(
            P("devices", None, None),
            P("devices", None, None),
            P("devices", None),
            P("devices", None),
            P("devices", None),
            P("devices", None),
        ),
        out_specs=P("devices", None, None),
        check_vma=False,
    )
    def do_ragged_a2a(x, out_est, in_off, send_s, out_off, recv_s):
        return jax.lax.ragged_all_to_all(
            x, out_est,
            in_off[0], send_s[0], out_off[0], recv_s[0],
            axis_name="devices",
            axis_index_groups=axis_groups,
        )

    # Warmup
    for _ in range(WARMUP):
        out = do_ragged_a2a(data, output_est, in_off, send_s, out_off, recv_s)
        out.block_until_ready()

    # Timed
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = do_ragged_a2a(data, output_est, in_off, send_s, out_off, recv_s)
        out.block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return np.median(times), np.min(times), np.max(times)


def main():
    devices = jax.devices()
    n_devices = len(devices)
    print(f"JAX devices: {n_devices} x {devices[0].device_kind}")
    print(f"Platform: {jax.default_backend()}")

    # Print topology
    chips = {}
    for d in devices:
        key = tuple(d.coords)
        if key not in chips:
            chips[key] = []
        chips[key].append(d.id)
    print(f"\nPhysical topology: {len(chips)} chips, {n_devices // len(chips)} cores/chip")
    for coords, dev_ids in sorted(chips.items()):
        print(f"  chip {coords}: devices {dev_ids}")
    print(f"\nICI mesh (2x2):")
    print(f"  (0,0)[{chips.get((0,0,0),['?'])}] --- (1,0)[{chips.get((1,0,0),['?'])}]")
    print(f"    |                    |")
    print(f"  (0,1)[{chips.get((0,1,0),['?'])}] --- (1,1)[{chips.get((1,1,0),['?'])}]")
    print()

    mesh = Mesh(np.array(devices).reshape(n_devices), axis_names=("devices",))

    # ====================================================================
    # Test 1: all_to_all with topology-aware sub-groups
    # ====================================================================
    for dtype in DTYPES:
        dtype_name = "bf16" if dtype == jnp.bfloat16 else "fp8"
        elem_bytes = 2 if dtype == jnp.bfloat16 else 1

        print(f"\n{'='*90}")
        print(f"  all_to_all — topology-aware sub-groups  |  dtype={dtype_name}")
        print(f"{'='*90}")

        configs = [
            ("full-8",    None),
            ("row-4",     SAME_ROW_GROUPS),    # 1-hop neighbors (x-dim)
            ("col-4",     SAME_COL_GROUPS),    # 1-hop neighbors (y-dim)
            ("diag-4",    CROSS_DIAG_GROUPS),  # 2-hop (diagonal chips)
            ("chip-2",    SAME_CHIP_GROUPS),   # 0-hop (same chip)
        ]

        header = f"{'ntok':>6} {'MB/dev':>7}"
        for label, _ in configs:
            header += f" {label:>10}"
        header += "  row/full  chip/full"
        print(header)
        print("-" * len(header))

        for ntok in TOKEN_COUNTS:
            mb = ntok * HIDDEN_DIM * elem_bytes / 1e6
            row = f"{ntok:>6} {mb:>7.1f}"
            meds = {}
            for label, axis_groups in configs:
                try:
                    gs = len(axis_groups[0]) if axis_groups else None
                    med, mn, mx = benchmark_all_to_all(mesh, ntok, dtype, gs,
                                                        axis_index_groups_override=axis_groups)
                    meds[label] = med
                    row += f" {med:>10.3f}"
                except Exception as e:
                    row += f" {'ERR':>10}"
            # Ratios
            f8 = meds.get("full-8", 1)
            row += f"  {meds.get('row-4', 0)/f8:>8.2f}x" if f8 else "      N/A"
            row += f"  {meds.get('chip-2', 0)/f8:>8.2f}x" if f8 else "      N/A"
            print(row, flush=True)

    # ====================================================================
    # Test 2: psum with topology-aware sub-groups
    # ====================================================================
    for dtype in DTYPES:
        dtype_name = "bf16" if dtype == jnp.bfloat16 else "fp8"
        elem_bytes = 2 if dtype == jnp.bfloat16 else 1

        print(f"\n{'='*90}")
        print(f"  psum (all-reduce) — topology-aware sub-groups  |  dtype={dtype_name}")
        print(f"{'='*90}")

        configs = [
            ("full-8",    None),
            ("row-4",     SAME_ROW_GROUPS),
            ("col-4",     SAME_COL_GROUPS),
            ("diag-4",    CROSS_DIAG_GROUPS),
            ("chip-2",    SAME_CHIP_GROUPS),
        ]

        header = f"{'ntok':>6} {'MB/dev':>7}"
        for label, _ in configs:
            header += f" {label:>10}"
        header += "  row/full  chip/full"
        print(header)
        print("-" * len(header))

        for ntok in TOKEN_COUNTS:
            mb = ntok * HIDDEN_DIM * elem_bytes / 1e6
            row = f"{ntok:>6} {mb:>7.1f}"
            meds = {}
            for label, axis_groups in configs:
                try:
                    gs = len(axis_groups[0]) if axis_groups else None
                    med, mn, mx = benchmark_psum(mesh, ntok, dtype, gs,
                                                  axis_index_groups_override=axis_groups)
                    meds[label] = med
                    row += f" {med:>10.3f}"
                except Exception as e:
                    row += f" {'ERR':>10}"
            f8 = meds.get("full-8", 1)
            row += f"  {meds.get('row-4', 0)/f8:>8.2f}x" if f8 else "      N/A"
            row += f"  {meds.get('chip-2', 0)/f8:>8.2f}x" if f8 else "      N/A"
            print(row, flush=True)

    # ====================================================================
    # Test 3: ppermute ring (nearest-neighbor baseline)
    # ====================================================================
    for dtype in DTYPES:
        dtype_name = "bf16" if dtype == jnp.bfloat16 else "fp8"
        elem_bytes = 2 if dtype == jnp.bfloat16 else 1

        print(f"\n{'='*90}")
        print(f"  ppermute (ring, nearest-neighbor)  |  dtype={dtype_name}")
        print(f"{'='*90}")
        print(f"{'ntok':>6} {'MB/dev':>7} {'med_ms':>10} {'min_ms':>10}")
        print("-" * 40)

        for ntok in TOKEN_COUNTS:
            mb = ntok * HIDDEN_DIM * elem_bytes / 1e6
            try:
                med, mn, mx = benchmark_ppermute_ring(mesh, ntok, dtype)
                print(f"{ntok:>6} {mb:>7.1f} {med:>10.3f} {mn:>10.3f}", flush=True)
            except Exception as e:
                print(f"{ntok:>6} {mb:>7.1f} {'ERR':>10} {str(e)[:20]:>10}", flush=True)

    # ====================================================================
    # Test 4: ragged_all_to_all (MoE-realistic variable-size dispatch)
    # ====================================================================
    for dtype in [jnp.bfloat16]:
        dtype_name = "bf16"
        elem_bytes = 2

        print(f"\n{'='*90}")
        print(f"  ragged_all_to_all  |  dtype={dtype_name}")
        print(f"{'='*90}")

        configs_ragged = [
            ("full-8", None),
            ("grp-4",  4),
            ("grp-2",  2),
        ]
        header = f"{'ntok':>6} {'MB/dev':>7}"
        for label, _ in configs_ragged:
            header += f" {label:>10}"
        print(header)
        print("-" * len(header))

        for ntok in TOKEN_COUNTS:
            mb = ntok * HIDDEN_DIM * elem_bytes / 1e6
            row = f"{ntok:>6} {mb:>7.1f}"
            for label, gs in configs_ragged:
                try:
                    med, mn, mx = benchmark_ragged_all_to_all(mesh, ntok, dtype, gs)
                    row += f" {med:>10.3f}"
                except Exception as e:
                    row += f" {'ERR':>10}"
            print(row, flush=True)

    # ====================================================================
    # Test 5: MoE round-trip (scatter A2A + gather A2A)
    # ====================================================================
    print(f"\n{'='*90}")
    print(f"  MoE round-trip: scatter + gather all_to_all  |  bf16")
    print(f"  (2× all_to_all to simulate forward+backward token dispatch)")
    print(f"{'='*90}")

    configs_rt = [
        ("full-8", None),
        ("row-4",  SAME_ROW_GROUPS),
        ("chip-2", SAME_CHIP_GROUPS),
    ]
    header = f"{'ntok':>6} {'MB/dev':>7}"
    for label, _ in configs_rt:
        header += f" {label+'_RT':>12}"
    header += "  row/full"
    print(header)
    print("-" * len(header))

    for ntok in TOKEN_COUNTS:
        mb = ntok * HIDDEN_DIM * 2 / 1e6
        row = f"{ntok:>6} {mb:>7.1f}"
        meds = {}
        for label, axis_groups in configs_rt:
            try:
                gs = len(axis_groups[0]) if axis_groups else None
                # Scatter
                ms, _, _ = benchmark_all_to_all(mesh, ntok, jnp.bfloat16, gs,
                                                 axis_index_groups_override=axis_groups)
                # Gather
                mg, _, _ = benchmark_all_to_all(mesh, ntok, jnp.bfloat16, gs,
                                                 axis_index_groups_override=axis_groups)
                total = ms + mg
                meds[label] = total
                row += f" {total:>12.3f}"
            except Exception as e:
                row += f" {'ERR':>12}"
        f8 = meds.get("full-8", 1)
        if f8 and "row-4" in meds:
            row += f"  {meds['row-4']/f8:>8.2f}x"
        print(row, flush=True)

    # ====================================================================
    # Summary
    # ====================================================================
    print(f"\n{'='*90}")
    print("  Analysis")
    print(f"{'='*90}")
    print("""
TPU v7x-8 topology: 4 chips × 2 cores, 2×2 ICI mesh.
  - Intra-chip (2 cores): ~free (shared HBM)
  - 1-hop ICI (x or y neighbor): 1 link traversal
  - 2-hop ICI (diagonal): 2 link traversals

Sub-group options for grouped expert routing:
  full-8: All 8 devices (4 chips) — current EP approach
  row-4:  4 devices on 2 chips in same row (1-hop ICI max)
  col-4:  4 devices on 2 chips in same col (1-hop ICI max)
  diag-4: 4 devices on 2 diagonal chips (2-hop ICI max)
  chip-2: 2 devices on same chip (0 ICI hops)

Key questions answered:
  1. row-4 vs diag-4: Does physical locality (1-hop vs 2-hop) matter?
  2. row-4 vs full-8: How much does 4-device sub-group save?
  3. chip-2 vs full-8: Maximum possible savings with intra-chip only?
  4. bf16 vs fp8: Does halving payload help A2A proportionally?

DeepSeek MoE context:
  - 256 experts, 8 groups of 32 → top-4 groups → top-8 experts
  - Group routing restricts to 4 of 8 devices  
  - With 2 cores/chip, aligning expert groups to chips means
    a sub-group of 4 uses 2 chips = row-4 topology
  - 2× A2A per MoE layer (scatter tokens, gather outputs)
""")
    print("DONE")


if __name__ == "__main__":
    main()
