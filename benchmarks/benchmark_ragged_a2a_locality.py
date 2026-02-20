"""
Benchmark: ragged_all_to_all with varying in-group locality.

Models DeepSeek-style expert group routing on TPU v7x-8:
  - 8 devices = 4 chips × 2 cores
  - 4 expert groups, 1 group per chip (2 experts per group)
  - "In-group" = self + same-chip partner (both cores on same chip)

Token distribution model:
  - `in_group_frac` of tokens stay in-group, split evenly between self
    and same-chip partner (half each)
  - Remaining tokens go uniformly to the 6 cross-chip devices
  - Uniform baseline: in_group_frac = 2/8 = 25% (every device gets 1/8)

Device layout:
  dev0,1 = chip A  (group 0, experts 0-1)
  dev2,3 = chip B  (group 1, experts 2-3)
  dev4,5 = chip C  (group 2, experts 4-5)
  dev6,7 = chip D  (group 3, experts 6-7)
"""

import os
# SC offload: disable by default, set ENABLE_SC=1 to use SparseCore
if os.environ.get('ENABLE_SC', '0') != '1':
    os.environ['LIBTPU_INIT_ARGS'] = os.environ.get('LIBTPU_INIT_ARGS', '') + \
        ' --xla_tpu_enable_sparse_core_collective_offload_ragged_all_to_all=false'

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import time
import gc
import numpy as np

devices = jax.devices()
NUM_DEVICES = len(devices)
mesh = Mesh(devices, axis_names=('x',))

H = 7168  # DeepSeek hidden dim
WARMUP = 5
ITERS = 30
NUM_TRIALS = 15
PROFILE_DIR = '/mnt/pd/xprof_a2a'


def bench(fn, data_args, iters=ITERS):
    for _ in range(WARMUP):
        out = fn(*data_args)
        if isinstance(out, tuple):
            out[-1].block_until_ready()
        else:
            out.block_until_ready()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*data_args)
        if isinstance(out, tuple):
            out[-1].block_until_ready()
        else:
            out.block_until_ready()
    return (time.perf_counter() - start) / iters * 1000


def make_ragged_a2a_params(ntok_per_device, in_group_frac):
    """
    Build ragged_all_to_all parameters for a given in-group locality fraction.

    "In-group" = self + same-chip partner (2 devices on same chip).

    Token distribution per source device:
      - in_group_frac × ntok → in-group, split evenly (half self, half partner)
      - (1 - in_group_frac) × ntok → split uniformly across 6 cross-chip devices

    Uniform baseline: in_group_frac = 2/8 = 0.25 (each of 8 devices gets ntok/8).
    """
    partner = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6}

    send_matrix = np.zeros((NUM_DEVICES, NUM_DEVICES), dtype=np.int32)

    for src in range(NUM_DEVICES):
        in_group_tokens = int(ntok_per_device * in_group_frac)
        cross_tokens = ntok_per_device - in_group_tokens

        # In-group: split evenly between self and partner
        self_tokens = in_group_tokens // 2
        partner_tokens = in_group_tokens - self_tokens  # partner gets remainder

        # Cross-chip: split uniformly among 6 cross-chip devices
        per_cross = cross_tokens // 6
        cross_leftover = cross_tokens - per_cross * 6

        for dst in range(NUM_DEVICES):
            if dst == src:
                send_matrix[src][dst] = self_tokens
            elif dst == partner[src]:
                send_matrix[src][dst] = partner_tokens
            else:
                send_matrix[src][dst] = per_cross

        # Distribute leftover to cross-chip devices, rotating start per src
        # to avoid systematically favoring low-indexed devices
        cross_devs = [d for d in range(NUM_DEVICES) if d != src and d != partner[src]]
        for i in range(cross_leftover):
            send_matrix[src][cross_devs[(src + i) % len(cross_devs)]] += 1
    
    # Build the ragged_all_to_all parameter arrays
    # Shape: [NUM_DEVICES, NUM_DEVICES] — first dim is sharded across devices
    # Each device's slice is [NUM_DEVICES] = one entry per peer
    
    input_offsets = np.zeros((NUM_DEVICES, NUM_DEVICES), dtype=np.int32)
    send_sizes = np.zeros((NUM_DEVICES, NUM_DEVICES), dtype=np.int32)
    output_offsets = np.zeros((NUM_DEVICES, NUM_DEVICES), dtype=np.int32)
    recv_sizes = np.zeros((NUM_DEVICES, NUM_DEVICES), dtype=np.int32)
    
    for src in range(NUM_DEVICES):
        offset = 0
        for dst in range(NUM_DEVICES):
            input_offsets[src][dst] = offset
            send_sizes[src][dst] = send_matrix[src][dst]
            offset += send_matrix[src][dst]
    
    # recv_sizes[dst][src] = send_sizes[src][dst] (transposed view)
    for dst in range(NUM_DEVICES):
        for src in range(NUM_DEVICES):
            recv_sizes[dst][src] = send_matrix[src][dst]
    
    # output_offsets[src][dst] = where src's data lands on dst
    # On each receiver (dst), data from different senders is packed contiguously
    for dst in range(NUM_DEVICES):
        offset = 0
        for src in range(NUM_DEVICES):
            output_offsets[src][dst] = offset
            offset += send_matrix[src][dst]
    
    # Verify the constraint: send_sizes == all_to_all(recv_sizes)
    # This means: send_sizes[i][j] == recv_sizes[j][i] for all i,j
    for i in range(NUM_DEVICES):
        for j in range(NUM_DEVICES):
            assert send_sizes[i][j] == recv_sizes[j][i], \
                f"Constraint violated: send[{i}][{j}]={send_sizes[i][j]} != recv[{j}][{i}]={recv_sizes[j][i]}"
    
    # Total received per device (for output buffer sizing)
    max_recv = max(recv_sizes[dst].sum() for dst in range(NUM_DEVICES))
    
    return input_offsets, send_sizes, output_offsets, recv_sizes, send_matrix, max_recv


def run_single(ntok_per_device, in_group_frac):
    """Run a single ragged_all_to_all benchmark and return latency in ms."""

    (inp_off, snd_sz, out_off, rcv_sz,
     send_matrix, max_recv) = make_ragged_a2a_params(ntok_per_device, in_group_frac)
    
    # Create sharded JAX arrays — use [N*ntok, H] with P('x', None) so each
    # device directly gets [ntok, H] without a leading size-1 shard dimension.
    # This avoids squeeze/unsqueeze ops (broadcast_in_dim) that waste 17% of time.
    operand = jnp.ones((NUM_DEVICES * ntok_per_device, H), dtype=jnp.bfloat16)
    operand = jax.device_put(operand, NamedSharding(mesh, P('x', None)))
    
    # Output buffer: exact size needed per device (max across all devices)
    # Pad to multiple of 8 for alignment
    out_size = ((max_recv + 7) // 8) * 8
    output_buf = jnp.zeros((NUM_DEVICES * out_size, H), dtype=jnp.bfloat16)
    output_buf = jax.device_put(output_buf, NamedSharding(mesh, P('x', None)))
    
    # Shard the offset/size arrays: flatten [8, 8] → [64] with P('x')
    # so each device gets [8] directly without squeeze
    inp_off_jax = jax.device_put(
        jnp.array(inp_off.reshape(-1)), NamedSharding(mesh, P('x')))
    snd_sz_jax = jax.device_put(
        jnp.array(snd_sz.reshape(-1)), NamedSharding(mesh, P('x')))
    out_off_jax = jax.device_put(
        jnp.array(out_off.reshape(-1)), NamedSharding(mesh, P('x')))
    rcv_sz_jax = jax.device_put(
        jnp.array(rcv_sz.reshape(-1)), NamedSharding(mesh, P('x')))
    
    def ragged_a2a(op, out, i_off, s_sz, o_off, r_sz):
        # Each device gets [ntok, H], [out_size, H], and [8] arrays directly
        return jax.lax.ragged_all_to_all(
            op, out, i_off, s_sz, o_off, r_sz,
            axis_name='x')
    
    fn = jax.jit(shard_map(
        ragged_a2a, mesh,
        in_specs=(
            P('x', None),
            P('x', None),
            P('x'),
            P('x'),
            P('x'),
            P('x'),
        ),
        out_specs=P('x', None),
        check_rep=False,
    ))
    
    ms = bench(fn, (operand, output_buf, inp_off_jax, snd_sz_jax, out_off_jax, rcv_sz_jax))
    
    del operand, output_buf, inp_off_jax, snd_sz_jax, out_off_jax, rcv_sz_jax, fn
    jax.clear_caches()
    gc.collect()
    return ms


if __name__ == "__main__":
    import sys

    print("=" * 90)
    print("ragged_all_to_all in-group locality benchmark")
    print(f"TPU v7x-8: 4 chips × 2 cores | {NUM_TRIALS} trials × {ITERS} iters each")
    print()
    print("Device layout (in-group = same chip = self + partner):")
    for d in devices:
        print(f"  dev{d.id}: coords={d.coords}, core={d.core_on_chip}")
    print()
    print("Token model: in_group_frac of tokens stay on-chip (half self, half partner)")
    print("             rest split uniformly across 6 cross-chip devices")
    print("  Uniform = 25% (2/8 devices are in-group)")
    print("=" * 90)

    if len(sys.argv) > 1:
        ntok_list = [int(x) for x in sys.argv[1:]]
    else:
        ntok_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    # Sweep: 0%, 12.5%, 25% (uniform), 37.5%, 50%, 62.5%, 75%, 87.5%, 100%
    fractions = [i / 8 for i in range(9)]
    frac_labels = [f"{f*100:.0f}%" for f in fractions]
    uniform_idx = 2  # 2/8 = 25%

    # Print the send distribution for one example to verify
    print("\nSend distribution (tokens from dev0, ntok=128):")
    for fi, frac in enumerate(fractions):
        _, _, _, _, sm, _ = make_ragged_a2a_params(128, frac)
        row = sm[0]
        in_grp = row[0] + row[1]
        cross = sum(row[2:])
        print(f"  {frac_labels[fi]:>5s} in-grp: self={row[0]:>4d} partner={row[1]:>4d}"
              f"  cross: {row[2]:>3d} {row[3]:>3d} {row[4]:>3d} {row[5]:>3d} {row[6]:>3d} {row[7]:>3d}"
              f"  | in-grp={in_grp:>4d} cross={cross:>4d}")
    print()

    # Collect all results: [frac_idx][ntok_idx] = list of ms over trials
    all_ms = [[[] for _ in ntok_list] for _ in fractions]

    # Warmup all configs first (triggers compilation + HLO dump)
    print("\nWarming up all configs (triggers XLA compilation + HLO dump)...")
    for fi, frac in enumerate(fractions):
        for ni, ntok in enumerate(ntok_list):
            _ = run_single(ntok, frac)
            print(f"  compiled: in_grp={frac_labels[fi]:>5s} tok={ntok}")
    print("Warmup done. Starting profiled run...")

    # Profile run with xprof
    jax.profiler.start_trace(PROFILE_DIR)
    for trial in range(NUM_TRIALS):
        print(f"\nTrial {trial+1}/{NUM_TRIALS}")
        for fi, frac in enumerate(fractions):
            for ni, ntok in enumerate(ntok_list):
                ms = run_single(ntok, frac)
                all_ms[fi][ni].append(ms)
                payload_mb = ntok * H * 2 / (1024 * 1024)
                print(f"  in_grp={frac_labels[fi]:>5s}  tok={ntok:>4d} ({payload_mb:.1f}MB)  {ms:.3f}ms")
    jax.profiler.stop_trace()
    print(f"\nXProf trace saved to: {PROFILE_DIR}")

    # Compute medians
    median_ms = np.zeros((len(fractions), len(ntok_list)))
    for fi in range(len(fractions)):
        for ni in range(len(ntok_list)):
            median_ms[fi][ni] = np.median(all_ms[fi][ni])

    # Compute speedup vs uniform
    uniform_ms = median_ms[uniform_idx, :]
    speedup = uniform_ms[np.newaxis, :] / median_ms

    # Print consolidated table
    print("\n")
    print("=" * 90)
    print(f"Speedup vs uniform (median of {NUM_TRIALS} trials × {ITERS} iters)")
    print("Rows = in-group fraction (self + same-chip partner)")
    print("Cols = tokens per device")
    print("Uniform = 25% (each of 8 devices gets equal share)")
    print("=" * 90)

    col_w = 10
    header = f"{'in_group':>{col_w}}"
    for ntok in ntok_list:
        header += f"  {ntok:>{col_w}}"
    print(header)

    sub = f"{'':>{col_w}}"
    for ntok in ntok_list:
        mb = ntok * H * 2 / (1024 * 1024)
        sub += f"  {f'({mb:.1f}MB)':>{col_w}}"
    print(sub)
    print("-" * len(header))

    for fi in range(len(fractions)):
        row = f"{frac_labels[fi]:>{col_w}}"
        for ni in range(len(ntok_list)):
            val = speedup[fi][ni]
            cell = f"{val:.2f}x"
            if fi == uniform_idx:
                cell = "1.00x"
            row += f"  {cell:>{col_w}}"
        marker = "  <-- uniform" if fi == uniform_idx else ""
        print(row + marker)

    print()

    # Raw latencies
    print("Raw median latencies (ms):")
    header2 = f"{'in_group':>{col_w}}"
    for ntok in ntok_list:
        header2 += f"  {ntok:>{col_w}}"
    print(header2)
    print("-" * len(header2))
    for fi in range(len(fractions)):
        row = f"{frac_labels[fi]:>{col_w}}"
        for ni in range(len(ntok_list)):
            row += f"  {median_ms[fi][ni]:>{col_w}.3f}"
        marker = "  <-- uniform" if fi == uniform_idx else ""
        print(row + marker)
    print()
