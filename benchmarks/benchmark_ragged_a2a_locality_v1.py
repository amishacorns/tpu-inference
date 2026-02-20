"""
Benchmark: ragged_all_to_all with varying locality (same-chip vs cross-chip).

Simulates MoE dispatch where we control what fraction of tokens go to
the same-chip partner expert vs cross-chip experts.

EP=8 on TPU v7x-8:
  dev0,1 = chip A  (experts 0-63)
  dev2,3 = chip B  (experts 64-127)
  dev4,5 = chip C  (experts 128-191)
  dev6,7 = chip D  (experts 192-255)

Each device has `ntok` tokens to dispatch. We vary what fraction goes to
the same-chip partner (cheap on-package transfer) vs the 6 cross-chip
devices (expensive ICI transfer).

Uses jax.lax.ragged_all_to_all — the actual primitive used in sparse_moe.py.
"""

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import time
import numpy as np

devices = jax.devices()
NUM_DEVICES = len(devices)
mesh = Mesh(devices, axis_names=('x',))

H = 7168  # DeepSeek hidden dim
WARMUP = 5
ITERS = 30
NUM_TRIALS = 15


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


def make_ragged_a2a_params(ntok_per_device, same_chip_frac):
    """
    Build ragged_all_to_all parameters for a given locality fraction.
    
    Each device sends `ntok_per_device` tokens total to 8 destinations.
    `same_chip_frac` of tokens go to the same-chip partner.
    The rest are split uniformly across the 6 cross-chip devices.
    Self-sends (device to itself) get a small share too.
    
    Returns numpy arrays (will be converted to jax arrays later):
      input_offsets:  [NUM_DEVICES] per-device — offset into operand for each dest
      send_sizes:     [NUM_DEVICES] per-device — how many tokens to send to each dest
      output_offsets: [NUM_DEVICES] per-device — where sent data lands on receiver
      recv_sizes:     [NUM_DEVICES] per-device — how many tokens received from each src
    
    All arrays have shape [NUM_DEVICES] per device (one entry per peer).
    Since we're in shard_map, each device sees its own slice.
    The full arrays are [NUM_DEVICES, NUM_DEVICES] — sharded on first dim.
    """
    # Same-chip partner for each device
    partner = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6}
    
    # For each device, compute how many tokens go to each destination
    # send_matrix[src][dst] = number of tokens
    send_matrix = np.zeros((NUM_DEVICES, NUM_DEVICES), dtype=np.int32)
    
    for src in range(NUM_DEVICES):
        same_chip_dst = partner[src]
        same_chip_tokens = int(ntok_per_device * same_chip_frac)
        remaining = ntok_per_device - same_chip_tokens
        # Split remaining across 6 cross-chip + self
        # Give self a share too (tokens routed to own experts)
        per_other = remaining // 7  # 6 cross-chip + self
        leftover = remaining - per_other * 7
        
        for dst in range(NUM_DEVICES):
            if dst == same_chip_dst:
                send_matrix[src][dst] = same_chip_tokens
            elif dst == src:
                send_matrix[src][dst] = per_other + leftover  # self gets leftovers
            else:
                send_matrix[src][dst] = per_other
    
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


def run_single(ntok_per_device, same_frac):
    """Run a single ragged_all_to_all benchmark and return latency in ms."""
    
    (inp_off, snd_sz, out_off, rcv_sz, 
     send_matrix, max_recv) = make_ragged_a2a_params(ntok_per_device, same_frac)
    
    # Create sharded JAX arrays
    operand = jnp.ones((NUM_DEVICES, ntok_per_device, H), dtype=jnp.bfloat16)
    operand = jax.device_put(operand, NamedSharding(mesh, P('x', None, None)))
    
    # Output buffer: exact size needed per device (max across all devices)
    # Pad to multiple of 8 for alignment
    out_size = ((max_recv + 7) // 8) * 8
    output_buf = jnp.zeros((NUM_DEVICES, out_size, H), dtype=jnp.bfloat16)
    output_buf = jax.device_put(output_buf, NamedSharding(mesh, P('x', None, None)))
    
    # Shard the offset/size arrays: shape [8, 8], sharded on dim 0
    inp_off_jax = jax.device_put(
        jnp.array(inp_off), NamedSharding(mesh, P('x', None)))
    snd_sz_jax = jax.device_put(
        jnp.array(snd_sz), NamedSharding(mesh, P('x', None)))
    out_off_jax = jax.device_put(
        jnp.array(out_off), NamedSharding(mesh, P('x', None)))
    rcv_sz_jax = jax.device_put(
        jnp.array(rcv_sz), NamedSharding(mesh, P('x', None)))
    
    def ragged_a2a(op, out, i_off, s_sz, o_off, r_sz):
        return jax.lax.ragged_all_to_all(
            op, out,
            i_off.squeeze(0), s_sz.squeeze(0),
            o_off.squeeze(0), r_sz.squeeze(0),
            axis_name='x')
    
    fn = jax.jit(shard_map(
        ragged_a2a, mesh,
        in_specs=(
            P('x', None, None),
            P('x', None, None),
            P('x', None),
            P('x', None),
            P('x', None),
            P('x', None),
        ),
        out_specs=P('x', None, None),
        check_rep=False,
    ))
    
    ms = bench(fn, (operand, output_buf, inp_off_jax, snd_sz_jax, out_off_jax, rcv_sz_jax))
    
    del operand, output_buf, inp_off_jax, snd_sz_jax, out_off_jax, rcv_sz_jax
    return ms


if __name__ == "__main__":
    import sys
    
    print("=" * 90)
    print("ragged_all_to_all locality benchmark")
    print(f"TPU v7x-8: 4 chips × 2 cores | {NUM_TRIALS} trials × {ITERS} iters each")
    print()
    print("Device layout:")
    for d in devices:
        print(f"  dev{d.id}: coords={d.coords}, core={d.core_on_chip}")
    print("=" * 90)
    
    if len(sys.argv) > 1:
        ntok_list = [int(x) for x in sys.argv[1:]]
    else:
        ntok_list = [32, 64, 96, 128, 160]
    
    fractions = [0.0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7]
    frac_labels = ["0%", "14%", "29%", "43%", "57%", "71%", "86%"]
    
    # Collect all results: [frac_idx][ntok_idx] = list of ms over trials
    all_ms = [[[] for _ in ntok_list] for _ in fractions]
    
    for trial in range(NUM_TRIALS):
        print(f"\nTrial {trial+1}/{NUM_TRIALS}")
        for fi, same_frac in enumerate(fractions):
            for ni, ntok in enumerate(ntok_list):
                ms = run_single(ntok, same_frac)
                all_ms[fi][ni].append(ms)
                payload_mb = ntok * H * 2 / (1024 * 1024)
                print(f"  local={frac_labels[fi]:>4s}  tok={ntok:>4d} ({payload_mb:.1f}MB)  {ms:.3f}ms")
    
    # Compute medians
    median_ms = np.zeros((len(fractions), len(ntok_list)))
    for fi in range(len(fractions)):
        for ni in range(len(ntok_list)):
            median_ms[fi][ni] = np.median(all_ms[fi][ni])
    
    # Compute speedup vs uniform (frac_idx=1 is 1/7 = uniform)
    uniform_ms = median_ms[1, :]  # row for 14% (uniform)
    speedup = uniform_ms[np.newaxis, :] / median_ms  # >1 means faster
    
    # Print consolidated table
    print("\n")
    print("=" * 90)
    print(f"Speedup vs uniform (median of {NUM_TRIALS} trials × {ITERS} iters)")
    print("Rows = fraction of tokens routed to same-chip partner")
    print("Cols = tokens per device")
    print("=" * 90)
    
    # Header row
    col_w = 10
    header = f"{'local':>{col_w}}"
    for ntok in ntok_list:
        header += f"  {ntok:>{col_w}}"
    print(header)
    
    # Sub-header with MB
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
            if fi == 1:
                cell = "1.00x"  # uniform baseline by definition
            row += f"  {cell:>{col_w}}"
        marker = "  <-- uniform" if fi == 1 else ""
        print(row + marker)
    
    print()
    
    # Also print raw latencies
    print("Raw median latencies (ms):")
    header2 = f"{'local':>{col_w}}"
    for ntok in ntok_list:
        header2 += f"  {ntok:>{col_w}}"
    print(header2)
    print("-" * len(header2))
    for fi in range(len(fractions)):
        row = f"{frac_labels[fi]:>{col_w}}"
        for ni in range(len(ntok_list)):
            row += f"  {median_ms[fi][ni]:>{col_w}.3f}"
        print(row)
    print()
