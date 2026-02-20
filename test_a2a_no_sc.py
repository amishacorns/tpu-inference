"""One-off: run a single ragged_all_to_all with SC offload disabled, dump HLO."""
import os
os.environ['LIBTPU_INIT_ARGS'] = os.environ.get('LIBTPU_INIT_ARGS', '') + \
    ' --xla_tpu_enable_sparse_core_collective_offload_ragged_all_to_all=false'
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + \
    ' --xla_dump_to=/mnt/pd/xla_dump_a2a_nosc --xla_dump_hlo_as_text'

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

devices = jax.devices()
mesh = Mesh(devices, axis_names=('x',))
NUM_DEVICES = len(devices)
H = 7168
ntok = 64

# Uniform distribution (25% in-group)
partner = {0:1,1:0,2:3,3:2,4:5,5:4,6:7,7:6}
send_matrix = np.zeros((NUM_DEVICES, NUM_DEVICES), dtype=np.int32)
per = ntok // NUM_DEVICES
leftover = ntok - per * NUM_DEVICES
for src in range(NUM_DEVICES):
    for dst in range(NUM_DEVICES):
        send_matrix[src][dst] = per
    for i in range(leftover):
        send_matrix[src][(src + i) % NUM_DEVICES] += 1

input_offsets = np.zeros((NUM_DEVICES, NUM_DEVICES), dtype=np.int32)
send_sizes = np.copy(send_matrix)
output_offsets = np.zeros((NUM_DEVICES, NUM_DEVICES), dtype=np.int32)
recv_sizes = np.zeros((NUM_DEVICES, NUM_DEVICES), dtype=np.int32)

for src in range(NUM_DEVICES):
    off = 0
    for dst in range(NUM_DEVICES):
        input_offsets[src][dst] = off
        off += send_matrix[src][dst]

for dst in range(NUM_DEVICES):
    for src in range(NUM_DEVICES):
        recv_sizes[dst][src] = send_matrix[src][dst]

for dst in range(NUM_DEVICES):
    off = 0
    for src in range(NUM_DEVICES):
        output_offsets[src][dst] = off
        off += send_matrix[src][dst]

operand = jax.device_put(
    jnp.ones((NUM_DEVICES, ntok, H), dtype=jnp.bfloat16),
    NamedSharding(mesh, P('x', None, None)))
output_buf = jax.device_put(
    jnp.zeros((NUM_DEVICES, ntok, H), dtype=jnp.bfloat16),
    NamedSharding(mesh, P('x', None, None)))

inp_off_jax = jax.device_put(jnp.array(input_offsets), NamedSharding(mesh, P('x', None)))
snd_sz_jax = jax.device_put(jnp.array(send_sizes), NamedSharding(mesh, P('x', None)))
out_off_jax = jax.device_put(jnp.array(output_offsets), NamedSharding(mesh, P('x', None)))
rcv_sz_jax = jax.device_put(jnp.array(recv_sizes), NamedSharding(mesh, P('x', None)))

def ragged_a2a(op, out, i_off, s_sz, o_off, r_sz):
    result = jax.lax.ragged_all_to_all(
        op.squeeze(0), out.squeeze(0),
        i_off.squeeze(0), s_sz.squeeze(0),
        o_off.squeeze(0), r_sz.squeeze(0),
        axis_name='x')
    return result[jnp.newaxis]

fn = jax.jit(shard_map(
    ragged_a2a, mesh,
    in_specs=(P('x',None,None), P('x',None,None), P('x',None), P('x',None), P('x',None), P('x',None)),
    out_specs=P('x', None, None),
    check_rep=False,
))

print("Compiling + running ragged_all_to_all with SC disabled...")
out = fn(operand, output_buf, inp_off_jax, snd_sz_jax, out_off_jax, rcv_sz_jax)
out.block_until_ready()
print("Done. HLO dumped to /mnt/pd/xla_dump_a2a_nosc/")
