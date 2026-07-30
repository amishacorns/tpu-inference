# Cross-Implementation Mixture-Of-Experts Benchmark

Six implementations of the same Mixture-of-Experts layer, measured against each
other one model shape at a time, over the same token counts, in the same
compiler frame, in one asserted environment, on one host. The measured record
is a cell. `MOE_COMPARISON.md` holds the tables, and every number in it is
derived from the cells that ship beside it.

## Checking The Published Numbers

```
python3 tools_rederive.py
```

That reads every store under `cells/`, revalidates each record against its own
content address, rebuilds each table of `MOE_COMPARISON.md` from those records,
and exits nonzero on any difference. It needs a Python interpreter and nothing
else. No device, no install, no network.

## What Ships

`cells/` holds six stores, one JSON Lines record per measurement.

| Store | What It Holds |
|---|---|
| `qwen3_5_397b.cells.jsonl` | Qwen3.5 397B under recorded routing |
| `qwen3_5_397b_tuned.cells.jsonl` | Qwen3.5 397B with the hand-written tile entries |
| `qwen3_30b.cells.jsonl` | Qwen3 30B under drawn routing |
| `gpt_oss_20b.cells.jsonl` | GPT-OSS 20B at eight-bit expert weights |
| `gpt_oss_20b_fp4.cells.jsonl` | GPT-OSS 20B at four-bit expert weights |
| `gpt_oss_120b_fp4.cells.jsonl` | GPT-OSS 120B at four-bit expert weights |

`manifests/` holds five manifests, one per shape. A manifest names the shape,
the compiler frame every arm compiles under, the routing seed, the default
tier, the store to append to, and one entry per arm saying which arm, which
profile, which routings and which token counts. It also names the row each arm
publishes as, the control readings that bracket the session, the switch that
becomes the Recommended row, and the refusals expected before the run starts.

`routing_capture/qwen3_5_397b/` holds the recorded routing the 397B tables
replay, with a manifest beside it describing what was recorded and what was
not. Each record is one Mixture-of-Experts layer call and says how many
token-expert row assignments each of the 512 experts received on that call.
There is no token text, no token identifier and no per-token routing in it.
Every replayed cell carries this file's sha256, so a reader can prove which
bytes the cell replayed.

`pins.json` holds one entry per source tree the benchmark measures, with the
repository, the commit and a sha256 for every file the arm executed.
`arms/_sources.py` fetches each entry into `arms/_sources/` at its commit and
refuses the run when any file hashes differently. An entry may name a
`patch_file` under `pins_patches/`, which is applied to the checkout before it
is hashed, and its pinned hashes are the patched ones. The entry the
production implementation binds pins a whole tree rather than a file list, so
it carries a commit and no per-file hashes.

`env.lock` is the environment the cells were measured in. It is asserted before
any cell is measured rather than offered as advice. The environment hash a cell
carries is not this file's hash: it is a sha256 of the full pip freeze of the
interpreter that measured the cell, which covers every installed distribution
and not only the ones this file pins.

`config.py` is the one place a shape, a flag value, a compiler frame, a profile
or a measurement constant may exist. `cells.py` is the record and its
validator. `run.py` is the only driver. `harness.py` is the measurement window
and the conversion from a device profile to device time. `gates.py` holds the
gates and their negative controls. `arms/` holds one module per implementation.

## What A Cell Is

One measurement of one implementation at one token count, with everything
needed to know what it means attached to it. A cell missing any field cannot be
written.

A cell carries what was measured, which is the arm, the shape, the token count
and the routing. It carries where the code came from, which is the tree, the
commit and a hash of the kernel source it executed. It carries what
configuration it ran, which is the profile, the configuration origin, every
flag value in force, the compiler frame and the frame's resolved option dict.
It carries what the measurement was, which is the replayed steps or the draw
seed, the iterations, the repeats, the warmup and the tier. It carries which
session it belongs to, which is the session id, the control drift measured at
its own token count, the environment hash and the device. Its result is a
status, the whole-layer program device time and the per-step times, or a
refusal naming who refused.

A cell is addressed by a hash of its fingerprint, never by file name, file
order or modification time, so two stores can be compared record by record.

## Re-Measuring A Manifest On A TPU Host

```
pip install --no-deps -r env.lock
python3 run.py --manifest manifests/qwen3_5_397b.json --deviceless
python3 run.py --manifest manifests/qwen3_5_397b.json
```

The first command installs the exact environment the numbers were taken in. The
second builds every point of every arm without touching a device, which is the
pre-flight that proves the plan builds. The third runs the manifest for real.
It snapshots the kit into a run directory and re-executes from that copy,
resolves the plan, gates the environment and the inputs, spawns one child
process per configuration so that no two source trees share an interpreter,
takes an opening and a closing control reading, and appends validated cells to
the store the manifest names.

`--only <arm>` and `--batches <list>` narrow a run. They cannot widen it, and
the narrowed plan is what every gate sees.

`run.py` exits 0 when the run completed, 2 when a gate refused it, 3 when cells
failed, and 5 on a defect in the instrument itself. A control that did not hold
is a refused gate, so it exits 2.

The device is eight TPU v7 chips on one host. A run refuses any other device
rather than writing a cell that claims one.

## The Reproduction Contract

Fixed are the environment, asserted line by line from `env.lock` before
anything runs, the device, the source of every arm by commit and by file hash,
every flag value, taken from `config.py`, the routing, which is either a replay
of recorded steps named on every cell or a draw from the manifest's seed, and
the token values, which are synthetic and built from a fixed seed.

Measured is whole-layer program device time from a device profile, divided down
to one call, never a host timer.

Varying is device time, within the drift each cell carries. A rerun that lands
inside that drift reproduces. A rerun that does not is a finding rather than a
rounding difference. Cells from different sessions are never averaged. A row
may compose sessions: every value in it is a single reading, and the control
readings that bracket each session agree with each other across sessions to
within a fraction of a percent.

## Adding An Arm

Write `arms/<name>.py` against the arm interface documented in `run.py`,
binding its source tree by path. Add a `pins.json` entry with the repository,
the commit and a sha256 for every file the arm executes, unless the arm binds
this branch's own tree, whose files are hashed and recorded on the cell
instead. Add the arm to a manifest with its profile, its routings, its token
counts and the row it publishes as. An arm that is not in a manifest is not in
the plan and not in the gates.

## Re-Measuring Notes

The env.lock command pins the measurement environment only. The three implementations that bind this repository's serving tree also need the repository's own Python dependencies and a vllm installation matching the tree, which env.lock deliberately does not carry.

The production implementation checks out a pinned commit far behind the branch tip. A clone with full history makes a worktree of it directly; a clone without it fetches the commit from upstream instead, which needs network.

A host that re-measures writes its own local tree paths into new measurement records. The shipped records carry neutral placeholders in that field instead.

The 397B manifest pins the start-up exclusion rule the published draws used. A fresh replay under the manifest's pinned rule reproduces the published steps at every count.
