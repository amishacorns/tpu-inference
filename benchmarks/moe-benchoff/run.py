#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The only driver. One manifest in, one cell store out.

    python3 run.py --manifest manifests/qwen3_5_397b.json
    python3 run.py --manifest manifests/qwen3_5_397b.json --deviceless
    python3 run.py --manifest manifests/qwen3_5_397b.json --only prod_pair
    python3 run.py --prove-controls

Nothing else in this kit starts a measurement, and this file measures
nothing itself: it plans, gates, spawns the processes that measure, and
writes cells.py records.

THE PROCESS MODEL, AND WHY IT IS NOT ONE PROCESS

A manifest spans source trees. One manifest can span three families at
once: the upstream pin, this branch's tree, and the pinned external
checkouts. Two arms that resolve the same top-level module name from two
different trees cannot share a process, and the arms assert that rather
than measuring a mixture.

So the parent orchestrates and never binds an arm, and every configuration
is bound and measured in a CHILD process that binds exactly one tree. The
arms' single-tree purity then holds by construction rather than by
discipline. The parent keeps everything that must stay in one place: the
resolved plan, every gate, the cell schema, the drift backfill and the
store. A child measures and reports; only the parent calls
cells.make_cell.

The parent does not touch the device either. It would hold the chips its
own children need, so the device facts come from a probe child and are
compared against config.DEVICE_REQUIRED in the one place that comparison
lives, gates.environment.

The known cost is the per-process preamble, and it is budgeted: children
are spawned once per configuration, plus one for the opening control and one
for the closing control.

CHILD ORDER, WHICH IS THE SESSION ORDER

    probe                       the device, once, before anything binds
    control-open child          the control configuration, opening reading
    one child per configuration in manifest order, each building its own
                                points and then measuring them
    control-close child         the control configuration, closing reading

A child build-checks its own points before measuring them, in its own
process, and refuses to measure a point that did not build. The build map
for the WHOLE plan is therefore complete only when the last child returns,
which is why the deviceless pass exists: run this file with --deviceless
and it spawns the same children, builds every point of every
configuration, and measures nothing. That pass is the pre-flight that
proves the plan builds before a device session starts.

THE CHILD CONTRACT

The parent writes <run dir>/children/<name>.spec.json and runs

    python3 run.py --child <spec path>

from the SNAPSHOT kit. The child writes the result file the spec names and
exits 0 when it produced a result, 5 when it hit a defect.

    spec  {"mode": "configuration" | "injected",
           "key": "arm@profile", "arm": ..., "profile": ...,
           "entry": {the manifest's arm entry, verbatim},
           "manifest": <path>,
           "build_batches": [64, ...],
           "points": [{"role": ..., "routing": ..., "batch": ...,
                       "tier": ...}],
           "facts": {run_dir, deviceless, session_id, env_hash, device},
           "numerics": {"batch": ..., "routing": ...} or null,
           "injection": "raises" | "asserts",
           "result": <path>, "outputs_dir": <path>}

    result {"key": ..., "defect": null or str,
            "device": {kinds, chips, hosts, fingerprint} or null,
            "bind": {"ok": bool, "refusal": str, "attribution": str},
            "bound": {tree, tree_sha, source_hash, sources, pin,
                      namespace, flags, contract},
            "mirror": {"declared": bool, "ok": bool, "detail": ...},
            "build": {"<batch>": {status, refusal, attribution, detail}},
            "measurements": [{role, routing, batch, tier, iters, repeats,
                              draw_seed, replay_steps, status, program_us,
                              per_step_us, kernel_self_us, refusal,
                              refusal_attribution}],
            "ops": [profiler rows of the first ok measurement],
            "output": <path to a .npy> or null}

Two things cannot cross a process boundary and so are done in the child and
reported: the arm's verify_mirror hook, whose verdict the mirror gate scores
in the parent, and the arm's output for the numerics comparison, which the
child saves as a .npy that the parent loads. Everything else the gates need
is data.

THE MANIFEST

One json file per shape and phase, under manifests/, which is the merged
authority this driver and the render layer share. The fields this driver
reads, all required unless called optional:

    {
      "shape_id":  "qwen3_5_397b",         key of config.SHAPES
      "frame":     "serving",              key of config.FRAMES, the one
                                           compiler option dict every arm
                                           in this manifest compiles under
      "seed":      0,                      the run's one routing seed
      "cells":     "cells/qwen3_5_397b.cells.jsonl",
                                           the store, relative to this kit
      "tier":      "screen",               key of config.TIERS, the default
                                           for arms that name none
      "arms": [
        {"arm":      "production_pair",    module name under arms/
         "profile":  "upstream-default",   key of config.PROFILES
         "routings": ["replay"],           subset of ("replay", "random")
         "batches":  [64, 8192],           subset of config.BATCHES
         "tier":     "screen",             optional, overrides the default
         "tree":     "/path/to/tree",      optional, the source tree the arm
                                           binds; the arm's own default when
                                           absent
         "flags":    {}}                   required only for the ours-tuned
                                           profile, whose values are a choice
                                           this manifest makes
      ],
      "control": {                         optional; the batches default to
                                           every batch any arm measures, so
                                           every cell can be stamped with a
                                           drift measured at its own count
         "arm": "production_pair", "profile": "ours-served",
         "routing": "random", "batches": [64, 8192], "tier": "screen"},
      "numerics": {                        optional
         "reference": "production_pair",   the arm every other arm is
                                           compared against
         "batch": 64, "routing": "replay"},
      "expected_refusals": [               optional pre-registration
        {"arm": "sglang_v2", "batch": 8192, "attribution": "kernel",
         "reason": "scalar memory"}],
      "capture": "routing_capture/qwen3_5_397b/group_sizes.jsonl",
                                           optional: the recording this run
                                           replays, when it is not the one
                                           config.SHAPES names. Relative to
                                           the editable kit
      "inputs": [                          optional: any other file the
                                           children will open late
        {"label": "...", "path": "...", "min_records": 1}]
    }

EVERY FILE A MANIFEST NAMES IS PROVEN AT SESSION OPEN. The capture and
anything under "inputs" are checked, hashed and recorded by the
manifest-inputs gate before a single child spawns, and their hashes are
recorded against every replay cell: a recording is an input pin like a
source file. A missing one refuses the session at minute zero instead of
one point at a time.

The render layer owns manifest_id, phase, shape_label, the per-arm row
label, control_rationale, recommended, recommended_when_available,
comparisons and note. This driver carries them through and reads none of
them: row identity comes from the config-origin stamp on the cell, never
from a label in a manifest.

ONE ARM, TWO CONFIGURATIONS. An arm is bound once per profile it runs, in
its own child, and a manifest may do exactly that: the production arm is a
row at upstream defaults and the session control at the as-served
profile. Points, children and identities are keyed on (arm, profile), so
the two never merge, and the control's child binds a different tree from
the row's child without either noticing.

THE RESOLVED PLAN IS WHAT RUNS AND WHAT IS GATED. --only and --batches
narrow the manifest; they cannot widen it. The list of cells about to run,
after those selectors, is the input to every gate, and a point in it that
the manifest does not declare fails the build-map gate. A selector that
reaches the measurement without reaching the gates is how an ungated arm
enters a table.

THE CONTROL CELLS CARRY ROLES. Each control batch is measured twice, once
before the arms and once after, in two children, stamped role control-open
and control-close. The role is in the fingerprint, so the two readings are
two cell_ids instead of one address written twice, and their exclusion from
tables is structural rather than a rendering convention.

PRE-REGISTERED REFUSALS ARE SCORED, NOT GATED. expected_refusals is a
prediction made before the run. The build-map gate reports which
predictions held, which missed and which refusals were not predicted, and
every line is printed. A prediction that missed does not stop the session:
a kernel that refused where it was expected to build, or built where it was
expected to refuse, is a finding, and the manifest is corrected on the
record rather than the session being spent on it.

THE ARM INTERFACE

arms/<name>.py defines one function:

    bind(tree_path, shape, profile_flags) -> dict

tree_path is the manifest's tree for the arm, or None for the arm's own
default. shape is the config.SHAPES entry, verbatim. profile_flags is the
config.PROFILES entry's flags dict, or None when the profile has no values
of its own (owner-selector and owner-default), in which case the arm
resolves the owner's values itself and returns them.

The returned dict, required keys:

    call         fn(x, gating) -> out. The layer, jitted BY THE ARM, one
                 program. x is [batch, hidden] bfloat16. gating is a dict
                 with logits [batch, experts] bfloat16, idx [batch, top_k]
                 int32, weights [batch, top_k] float32 and group_sizes
                 [experts] int32; an arm reads the entries its contract
                 names and ignores the rest. No host work, no placement and
                 no cast inside call: everything the benchmark owns happens
                 before the window.
    contract     the window contract, checked against the trace by
                 gates.window_census:
                   anchor                hlo operation name prefix (str or
                                         tuple) of an operation that runs
                                         exactly once per call
                   router_in_window      bool, does the selection run inside
                                         the timed program
                   dispatch_collectives  tuple from gates.COLLECTIVE_KINDS
                   combine_kind          one of gates.COMBINE_KINDS
                   gating_consumed       tuple of the gating keys call reads
                   executions_per_call   optional, only for an arm whose
                                         program runs more than once per
                                         call per device. The default is the
                                         device count and more than that
                                         raises rather than being averaged
    source_hash  gates.source_digest(sources, tree_path). Do not roll your
                 own digest: the gate re-derives this one.
    tree_sha     commit of the tree the arm bound.

Optional keys:

    sources      the kernel source files the arm executed, absolute paths.
                 Without them the source gate has nothing to re-derive and
                 fails the run.
    pin          name of the pins.json entry the sources are pinned by.
                 Absent for arms binding this branch's own tree, whose
                 files are hashed and recorded instead.
    namespace    the top-level module the arm resolves, e.g. tpu_inference.
                 The source gate checks the loaded module's file is under
                 the arm's tree, which is what keeps an arm from measuring
                 the environment's own installed package and citing a pin.
    flags        the resolved flag values in force. REQUIRED when
                 profile_flags is None, because a cell cannot be written
                 without the verbatim dict of values in force.
    prepare      fn(case) -> (x, gating), the arm's own placement and
                 sharding, called once per cell outside the window.
                 harness.place is used when this is absent.
    build_check  fn(batch) -> None, the arm's own deviceless build check.
                 The default traces and lowers call at the case shapes.
    verify_mirror fn() -> (ok, detail), proof that the flags are the
                 owner's selection. Required for an owner-selector profile.
                 It runs in the child; the gate scores its verdict.
    close        fn() -> None, released after the arm's last cell.
    tree         the tree path the arm actually bound, when it differs from
                 the manifest's.

HOW AN ARM REFUSES. It raises one of the typed refusals in harness.py, and
the type decides the attribution the cell carries:

    harness.KernelRefusal            kernel
    harness.ShapeConstraintRefusal   shape-constraint
    harness.FetchRefusal             fetch
    harness.OutOfScopeRefusal        declared-out-of-scope
    harness.HarnessRefusal           harness

Anything else it raises is attributed to the kernel, because it came out of
the kernel's own code path. An AssertionError is NOT a refusal: it says
something the instrument believed impossible happened, so the child exits
5, the parent stops the run, and no cell is written for it. Booking
assertion failures as benign refusals is how a harness artefact gets
published as a kernel's refusal.

Weights are the arm's own business with one requirement: they are derived
from the shape alone, deterministically, so two arms multiply identical
numbers. The numerics gate compares outputs and has no other way to know
they were the same layer.

WHERE AN ARM FINDS ITS OWN TREE. The snapshot re-executes a COPY of the arm
modules, and only the modules: fetched checkouts under arms/ are pinned by
content hash and are not copied. An arm therefore resolves its tree from
BENCHOFF_KIT_ORIGIN, which the snapshot exports to every child, rather than
from its own module location, which is inside the run directory.

THE COMPILER FRAME IS THE DRIVER'S, NOT THE ARM'S, AND IT IS CONTENT. Every
arm's call is compiled in its child under exactly
config.FRAMES[manifest frame], no arm passes compiler options of its own,
the resolved option dict is stamped on every cell as frame_options, the run
refuses if anything in the environment would extend or override it, and the
persistent compile cache is keyed on a hash of that dict so a shared cache
cannot serve a binary compiled under other options. One kernel's tuning
options riding into a comparison as if they were the serving frame is the
failure this closes.

NOTHING RESOLVES BY FILE ORDER. Cells are identified by cells.cell_id,
which is a hash of the fingerprint. No listing order, no newest-first
directory read and no modification time decides which file a number came
from.

THE SEQUENCE

     1. snapshot and re-execute from the run directory
     2. resolve the plan, isolate the compile cache
     3. probe child for the device facts, the environment gate, then the
        manifest-inputs gate, which refuses the session before any child
        spawns if a file the plan needs is absent or cannot serve it
     4. one child per configuration, in session order: bind, build-check,
        and, in a device run, measure. A refusal becomes a cell with an
        attribution; a window census runs on each configuration's first ok
        measurement
     5. the source gate, the mirror gate, the build-map gate and its scope
        control, over what the children reported
     6. the deliberate-breakage control, through a child and the parent's
        own cell path
     7. controls gate over the opening and closing children, then
        session_drift_us backfilled onto every cell
     8. numerics gate against the reference configuration
     9. cells written through cells.write_cells, run report beside them

REFUSED VERSUS FAILED. A point that cannot build is refused: its cell says
so, with an attribution, and the run continues. A point that built and then
could not be measured is failed. The distinction is mechanical, not
editorial: refusals come from a child's build map and failures from its
measurements.

EXIT CODES. 0 the run completed. 2 a gate refused it. 3 cells failed. 5 a
defect: an assertion fired, and the instrument is what is broken. A control
that did not hold is a refused gate like any other and exits 2.
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import cells  # noqa: E402
import config  # noqa: E402
import gates  # noqa: E402
import harness  # noqa: E402

SNAPSHOT_ENV = "BENCHOFF_SNAPSHOT"
ORIGIN_ENV = "BENCHOFF_KIT_ORIGIN"

# The persistent compile cache is keyed per frame under this directory, so a
# binary compiled under one option dict can never be served to a cell
# claiming another. The variable name is jax's own; this kit sets it rather
# than inheriting whatever the shell had, and every child inherits it. The
# variable is read at `import jax`, so it is set before any child starts.
CACHE_ENV = "JAX_COMPILATION_CACHE_DIR"
# Under the results root, beside the run directories: a cache is run output.
# At the default root that is results/, which the kit's .gitignore already
# excludes, so it never appears as an untracked directory in the tree.
CACHE_DIR_NAME = "compile-cache"

# WHERE RUN OUTPUT LIVES. By default results/ beside the kit, so a reviewer
# who takes the kit and runs it finds its output next to it with nothing
# configured. BENCHOFF_RESULTS_ROOT moves that home and changes nothing else:
# a run directory's layout is identical wherever its root is, and every path
# inside a report is absolute already. Point it at a large disk when the
# filesystem holding the default root is small: a session's captures and
# profiles can fill a small one while it is still measuring.
RESULTS_ROOT_ENV = "BENCHOFF_RESULTS_ROOT"


def results_root():
    """The directory run dirs and the compile cache are made under.

    Only run output moves. The rendered tables committed in the kit's
    results/ are source and stay where they are: they are read by a reader of
    the repository, not written by a run.
    """
    named = os.environ.get(RESULTS_ROOT_ENV)
    if named:
        return os.path.abspath(os.path.expanduser(named))
    return os.path.join(origin_kit(), "results")

# The runtime's spelling of config.DEVICE_REQUIRED["kind"], needed by a
# deviceless build map to answer trace-time device probes. It matches the
# device_kind string a stored capture carries.
RUNTIME_DEVICE_KIND = "TPU7x"

# The one flag the deviceless phase is allowed to put in the environment: it
# selects how many host platform devices exist, which is a platform choice
# and not a compiler option. Every other xla flag in the environment refuses
# the run.
DEVICELESS_ALLOWED_FLAG = "--xla_force_host_platform_device_count"

# What a refused replay cell carries where its steps would be. cells.py
# requires a replay cell's replay_steps to be non-empty, and a refused cell
# ran no steps, so it says exactly that instead of naming calls it never
# replayed. Render reads this as "no steps", never as a capture id.
NO_STEPS = [["refused-before-step-selection", 0]]

# A child that hit a defect exits with this code, and the parent turns it
# back into harness.Defect. A defect is not a refusal and never a cell.
DEFECT_EXIT = 5

# How an editable install names itself. setuptools writes a .pth that imports
# a finder module and appends its finder to sys.meta_path, and that finder
# answers for a package from wherever the install was made, which is never a
# tree this kit pinned. For an editable tpu_inference the file in
# site-packages is __editable__.tpu_inference-0.0.0.pth, whose single line is
#   import __editable___tpu_inference_0_0_0_finder; ...install()
# and whose finder carries
#   MAPPING = {"tpu_inference": "the directory the editable install was
#              made from"}.
EDITABLE_FINDER_PREFIX = "__editable__"


class PlanRefused(harness.HarnessRefusal):
    """The run cannot start: the manifest or the selection does not resolve."""


# ---------------------------------------------------------------------------
# The Snapshot Guard
# ---------------------------------------------------------------------------
def snapshot_and_reexec(argv, manifest_path, run_dir):
    """Copy the kit and its inputs into the run directory, run the copy.

    A python file is read whole before it executes, so the mid-byte hazard a
    shell script faces is bash's alone. What remains, and what this closes,
    is everything read LATER: the arm modules, the manifest, the pins, this
    file's own imports, and every child process, which is this same file from
    the snapshot. An edit during a run would otherwise land inside the run,
    and the run directory would not hold the code that produced its cells.
    """
    if os.environ.get(SNAPSHOT_ENV):
        return None
    kit = os.path.join(run_dir, "kit")
    os.makedirs(kit, exist_ok=True)
    for name in sorted(os.listdir(HERE)):
        source = os.path.join(HERE, name)
        if os.path.isfile(source) and (name.endswith(".py")
                                       or name.endswith(".json")
                                       or name == config.ENV_LOCK_FILE):
            shutil.copy2(source, os.path.join(kit, name))
    # The arm modules, and only the modules. Subdirectories of arms/ hold
    # fetched checkouts, which are pinned by content hash and are not kit
    # source: the pins are the record, not the checkouts (.gitignore:1-3).
    # Copying them would put tens of megabytes in every run directory and
    # would point each arm's tree at a copy that no pin describes.
    arms_dir = os.path.join(HERE, "arms")
    if os.path.isdir(arms_dir):
        os.makedirs(os.path.join(kit, "arms"), exist_ok=True)
        for name in sorted(os.listdir(arms_dir)):
            source = os.path.join(arms_dir, name)
            if os.path.isfile(source) and name.endswith(".py"):
                shutil.copy2(source, os.path.join(kit, "arms", name))
    snapshot_manifest = os.path.join(kit, "manifest.json")
    shutil.copy2(manifest_path, snapshot_manifest)
    argv = list(argv)
    for i, arg in enumerate(argv):
        if arg == "--manifest" and i + 1 < len(argv):
            argv[i + 1] = snapshot_manifest
    if "--run-dir" not in argv:
        # The re-exec inherits THIS run directory rather than recomputing one
        # from the clock: a re-exec crossing a second boundary would write
        # into a directory that does not hold the kit that produced it.
        argv += ["--run-dir", run_dir]
    env = dict(os.environ)
    env[SNAPSHOT_ENV] = kit
    env[ORIGIN_ENV] = HERE
    target = os.path.join(kit, os.path.basename(os.path.abspath(__file__)))
    print(f"[snapshot] running {target}, not the editable kit")
    sys.stdout.flush()
    os.execve(sys.executable, [sys.executable, target] + argv, env)


def origin_kit():
    """The editable kit this run was launched from. Cells land there."""
    return os.environ.get(ORIGIN_ENV) or HERE


_KIT_SOURCE_HASH = None


def kit_source_hash():
    """One digest over the kit's own python sources, stamped on every cell.

    The instrument names itself: a stored cell measured by a run.py that no
    longer exists on disk is otherwise unfalsifiable, and a kit edited
    between a child's spec and its measurement would leave no trace.
    Computed from HERE, which inside a run is the snapshot, so the parent
    and every child of one session derive the same value.
    """
    global _KIT_SOURCE_HASH
    if _KIT_SOURCE_HASH is None:
        import hashlib
        digest = hashlib.sha256()
        for root in (HERE, os.path.join(HERE, "arms")):
            if not os.path.isdir(root):
                continue
            for name in sorted(os.listdir(root)):
                path = os.path.join(root, name)
                if os.path.isfile(path) and name.endswith(".py"):
                    digest.update(name.encode())
                    with open(path, "rb") as fh:
                        digest.update(fh.read())
        _KIT_SOURCE_HASH = digest.hexdigest()[:16]
    return _KIT_SOURCE_HASH


# ---------------------------------------------------------------------------
# The Manifest And The Resolved Plan
# ---------------------------------------------------------------------------
def read_manifest(path):
    """The manifest, with every field this driver depends on checked here.

    A manifest that names a shape, frame, profile, tier or batch this kit
    does not have is refused before anything is bound: a run that discovers
    that at cell 40 has spent the session.

    Fields this driver reads: shape_id, frame, seed, cells, tier, arms
    (arm, profile, routings, batches, and optionally tier, tree, flags),
    control (arm, profile, routing, batch list, tier), numerics (reference,
    batch, routing), expected_refusals (arm, batch, attribution, reason),
    capture (a recording overriding the shape's) and inputs (other files
    children read late).
    Fields the render layer owns are carried through untouched and are
    listed in this module's docstring.
    """
    with open(path) as fh:
        manifest = json.load(fh)
    # A manifest may pin the start-up exclusion factor for COLUMN COHERENCE:
    # every row of a published column must replay the same drawn steps, and
    # the draw depends on the eligible pool, which depends on this factor.
    # A cell re-measured into a column whose other rows were drawn under the
    # old factor must pin the old factor; a column re-measured whole uses
    # the current default. The value used is stamped with the session's
    # snapshot of this manifest.
    if "replay_exclude_factor" in manifest:
        config.REPLAY_EXCLUDE_ACTIVE_FACTOR = int(
            manifest["replay_exclude_factor"])
    problems = []
    for field in ("shape_id", "frame", "seed", "cells", "arms"):
        if field not in manifest:
            problems.append(f"the manifest has no {field!r}")
    if manifest.get("shape_id") not in config.SHAPES:
        problems.append(f"shape_id {manifest.get('shape_id')!r} is not in "
                        f"config.SHAPES")
    if manifest.get("frame") not in config.FRAMES:
        problems.append(f"frame {manifest.get('frame')!r} is not in "
                        f"config.FRAMES")
    default_tier = manifest.get("tier", "heavy")
    if default_tier not in config.TIERS:
        problems.append(f"tier {default_tier!r} is not in config.TIERS")
    for entry in manifest.get("arms") or []:
        for field in ("arm", "profile", "routings", "batches"):
            if field not in entry:
                problems.append(f"an arm entry has no {field!r}: {entry}")
        if entry.get("profile") not in config.PROFILES:
            problems.append(f"profile {entry.get('profile')!r} is not in "
                            f"config.PROFILES")
        for routing in entry.get("routings") or []:
            if routing not in cells.VALID_ROUTING:
                problems.append(f"routing {routing!r} is not one of "
                                f"{cells.VALID_ROUTING}")
        for batch in entry.get("batches") or []:
            if batch not in config.BATCHES:
                problems.append(f"batch {batch} is not in config.BATCHES")
        if entry.get("tier", default_tier) not in config.TIERS:
            problems.append(f"tier {entry.get('tier')!r} is not in "
                            f"config.TIERS")
        if (config.PROFILES.get(entry.get("profile"), {}).get("flags") is None
                and entry.get("profile") == "ours-tuned"
                and not isinstance(entry.get("flags"), dict)):
            problems.append(
                f"{entry.get('arm')} runs the ours-tuned profile, whose "
                f"values are a choice this manifest makes, so the entry has "
                f"to carry the flags dict it chose")
    control = manifest.get("control")
    if control:
        for field in ("arm", "profile"):
            if field not in control:
                problems.append(f"the control block has no {field!r}")
        if control.get("profile") not in config.PROFILES:
            problems.append(f"control profile {control.get('profile')!r} is "
                            f"not in config.PROFILES")
        if control.get("routing", "random") not in cells.VALID_ROUTING:
            problems.append(f"control routing {control.get('routing')!r} is "
                            f"not one of {cells.VALID_ROUTING}")
    for expected in manifest.get("expected_refusals") or []:
        for field in ("arm", "batch"):
            if field not in expected:
                problems.append(f"an expected_refusals entry has no "
                                f"{field!r}: {expected}")
        if (expected.get("attribution")
                and expected["attribution"] not in cells.VALID_ATTRIBUTION):
            problems.append(
                f"expected refusal attribution "
                f"{expected.get('attribution')!r} is not one of "
                f"{cells.VALID_ATTRIBUTION}")
    if "capture" in manifest and not isinstance(manifest["capture"], str):
        problems.append(f"capture {manifest.get('capture')!r} is not a path")
    for extra in manifest.get("inputs") or ():
        if not isinstance(extra, dict) or not extra.get("path"):
            problems.append(f"an inputs entry names no path: {extra}")
    numerics = manifest.get("numerics")
    if numerics and numerics.get("reference") not in {
            e.get("arm") for e in manifest.get("arms") or []}:
        problems.append(f"the numerics reference "
                        f"{numerics.get('reference')!r} is not an arm of this "
                        f"manifest")
    if problems:
        raise PlanRefused("the manifest cannot be run: " + "; ".join(problems))
    manifest["tier"] = default_tier
    return manifest


def control_batches(manifest):
    """Where the controls run: every batch any arm measures, unless the
    manifest names fewer. A cell whose own count has no control can only be
    stamped with no drift at all, and the report says which counts those
    were."""
    control = manifest.get("control") or {}
    if control.get("batches"):
        return sorted(int(b) for b in control["batches"])
    everything = set()
    for entry in manifest["arms"]:
        everything.update(int(b) for b in entry["batches"])
    return sorted(everything)


def control_entry(manifest):
    """The control arm as an arm entry, or None."""
    control = manifest.get("control")
    if not control:
        return None
    return {"arm": control["arm"], "profile": control["profile"],
            "routings": [control.get("routing", "random")],
            "batches": control_batches(manifest),
            "tier": control.get("tier", manifest["tier"]),
            "tree": control.get("tree"), "flags": control.get("flags")}


def registry_key(arm, profile):
    """How a bound configuration is named, in the plan and in the children.

    The arm alone is not enough: one arm runs under two configurations in a
    manifest, and a session control is exactly that (the production arm at
    the as-served profile, while the production row is the same arm at
    upstream defaults). Two configurations are two children, two build-map
    points and two identities.
    """
    return f"{arm}@{profile}"


def declared_points(manifest):
    """Every (arm, profile, batch) the manifest allows, controls included."""
    points = set()
    control = control_entry(manifest)
    for entry in manifest["arms"] + ([control] if control else []):
        for batch in entry["batches"]:
            points.add((entry["arm"], entry["profile"], int(batch)))
    return sorted(points)


def resolve_plan(manifest, only=None, batches=None):
    """The exact list of cells about to run, after the command line.

    Every gate is given this list. A selector narrows it; nothing widens it,
    and the build-map gate refuses a plan holding a point the manifest does
    not declare.

    Each point carries the role the cell will be stamped with: measure for a
    manifest arm, control-open and control-close for the two readings of the
    control.
    """
    only = set(only or ())
    batches = set(int(b) for b in (batches or ()))
    allowed = set(declared_points(manifest))
    plan = []
    control = control_entry(manifest)
    for entry in manifest["arms"]:
        if only and entry["arm"] not in only:
            continue
        for routing in entry["routings"]:
            for batch in entry["batches"]:
                if batches and int(batch) not in batches:
                    continue
                plan.append({
                    "role": "measure", "arm": entry["arm"],
                    "key": registry_key(entry["arm"], entry["profile"]),
                    "profile": entry["profile"], "routing": routing,
                    "batch": int(batch),
                    "tier": entry.get("tier", manifest["tier"]),
                    "tree": entry.get("tree")})
    if control:
        for role in ("control-open", "control-close"):
            for batch in control["batches"]:
                if batches and int(batch) not in batches:
                    continue
                plan.append({
                    "role": role, "arm": control["arm"],
                    "key": registry_key(control["arm"], control["profile"]),
                    "profile": control["profile"],
                    "routing": control["routings"][0], "batch": int(batch),
                    "tier": control["tier"], "tree": control.get("tree")})
    if not plan:
        raise PlanRefused(
            f"the selection resolves to no cells at all: only={sorted(only)}, "
            f"batches={sorted(batches)}")
    outside = [f"{p['arm']}@{p['profile']}:B{p['batch']}" for p in plan
               if (p["arm"], p["profile"], p["batch"]) not in allowed]
    if outside:
        # The gate refuses this too. Saying it here as well means the reason
        # is visible before any child starts.
        print(f"[plan] the resolved plan holds points the manifest does not "
              f"declare: {sorted(set(outside))}")
    return plan


def plan_entries(manifest, plan):
    """{registry key: entry} for the configurations the plan actually runs."""
    wanted = {}
    control = control_entry(manifest)
    catalogue = {registry_key(e["arm"], e["profile"]): e
                 for e in manifest["arms"]}
    if control:
        catalogue.setdefault(registry_key(control["arm"], control["profile"]),
                             control)
    for point in plan:
        entry = catalogue.get(point["key"])
        if entry is None:
            entry = {"arm": point["arm"], "profile": point["profile"],
                     "routings": [point["routing"]],
                     "batches": [point["batch"]], "tier": point["tier"],
                     "tree": point.get("tree")}
        wanted[point["key"]] = entry
    return wanted


def numerics_plan(manifest, plan):
    """Which configuration every other one is compared against, and where.

    The reference is named by arm; the configuration compared against is
    that arm's measure point, never its control reading, because a control
    is not a row.
    """
    spec = manifest.get("numerics") or {}
    reference = spec.get("reference")
    if not reference:
        return None
    points = [p for p in plan
              if p["arm"] == reference and p["role"] == "measure"]
    if not points:
        return None
    batch = spec.get("batch")
    if batch is None:
        batch = min(int(p["batch"]) for p in points)
    routing = spec.get("routing")
    if routing is None:
        routing = points[0]["routing"]
    return {"reference": reference, "reference_key": points[0]["key"],
            "batch": int(batch), "routing": routing}


# ---------------------------------------------------------------------------
# Compile Frame Isolation
# ---------------------------------------------------------------------------
def isolate_compile_cache(frame_name, frame):
    """Point the persistent compile cache at a directory keyed on the frame.

    Set in the parent before any child starts, because children inherit the
    environment and the variable is read at `import jax`. Two frames that
    differ in one option get different directories, so a cache warmed under
    other options cannot serve this run's cells.

    Expect this in a device session's logs and do not chase it:

        Failed to serialize TpuExecutableProto ... 5.19GB

    The production pair path's executable is larger than the cache's entry
    limit, so that entry is skipped. The compile still happens and the cells
    are unaffected; only the reuse of that one binary across runs is lost.
    """
    fingerprint = gates.frame_fingerprint(frame)
    previous = os.environ.get(CACHE_ENV)
    path = os.path.join(results_root(), CACHE_DIR_NAME,
                        f"{frame_name}-{fingerprint}")
    os.makedirs(path, exist_ok=True)
    os.environ[CACHE_ENV] = path
    return {"cache_dir": path, "frame_fingerprint": fingerprint,
            "previous_cache_dir": previous}


def apply_libtpu_args(frame):
    """Set LIBTPU_INIT_ARGS from the frame, in the parent, before any child.

    THE OTHER HALF OF THE FRAME. The serving compile environment is two
    things: the jit options a call carries, and the flags libtpu is
    initialized with. A frame has both halves and both are set here; carry
    only the first and the frame-hazard check refuses the serving runtime
    flags as if they were contamination.

    Set here, once, beside the compile cache, so every child inherits exactly
    the frame's runtime half and nothing else. Deduplicated in the frame's own
    order: the serving tree's env_override prepends one of these flags at
    import time without checking, and a token repeated is the same token.
    Cleared outright when a frame declares no runtime half, so a shell that
    had one cannot leak into a plain-frame run.
    """
    declared = gates.frame_libtpu_args(frame)
    previous = os.environ.get("LIBTPU_INIT_ARGS")
    deduped = list(dict.fromkeys(declared))
    if deduped:
        os.environ["LIBTPU_INIT_ARGS"] = " ".join(deduped)
    else:
        os.environ.pop("LIBTPU_INIT_ARGS", None)
    return {"declared": deduped, "previous": previous,
            "value": os.environ.get("LIBTPU_INIT_ARGS")}


def deviceless_env(chips):
    """The host-platform environment for a deviceless pass. Before jax.

    Set in the parent so every child inherits it, since JAX_PLATFORMS is
    read at `import jax`.
    """
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" {DEVICELESS_ALLOWED_FLAG}={int(chips)}").strip()


def pin_mosaic_probe(kind=RUNTIME_DEVICE_KIND, cores=1):
    """Answer the trace-time device probe so kernels lower on a host box.

    Called in a child, after jax is importable, before tracing.
    """
    try:
        from jax._src.pallas.mosaic import tpu_info as probe
    except ImportError:
        return False
    probe.get_device_kind = lambda: kind
    probe.get_num_device_cores = lambda: cores
    return True


# ---------------------------------------------------------------------------
# Facts, Shared By The Parent And Its Children
# ---------------------------------------------------------------------------
class Facts:
    """What every cell of one run shares, on either side of the boundary.

    The parent builds these from the manifest and hands the serializable
    part to each child, so a child cannot invent a session id, a seed, a
    frame or a device string of its own.
    """

    def __init__(self, manifest, run_dir, deviceless, device=None,
                 env_hash=None, session_id=None, keep_traces=False):
        self.manifest = manifest
        self.run_dir = run_dir
        self.deviceless = deviceless
        self.shape_id = manifest["shape_id"]
        self.shape = config.SHAPES[self.shape_id]
        self.frame = manifest["frame"]
        # The whole frame entry, both halves. Cells stamp it as
        # frame_options, which is the field cells.py names, and its content is
        # what two cells are compared on: the jit options AND the runtime
        # flags, because a frame is both.
        self.frame_options = dict(config.FRAMES[manifest["frame"]])
        self.compiler_options = gates.frame_compiler_options(
            self.frame_options)
        self.seed = int(manifest["seed"])
        self.session_id = session_id or os.path.basename(
            os.path.normpath(run_dir))
        self.env_hash = env_hash or gates.env_hash()
        # Retention is a session fact, handed to every child like the rest:
        # a child cannot decide on its own to keep or drop a capture.
        self.keep_traces = bool(keep_traces)
        self.device = device
        self.numerics = None
        self.capture = None
        # Filled by the manifest-inputs gate at session open: {label: sha256}
        # for every file the children will read late.
        self.input_hashes = {}

    def portable(self):
        """The part a child is given, verbatim."""
        return {"run_dir": self.run_dir, "deviceless": self.deviceless,
                "session_id": self.session_id, "env_hash": self.env_hash,
                "device": self.device, "keep_traces": self.keep_traces}

    def capture_path(self):
        """The recorded capture this run replays, resolved once.

        The shape names one; a manifest may name another, because two
        manifests at the same widths can replay different recordings.

        Resolution is against the EDITABLE kit, never against the snapshot: a
        capture is a large pinned input, so the snapshot copies the kit's
        code and reaches back for the recording, exactly as the arms reach
        back for their fetched trees. Children inherit BENCHOFF_KIT_ORIGIN
        and so resolve the same file the session-open gate hashed.
        """
        path = self.manifest.get("capture") or self.shape["capture"]
        if not path:
            return None
        if not os.path.isabs(path):
            path = os.path.join(origin_kit(), path)
        return path

    def capture_records(self):
        if self.capture is None:
            path = self.capture_path()
            if not path:
                raise harness.ShapeConstraintRefusal(
                    f"shape {self.shape_id} has no recorded routing, so a "
                    f"replay cell cannot exist for it")
            self.capture = harness.load_capture(path)
        return self.capture


# ---------------------------------------------------------------------------
# The Child: One Configuration, One Tree
# ---------------------------------------------------------------------------
def env_registry_names(roots, limit_depth=3):
    """Every environment variable a tree DECLARES, read from its registry.

    A serving tree keeps one registry of the variables it reads: a module
    named envs.py holding `environment_variables`, a dict from variable name
    to a lambda that reads it. The names are parsed out of the source with
    ast, never imported and never hand-listed here, so a tree that adds a
    variable is quarantined for it the moment it is bound.

    The registry has that shape in every tree this kit binds:
    tpu_inference/envs.py, several dozen declared names, resolved lazily
    through the module's __getattr__, which is why scrubbing before the layer
    is built is enough to change what it reads.
    """
    import ast
    names, sources = set(), []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for base, dirs, files in os.walk(root):
            depth = base[len(root):].count(os.sep)
            if depth >= limit_depth:
                dirs[:] = []
            dirs[:] = [d for d in dirs if not d.startswith(("__", "."))]
            if "envs.py" not in files:
                continue
            path = os.path.join(base, "envs.py")
            try:
                tree = ast.parse(open(path).read())
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                target = None
                if isinstance(node, ast.AnnAssign):
                    target = getattr(node.target, "id", None)
                elif isinstance(node, ast.Assign) and node.targets:
                    target = getattr(node.targets[0], "id", None)
                if target != "environment_variables":
                    continue
                if not isinstance(node.value, ast.Dict):
                    continue
                found = [k.value for k in node.value.keys
                         if isinstance(k, ast.Constant)
                         and isinstance(k.value, str)]
                if found:
                    names.update(found)
                    sources.append({"registry": path, "declared": len(found)})
    return names, sources


# What the quarantine leaves alone, one justification per entry. Every one of
# these is set BY THIS DRIVER, deliberately, and none of them changes what a
# layer computes:
#   JAX_COMPILATION_CACHE_DIR  the frame-keyed cache directory this file sets,
#                              which replaces whatever the shell had
#   JAX_PLATFORMS              the platform the deviceless pass selects; it
#                              chooses where a program runs, not what it is
#   ALLOW_MULTIPLE_LIBTPU_LOAD the deviceless setup's own flag
#   XLA_FLAGS                  carries only the host-platform device count in
#                              the deviceless pass, and the frame-hazard
#                              check refuses anything else in it
#   BENCHOFF_SNAPSHOT          this kit's own, not a tree's
#   BENCHOFF_KIT_ORIGIN        this kit's own, not a tree's
ENV_QUARANTINE_ALLOWLIST = (CACHE_ENV, "JAX_PLATFORMS",
                            "ALLOW_MULTIPLE_LIBTPU_LOAD", "XLA_FLAGS",
                            SNAPSHOT_ENV, ORIGIN_ENV, RESULTS_ROOT_ENV)

# What the post-quarantine snapshot always reports, whether or not it is set:
# the two channels that reach the compiler and the runtime, so a cell's basis
# says what they held at measurement time.
ENV_WATCHLIST = ("XLA_FLAGS", "LIBTPU_INIT_ARGS", CACHE_ENV, "JAX_PLATFORMS")


def quarantine_env(names, allowlist=ENV_QUARANTINE_ALLOWLIST):
    """Remove every named variable from this process, and say what it held.

    THE FAILURE THIS CLOSES. A child inherits the whole parent environment,
    so any variable a serving tree reads can change a measured program
    without appearing anywhere in the cell: a serving config that exports
    NEW_MODEL_DESIGN=1 flips the sharding axis-name class on the measured
    path (sharding.py:115), and MOE_APPROX_TOPK, FORCE_MOE_RANDOM_ROUTING or
    USE_2D_TP would alter the routing outright. Without this, a clean
    launching shell is doing the work that this does.

    Configuration reaches an arm as call parameters, from the profile the cell
    stamps, and never through the environment.
    """
    removed = {}
    for name in sorted(names):
        if name in allowlist:
            continue
        if name in os.environ:
            removed[name] = os.environ.pop(name)
    return removed


def env_watchlist_snapshot(names):
    """What is still set after the quarantine, for the record.

    Anything the registries declare that somehow survived, the two channels
    that reach the compiler and the runtime, and every JAX variable in the
    process. A snapshot with nothing in it is itself the evidence.
    """
    watched = set(ENV_WATCHLIST) | set(names)
    snapshot = {name: os.environ[name] for name in sorted(watched)
                if name in os.environ}
    for name in sorted(os.environ):
        if name.startswith("JAX_"):
            snapshot[name] = os.environ[name]
    return snapshot


def child_isolate(run_dir, kit=None, tree_roots=()):
    """Leave this child no way to import an arm's package except its tree.

    THE FAILURE THIS CLOSES. A child can import the serving package from the
    checkout the kit itself lives in rather than from the tree the arm bound,
    and the run then books the resulting import error as the kernel's
    refusal. Two channels do that, and neither is the arm's fault:

    A checkout on the import path. The working directory, an inherited path
    entry or the repository above the kit all put a checkout where an
    ordinary `import` will find it, and whichever one comes first answers for
    every arm. So the current directory becomes the run directory, and every
    path entry that is a checkout root, along with the two entries that mean
    "here", is removed before any arm module is imported. The kit's own
    directory stays: that is where cells.py, config.py and arms/ live.

    An editable install. It answers through sys.meta_path rather than
    sys.path, from wherever someone last installed the package, so no path
    hygiene can reach it. Every editable finder is removed and whatever it
    claimed is purged from sys.modules.

    AND THE ENVIRONMENT IS QUARANTINED. tree_roots names the trees whose
    registries declare the variables a serving tree reads; every one of those
    names is scrubbed from this process before an arm module is imported, so
    configuration reaches an arm as call parameters and never through the
    environment. What was scrubbed, with the values it held, and a
    post-quarantine watchlist snapshot come back in the detail.

    Returns what it did, and the child reports it, because an import path or
    an environment that was quietly adjusted is not a fingerprint.
    """
    kit = os.path.realpath(kit or HERE)
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)
    detail = {"cwd": os.getcwd(), "kept_kit": kit, "removed_paths": [],
              "removed_finders": [], "purged_modules": [],
              "quarantine": {}, "registries": [], "watchlist": {},
              # Everything this child INHERITED, by name only. A variable that
              # is set later and is not in here was set by the arm itself
              # while it bound, which is the arm's own declared behaviour and
              # not an inherited value; the two are recorded apart.
              "inherited_names": sorted(os.environ)}
    dropped = []
    keep = []
    for entry in list(sys.path):
        resolved = os.path.realpath(entry) if entry else os.getcwd()
        if entry in ("", "."):
            detail["removed_paths"].append(entry or "(the working directory)")
            dropped.append(resolved)
            continue
        if resolved != kit and os.path.exists(os.path.join(resolved, ".git")):
            detail["removed_paths"].append(entry)
            dropped.append(resolved)
            continue
        keep.append(entry)
    sys.path[:] = keep
    claimed = set()
    for finder in list(sys.meta_path):
        module = getattr(finder, "__module__", None) or type(finder).__module__
        if not str(module).startswith(EDITABLE_FINDER_PREFIX):
            continue
        sys.meta_path.remove(finder)
        detail["removed_finders"].append(str(module))
        claimed.update((getattr(finder, "MAPPING", None) or {}).keys())
        claimed.update((getattr(finder, "NAMESPACES", None) or {}).keys())
    detail["editable_claimed"] = sorted(claimed)
    for name in sorted(sys.modules):
        if "." in name:
            continue
        module = sys.modules.get(name)
        path = getattr(module, "__file__", None)
        under_dropped = path and any(
            os.path.realpath(path).startswith(d + os.sep) for d in dropped)
        if name in claimed or under_dropped:
            del sys.modules[name]
            detail["purged_modules"].append(name)
    names, registries = env_registry_names(tree_roots)
    detail["registries"] = registries
    detail["declared_variables"] = len(names)
    detail["quarantine"] = quarantine_env(names)
    detail["watchlist"] = env_watchlist_snapshot(names)
    # The compile environment as the SHELL left it, after the quarantine and
    # before a single line of the code under measurement is imported. This is
    # the snapshot the frame is held to: whatever an import adds afterwards is
    # that code acting on its own process and is recorded rather than refused.
    detail["frame_environment_pre_import"] = gates.frame_environment(names)
    return detail


def load_arm(name):
    """arms/<name>.py, imported, with its bind function returned.

    The registry is the directory: an arm exists when its module does. No
    list of names lives in this file, so adding an arm is adding a file.
    """
    import importlib
    if not name.replace("_", "").isalnum():
        raise PlanRefused(f"arm name {name!r} is not a module name")
    path = os.path.join(HERE, "arms", f"{name}.py")
    if not os.path.exists(path):
        raise harness.FetchRefusal(f"there is no arm at {path}")
    module = importlib.import_module(f"arms.{name}")
    bind = getattr(module, "bind", None)
    if bind is None:
        raise harness.HarnessRefusal(f"arms/{name}.py defines no bind "
                                     f"function")
    return bind


def bind_arm(entry, shape, compiler_options):
    """One configuration of one arm bound, checked, and put in the frame.

    WHICH FLAGS REACH THE ARM. A profile that carries values passes them in
    and they are what the cell stamps. A profile that carries none is one of
    two things. Either the values are the owner's, which only the arm can
    resolve, so it is handed None and has to return the resolved dict; or the
    profile is ours-tuned, where the values are a choice the manifest makes,
    so the manifest's own flags dict is passed in and stamped. A cell can
    never be written without the verbatim dict of values in force.

    The frame is applied here and only here: the arm's call is wrapped in
    one jit carrying the resolved option dict, so every arm in a manifest
    compiles under exactly the same options and no arm can bring its own.
    """
    import jax
    name = entry["arm"]
    bind = load_arm(name)
    profile = config.PROFILES[entry["profile"]]
    given = profile["flags"]
    if given is None and isinstance(entry.get("flags"), dict):
        given = dict(entry["flags"])
    bound = bind(entry.get("tree"), dict(shape),
                 dict(given) if given is not None else None)
    for field in ("call", "contract", "source_hash", "tree_sha"):
        if not bound.get(field):
            raise harness.HarnessRefusal(f"arms/{name}.py bind returned no "
                                         f"{field!r}")
    if bound.get("operands") is not None and not callable(bound["operands"]):
        raise harness.HarnessRefusal(
            f"arms/{name}.py returned operands that are not callable")
    if bound.get("operands") and not bound.get("build_check"):
        raise harness.HarnessRefusal(
            f"arms/{name}.py takes operands and declares no build_check: the "
            f"default build path lowers only (x, gating), which is not the "
            f"program a cell of this arm measures")
    # THE CELL CARRIES EVERYTHING THAT SHAPED THE PROGRAM. The profile's
    # values and the arm's RESOLVED values are merged, not chosen between.
    # Replacing one with the other drops exactly the parameters that
    # parameterize a kernel: an arm's resolved capacity and weight format
    # would survive only in a child log, and a cell that omits them cannot
    # say what it measured. The profile wins where both name a key, and a key
    # both claim with DIFFERENT values is a disagreement about the
    # measurement rather than something to overwrite quietly, so it refuses.
    resolved = bound.get("flags")
    if given is None:
        if not isinstance(resolved, dict):
            raise harness.HarnessRefusal(
                f"profile {entry['profile']!r} carries no flag values of its "
                f"own, so arms/{name}.py has to return the resolved flags it "
                f"ran under; it returned {type(resolved).__name__}")
        flags = dict(resolved)
    else:
        flags = dict(resolved) if isinstance(resolved, dict) else {}
        for key, value in dict(given).items():
            if key in flags and flags[key] != value:
                raise harness.HarnessRefusal(
                    f"profile {entry['profile']!r} sets {key}={value!r} and "
                    f"arms/{name}.py resolved {key}={flags[key]!r}. One cell "
                    f"cannot claim both, and the profile silently winning is "
                    f"how a resolved parameter disappears from a stamp.")
            flags[key] = value
    bound = dict(bound)
    bound["flags"] = dict(flags)
    bound["config_origin"] = profile["origin"]
    bound["profile_name"] = entry["profile"]
    bound["tree"] = bound.get("tree") or entry.get("tree") or HERE
    bound["framed_call"] = jax.jit(bound["call"],
                                   compiler_options=compiler_options or None)
    return bound


def build_point(bound, shape, batch):
    """Trace and lower one point FOR THE DEVICE, without running it.

    A build check lowers the program the measured run will compile, and it
    does that on this host by naming the lowering platform. It does not
    lower a host-platform neighbour of the program: a Pallas kernel refuses
    to lower for the host platform at all, so a host lowering would prove
    nothing about the arms whose kernels are Pallas, and forcing interpret
    mode to get around that lowers a different program and then collides
    with the device platform the arm asked for.

    An arm that declares build_check owns this step and it is called exactly
    as it stands, with no wrapper: it knows its own operands. The default
    path is the same two-step the arms use,
    trace(...).lower(lowering_platforms=("tpu",)).

    NO COMPILER OPTIONS HERE, AND THAT IS DECLARED. The frame is an XLA
    compile-side setting and belongs to measurement, where the child's framed
    call applies it and every cell stamps it. Handing it to a lowering that
    names a platform is what produced "the platform for the specified backend
    cpu is different from the lowering platform tpu". So a built verdict means
    the program traces and lowers for the device UNFRAMED: it is evidence
    about shapes and kernels, not about the frame, and the report says so
    (build_map_note). The frame's own compile happens in the measured window.
    """
    import jax
    import jax.numpy as jnp
    check = bound.get("build_check")
    if check is not None:
        check(int(batch))
        return "the arm's own build check"
    x_spec, gating_spec = harness.case_specs(shape, batch)

    def spec(pair):
        return jax.ShapeDtypeStruct(pair[0], jnp.dtype(pair[1]))

    x = spec(x_spec)
    gating = {k: spec(v) for k, v in gating_spec.items()}
    jax.eval_shape(bound["call"], x, gating)
    jax.jit(bound["call"]).trace(x, gating).lower(
        lowering_platforms=("tpu",))
    return "traced and lowered for the device"


def cell_cases(facts, batch, routing, tier):
    """The cases one cell measures, and the stamps that name them.

    A replay cell measures the capture's own serving steps, one capture
    each. A drawn cell measures one case, captured as many times as the
    tier's repeats say.
    """
    if routing == "replay":
        by_count = facts.capture_records()
        harness.check_capture(by_count, facts.shape, [batch])
        records, steps = harness.select_replay_steps(
            by_count, batch, facts.seed, facts.shape)
        cases = [harness.replay_case(r, facts.shape, batch, facts.seed)
                 for r in records]
        # A replay cell's independent captures are its steps, so it repeats
        # nothing: four real serving calls say more about variance than four
        # captures of one of them.
        return cases, {"draw_seed": None, "replay_steps": steps, "repeats": 1}
    case = harness.draw_case(facts.shape, batch, facts.seed)
    repeats = int(config.TIERS[tier]["repeats"])
    return [case] * repeats, {
        "draw_seed": harness.draw_seed(facts.seed, "random", batch),
        "replay_steps": None, "repeats": repeats}


def materialize_arm(bound, args, key):
    """Run the arm's own call once, eagerly, before anything is traced.

    THE FAILURE THIS CLOSES. An arm builds its weights the first time its
    own call runs, and the driver wraps that call in one jit to apply the
    compiler frame. So the first traced call builds the weights INSIDE a
    trace, the arm caches the resulting tracers, and the next batch's trace
    trips over them:

        UnexpectedTracerError ... float8_e4m3fn[512,4096,2048] ... created
        at arms/_sources.py:415 (put), from production_pair.py:147
        (weights), traced for jit

    That takes every later cell of the configuration with it, and books them
    against the kernel. Running the arm's own call once outside any trace
    makes everything it caches a concrete array, and every later trace sees
    arrays instead of tracers.

    This runs no window and produces no number: it is the untimed step that
    puts the arm in the state its own design assumes. An assertion here is
    still a defect. Any other failure is recorded and left alone, because
    the window that follows will fail the same way with its own attribution.

    It warms the FRAMED call, the same program the window measures: with the
    operands concrete before any trace (the operand seam), warming the arm's
    bare call would only compile a second, unframed copy of the program.
    """
    import jax
    try:
        jax.block_until_ready(bound["framed_call"](*args))
        return "the framed call ran once, eagerly, before any trace"
    except AssertionError as exc:
        raise harness.Defect(
            f"materializing {key} hit an assertion: {exc}") from exc
    except harness.Defect:
        raise
    except Exception as exc:
        return (f"the arm's state was built and its call then stopped at "
                f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Trace Retention
# ---------------------------------------------------------------------------
# WHY A CAPTURE IS DROPPED ONCE IT HAS BEEN READ. One point's raw capture is
# 130 MB and up, and a session takes one per point, so a run directory can
# fill the filesystem it sits on while the session is still measuring. A full
# disk does not lose one point, it kills the session.
#
# Everything a cell reports is taken out of the capture at the moment it is
# captured, in the loop below: the whole-layer figure, the kernel figure, the
# window's program census, the executions the window held, the basis. Once
# that readout returns, nothing in the run reads the file again. So the
# default drops it and says so, and --keep-traces keeps every capture for a
# debugging session.
#
# THIS ONLY EVER TOUCHES A CAPTURE THIS RUN JUST READ, in this run's own
# directory. Captures kept by earlier sessions are evidence and their
# deletion is the researcher's, never a later run's.
TRACE_KEEP = (
    ("production_pair@ours-served/control-open/random/B64",
     "the anchor window the basis proofs re-derive from: the whole-layer and "
     "kernel figures of the frozen table are re-read out of this capture, so "
     "it outlives the session that took it"),
)


def window_name(key, role, routing, batch):
    """A measured window's name, the one the keep-list is written in."""
    return f"{key}/{role}/{routing}/B{batch}"


def trace_keep_reason(window):
    """Why this window's capture is kept, or None to drop it once read.

    An entry matches its own window and every window under it, so a keep-list
    line may name one window or a whole arm.
    """
    for named, reason in TRACE_KEEP:
        if window == named or window.startswith(named + "/"):
            return reason
    return None


def prune_trace(logdir, window):
    """Drop a capture this run has already read, leaving why behind.

    The step directory stays, holding the record: a reader who goes looking
    for the capture finds the reason it is gone and where its numbers went,
    rather than a path that is not there.
    """
    payload = os.path.join(logdir, "plugins")
    freed = 0
    for root, _dirs, files in os.walk(payload):
        for name in files:
            try:
                freed += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    shutil.rmtree(payload, ignore_errors=True)
    record = {"window": window, "kept": False, "freed_bytes": freed,
              "reason": "the capture was read at capture time and its "
                        "figures, program census and basis are in this "
                        "session's report; the raw file had no reader left. "
                        "--keep-traces keeps it."}
    try:
        with open(os.path.join(logdir, "retention.json"), "w") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
    except OSError:
        pass
    return record


def measure_point(facts, bound, point, key, save_output_to=None,
                  materialize=False):
    """One point measured in this child. Returns (measurement, ops, output).

    A measurement is data, not a cell: the child never writes the schema.
    Nothing is caught inside the window; what is caught is the whole point,
    once, so one point cannot take the table with it, and the reason is
    reported where the number would have been. An assertion is not caught at
    all: it leaves as a defect and the child fails the run.
    """
    role, routing = point["role"], point["routing"]
    batch, tier = int(point["batch"]), point["tier"]
    iters = int(config.TIERS[tier]["iters"])
    measurement = {"role": role, "routing": routing, "batch": batch,
                   "tier": tier, "iters": iters, "repeats": 1,
                   "draw_seed": None, "replay_steps": None,
                   "status": "failed", "program_us": None,
                   "per_step_us": None, "kernel_self_us": None,
                   "refusal": None, "refusal_attribution": None,
                   "materialized": None, "executions": None,
                   "expected_executions": None, "executions_per_call": None,
                   "coverage": None, "derivation": None,
                   "ops_per_call": None, "basis": None,
                   "wall_us": None, "per_step_wall_us": None,
                   "window_wall_us": None, "wall_minus_program_us": None,
                   "wall_over_program": None, "programs_in_window": None,
                   "program_census": None, "trace_retention": None}
    ops, output = None, None
    try:
        cases, stamps = cell_cases(facts, batch, routing, tier)
        measurement.update(stamps)
        prepare = bound.get("prepare") or harness.place
        # The arm's operands (its weights), built once per child, concrete
        # before any trace, and passed to every call as jit ARGUMENTS. Closed
        # over instead, they compile into the program as constants: 5.2 GB
        # executables the compile cache cannot store, and 505 us/call of
        # constant re-slicing at 64 tokens that serving never pays.
        extra = bound.get("operand_values")
        if extra is None:
            operands = bound.get("operands")
            extra = tuple(operands()) if operands else ()
            bound["operand_values"] = extra
        whole, kernel, walls, window_walls, step_basis = [], [], [], [], []
        for number, case in enumerate(cases):
            args = tuple(prepare(case)) + extra
            if materialize and number == 0:
                measurement["materialized"] = materialize_arm(bound, args, key)
                print(f"[child {key}] materialize: "
                      f"{measurement['materialized']}")
            logdir = os.path.join(facts.run_dir, "traces", key, role,
                                  routing, f"B{batch}", f"step{number}")
            window = harness.measure(bound["framed_call"], args, iters,
                                     int(config.WARMUP), logdir)
            walls.append(window["wall_us"])
            window_walls.append(window["window_wall_us"])
            # STAMPED THE MOMENT IT EXISTS, not once the readout succeeds. A
            # point whose capture produced nothing is exactly the point whose
            # wall figure is the only evidence left, so the physical bound
            # survives a failed readout rather than being lost with it.
            measurement["wall_us"] = sum(walls) / len(walls)
            measurement["per_step_wall_us"] = list(walls)
            measurement["window_wall_us"] = (sum(window_walls)
                                             / len(window_walls))
            totals = harness.program_totals(
                logdir, bound["contract"]["anchor"], iters,
                bound["contract"].get("executions_per_call"))
            whole.append(totals["whole_layer_us"])
            kernel.append(totals["kernel_self_us"])
            # The basis of the figure travels with it: how many program
            # executions the window held, how many of those one call
            # performs, and how many anchor operations run in one execution.
            measurement["executions"] = totals["executions"]
            measurement["expected_executions"] = totals["expected_executions"]
            measurement["executions_per_call"] = totals["executions_per_call"]
            measurement["coverage"] = totals["coverage"]
            measurement["derivation"] = totals["derivation"]
            measurement["ops_per_call"] = totals["anchor_ops_per_execution"]
            measurement["basis"] = totals["basis"]
            measurement["programs_in_window"] = totals["programs_in_window"]
            measurement["program_census"] = totals["program_census"]
            step_basis.append({"executions": totals["executions"],
                               "coverage": totals["coverage"],
                               "derivation": totals["derivation"]})
            measurement["per_step_basis"] = list(step_basis)
            if ops is None:
                ops = totals["ops"]
            # THE CAPTURE HAS NOW BEEN READ, and everything above came out of
            # it. Retention decides here, per window, and the decision travels
            # with the figure rather than being a property of the disk.
            window_id = window_name(key, role, routing, batch)
            keep_reason = trace_keep_reason(window_id)
            if facts.keep_traces:
                measurement["trace_retention"] = {
                    "window": window_id, "kept": True,
                    "reason": "--keep-traces: this session keeps every "
                              "capture"}
            elif keep_reason:
                measurement["trace_retention"] = {
                    "window": window_id, "kept": True, "reason": keep_reason}
            else:
                measurement["trace_retention"] = prune_trace(logdir, window_id)
            if save_output_to and output is None:
                import jax
                import numpy as np
                array = np.asarray(
                    jax.device_get(bound["framed_call"](*args)),
                    dtype=np.float32)
                np.save(save_output_to, array)
                output = save_output_to
        measurement["status"] = "ok"
        measurement["program_us"] = sum(whole) / len(whole)
        measurement["per_step_us"] = whole
        measurement["kernel_self_us"] = sum(kernel) / len(kernel)
        # The cell's scalar basis is the WEAKEST step, not the last one: a
        # four-step cell whose steps read 100/100/100/45 percent of their
        # windows says 45, and the per-step list carries the rest.
        measurement["executions"] = min(b["executions"] for b in step_basis)
        measurement["coverage"] = min(b["coverage"] for b in step_basis)
        measurement["derivation"] = (
            "full-window"
            if all(b["derivation"] == "full-window" for b in step_basis)
            else "self-normalized per execution")
        # THE PHYSICAL CROSS-CHECK ON WHAT THE TRACE MISSED. Host wall time
        # for one call, from the untraced loop, against the device figure the
        # capture produced. A healthy ratio sits just above one, 1.016 to
        # 1.034; a ratio that drifts upward is work the trace did not account
        # for, which is the one tell a program-level sum cannot give by
        # itself.
        measurement["wall_minus_program_us"] = (measurement["wall_us"]
                                                - measurement["program_us"])
        measurement["wall_over_program"] = (measurement["wall_us"]
                                            / measurement["program_us"]
                                            if measurement["program_us"]
                                            else None)
        # THE WALL ALARMS. Recording both clocks and never checking them is
        # the failure this closes: two figures exist and nothing refuses when
        # they disagree. Same-window ratio below one says the self-time sum
        # exceeded the elapsed wall of its own window (double-counted
        # concurrency, or a window that did not hold the work); a gap above
        # a fifth of the figure says the figure is not the cost of the call.
        same_window = (measurement["window_wall_us"]
                       / measurement["program_us"])
        if same_window < 0.98:
            raise harness.WindowError(
                f"the device self-time sum exceeds the wall of its own "
                f"window (ratio {same_window:.3f}): concurrent units "
                f"double-counted, or the window did not hold the work. The "
                f"figure is not a measurement.")
        gap = measurement["wall_us"] - measurement["program_us"]
        if gap > max(150.0, 0.20 * measurement["program_us"]):
            raise harness.WindowError(
                f"{gap:.0f} us of every call is outside the program "
                f"({measurement['wall_us']:.0f} us wall against "
                f"{measurement['program_us']:.0f} us program): too much of "
                f"the call is work the figure does not describe.")
    except AssertionError as exc:
        raise harness.Defect(
            f"measuring {key} at {batch} tokens hit an assertion: {exc}. An "
            f"assertion is a broken instrument, not a point the kernel "
            f"cannot do, so this run stops instead of booking a cell."
        ) from exc
    except harness.Defect:
        raise
    except Exception as exc:
        attribution = harness.attribution_for(exc)
        if attribution is None:
            raise harness.Defect(
                f"measuring {key} at {batch} tokens raised "
                f"{type(exc).__name__}: {exc}") from exc
        # A refusal attributed to the kernel or its own declared limits is a
        # RESULT and renders as one; "failed" is reserved for the harness's
        # incompletions, which render as gaps and never as a kernel's
        # capability. Booking a kernel's own compile refusal as failed
        # renders a pre-registered refusal as a hole in the table instead of
        # the result it is.
        measurement["status"] = ("refused" if attribution in
                                 ("kernel", "shape-constraint",
                                  "declared-out-of-scope") else "failed")
        measurement["refusal"] = f"{type(exc).__name__}: {exc}"
        measurement["refusal_attribution"] = attribution
        measurement["program_us"] = None
        measurement["per_step_us"] = None
        measurement["kernel_self_us"] = None
    return measurement, ops, output


def child_bind(spec, facts, result):
    """Bind this child's one configuration, or record why it could not."""
    if spec["mode"] == "injected":
        bound = dict(gates.broken_arm(spec.get("injection", "raises")))
        bound["profile_name"] = spec["profile"]
        bound["framed_call"] = bound["call"]
        result["bind"] = {"ok": True, "refusal": None, "attribution": None}
        result["bound"] = {
            "tree": bound["tree"], "tree_sha": bound["tree_sha"],
            "source_hash": bound["source_hash"], "sources": [], "pin": None,
            "namespace": None, "flags": bound["flags"],
            "contract": bound["contract"]}
        result["mirror"] = {"declared": False, "ok": None, "detail": None}
        return bound
    try:
        bound = bind_arm(spec["entry"], facts.shape,
                         facts.compiler_options)
    except AssertionError as exc:
        raise harness.Defect(f"binding {spec['key']} hit an assertion: {exc}"
                             ) from exc
    except Exception as exc:
        attribution = harness.attribution_for(exc)
        if attribution is None:
            raise harness.Defect(f"binding {spec['key']} raised "
                                 f"{type(exc).__name__}: {exc}") from exc
        result["bind"] = {"ok": False,
                          "refusal": f"{type(exc).__name__}: {exc}",
                          "attribution": attribution}
        return None
    result["bind"] = {"ok": True, "refusal": None, "attribution": None}
    # Where the arm's top-level module actually came from is a fact only this
    # process can report: it is the one that did the import. The source gate
    # compares it against the tree the arm stamps.
    namespace = bound.get("namespace")
    module = sys.modules.get(namespace) if namespace else None
    # AND IT IS ASSERTED HERE, BEFORE ANYTHING IS BUILT. A package that came
    # from anywhere but the bound tree means this child measured code no pin
    # describes, and that has to stop the run rather than become a cell,
    # where the import error reads as the kernel's refusal. A defect cannot
    # be misattributed, which is the point.
    if namespace:
        tree = os.path.realpath(bound["tree"])
        loaded = getattr(module, "__file__", None)
        real = os.path.realpath(loaded) if loaded else None
        if real is None or not real.startswith(tree + os.sep):
            raise harness.Defect(
                f"{spec['key']} bound tree {tree} and its declared package "
                f"{namespace!r} is loaded from {real}, which is outside it. "
                f"Something on this child's import path answered for the tree "
                f"(a checkout on sys.path, or an editable install). No cell "
                f"can describe this measurement, so the run stops.")
    result["bound"] = {
        "tree": bound["tree"], "tree_sha": bound["tree_sha"],
        "source_hash": bound["source_hash"],
        "sources": list(bound.get("sources") or ()),
        "pin": bound.get("pin"), "namespace": namespace,
        "namespace_file": getattr(module, "__file__", None),
        "flags": bound["flags"], "contract": bound["contract"]}
    # The mirror hook can only run where the tree is bound. Its verdict
    # crosses the boundary and the mirror gate scores it in the parent.
    hook = bound.get("verify_mirror")
    if hook is None:
        result["mirror"] = {"declared": False, "ok": None, "detail": None}
    else:
        try:
            verdict = hook()
            if isinstance(verdict, tuple) and len(verdict) == 2:
                ok, mirror_detail = verdict
            else:
                ok, mirror_detail = bool(verdict), None
            result["mirror"] = {"declared": True, "ok": bool(ok),
                                "detail": mirror_detail}
        except AssertionError as exc:
            raise harness.Defect(f"the mirror check of {spec['key']} hit an "
                                 f"assertion: {exc}") from exc
        except Exception as exc:
            result["mirror"] = {"declared": True, "ok": False,
                                "detail": f"{type(exc).__name__}: {exc}"}
    return bound


def child_build(spec, facts, bound, result):
    """Build-check this child's own batches, in this child's own process."""
    for batch in spec.get("build_batches") or ():
        batch = int(batch)
        try:
            detail = build_point(bound, facts.shape, batch)
            result["build"][str(batch)] = {"status": "built", "detail": detail}
        except AssertionError as exc:
            raise harness.Defect(
                f"building {spec['key']} at {batch} tokens hit an assertion: "
                f"{exc}. An assertion is a broken instrument, not a refusal, "
                f"and this run stops here.") from exc
        except Exception as exc:
            attribution = harness.attribution_for(exc)
            if attribution is None:
                raise harness.Defect(
                    f"building {spec['key']} at {batch} tokens raised "
                    f"{type(exc).__name__}: {exc}") from exc
            # ATTRIBUTION IN A DEVICELESS BUILD. A typed refusal keeps its own
            # attribution: an arm that says the shape is outside what it
            # accepts is stating its own limit and it holds on the device too.
            # Anything else raised while building on a host platform is the
            # harness's, because a deviceless build is this kit's
            # approximation of the device, and an untyped failure inside an
            # approximation says more about the approximation than about the
            # kernel.
            if facts.deviceless and not isinstance(exc, harness.Refusal):
                attribution = "harness"
            result["build"][str(batch)] = {
                "status": "refused",
                "refusal": f"{type(exc).__name__}: {exc}",
                "attribution": attribution}
        print(f"[child {spec['key']}] build B{batch} "
              f"{result['build'][str(batch)]['status']}")


def child_measure(spec, facts, bound, result):
    """Measure this child's own points, in order.

    The first point materializes the arm's own lazily built state, once, so
    nothing weight-derived is ever created inside a trace.

    THE FRAME IS CHECKED HERE TOO, in the child, after every tree import and
    before the first compile. The parent sets both halves before spawning,
    but a tree can mutate the environment while it is imported: the serving
    tree's env_override prepends to LIBTPU_INIT_ARGS and appends to
    XLA_FLAGS at import time. Those prepends land AFTER backend
    initialization in this child, so they cannot change what compiles here,
    which is exactly why a duplicate of a token the frame already names is
    accepted as a no-op and anything else refuses.
    """
    before = (result.get("isolation")
              or {}).get("frame_environment_pre_import") or {}
    after = gates.frame_environment()
    problems, injected = gates.frame_injection(before, after,
                                               facts.frame_options)
    result["frame_environment"] = after
    result["frame_environment_injected_by_imports"] = injected
    result["frame_problems"] = problems
    if injected:
        print(f"[child {spec['key']}] the code under measurement changed its "
              f"own compile environment while importing: {sorted(injected)}")
    if problems:
        raise harness.HarnessRefusal(
            "an import changed a flag this frame declares: "
            + "; ".join(problems))
    numerics = spec.get("numerics")
    first = True
    for point in spec.get("points") or ():
        batch = int(point["batch"])
        if (result["build"].get(str(batch)) or {}).get("status") == "refused":
            continue
        save_to = None
        if (numerics and point["role"] == "measure"
                and batch == int(numerics["batch"])
                and point["routing"] == numerics["routing"]
                and not result.get("output")):
            os.makedirs(spec["outputs_dir"], exist_ok=True)
            save_to = os.path.join(spec["outputs_dir"], f"{spec['key']}.npy")
        measurement, ops, output = measure_point(
            facts, bound, point, spec["key"], save_output_to=save_to,
            materialize=first)
        first = False
        result["measurements"].append(measurement)
        if output:
            result["output"] = output
        if ops is not None and result.get("ops") is None:
            result["ops"] = ops
        # Every measured batch's operation rows, so the window census can run
        # per batch: the program restructures across the sweep (62 to 201
        # operations between 64 and 8192 tokens), and a contract verified at
        # one batch says nothing about the others.
        if ops is not None and point["role"] == "measure":
            result.setdefault("ops_by_point", {})[str(batch)] = ops
        # A failed point says WHY on the line that reports it: a log reading
        # "failed None" sends a reader to the result file for something the
        # measurement already knew.
        if measurement["status"] == "ok":
            outcome = (f"{measurement['program_us']:.2f} us program, "
                       f"{measurement['wall_us']:.2f} us wall, "
                       f"ratio {measurement['wall_over_program']:.3f}")
        else:
            clause = str(measurement["refusal"]).splitlines()[0][:110]
            outcome = f"{measurement['refusal_attribution']}: {clause}"
        print(f"[child {spec['key']}] {point['role']} {point['routing']} "
              f"B{batch} {measurement['status']} {outcome}")


def child_main(spec_path):
    """One configuration, one tree, one process.

    Everything this child knows comes from the spec the parent wrote. It
    reports data and never writes a cell: the schema authority is the
    parent's, in one place, so a child cannot invent a fingerprint.
    """
    with open(spec_path) as fh:
        spec = json.load(fh)
    portable = spec["facts"]
    # Before any arm module is imported: nothing but this child's own kit and
    # the tree the arm binds may answer for the arm's package.
    isolation = child_isolate(portable["run_dir"],
                              tree_roots=spec.get("tree_roots") or ())
    print(f"[child {spec['key']}] import path: dropped "
          f"{isolation['removed_paths']}, finders "
          f"{isolation['removed_finders']}, purged "
          f"{isolation['purged_modules']}")
    print(f"[child {spec['key']}] quarantine: "
          f"{isolation.get('declared_variables')} declared, scrubbed "
          f"{sorted(isolation.get('quarantine') or {})}")
    # THE SHELL'S CONTRIBUTION IS HELD TO THE FRAME, here, before any import.
    manifest_frame = dict(config.FRAMES[read_manifest(spec["manifest"])
                                        ["frame"]])
    shell_problems, _ = gates.frame_conflicts(
        manifest_frame,
        allow_flags=((DEVICELESS_ALLOWED_FLAG,)
                     if portable["deviceless"] else ()))
    if shell_problems:
        raise harness.HarnessRefusal(
            "the environment this child was launched with does not match the "
            "frame the cells will claim: " + "; ".join(shell_problems))
    if portable["deviceless"]:
        pin_mosaic_probe()
    manifest = read_manifest(spec["manifest"])
    facts = Facts(manifest, portable["run_dir"], portable["deviceless"],
                  device=portable["device"], env_hash=portable["env_hash"],
                  session_id=portable["session_id"],
                  keep_traces=portable.get("keep_traces", False))
    result = {"key": spec["key"], "defect": None, "device": None,
              "bind": None, "bound": None, "mirror": None, "build": {},
              "measurements": [], "ops": None, "output": None,
              "isolation": isolation, "resolved_frame": None,
              "frame_environment": None, "frame_problems": None,
              "post_bind_quarantine": None}
    code = 0
    # What frame THIS child resolved, for the parent to compare against its
    # own before it stamps a cell: two processes reading one config file is
    # still two reads, and a cell may not claim a frame its measuring process
    # did not use.
    result["resolved_frame"] = dict(facts.frame_options)
    try:
        if not facts.deviceless:
            result["device"] = gates.device_facts()
        bound = child_bind(spec, facts, result)
        if bound is not None:
            # The bound tree is known now, and so is anything the arm declares
            # it reads (env_reads). Quarantine again over both: an arm that
            # binds a tree the spec could not name, or that reads variables of
            # its own like a kernel's TPU_MOE_ family, is covered here even
            # though its registry was not visible earlier.
            declared, registries = env_registry_names([bound.get("tree")])
            declared |= set(bound.get("env_reads") or ())
            inherited = set(isolation.get("inherited_names") or ())
            # An inherited value that only became visible now, because only
            # the bound tree's registry or the arm's own declaration names it,
            # is scrubbed here. A variable the ARM set while binding is left
            # alone and recorded: that is the arm's declared behaviour, in its
            # own source, not a value this session dragged in from a shell.
            present = {n for n in declared if n in os.environ}
            arm_set = {n: os.environ[n] for n in sorted(present - inherited)}
            result["post_bind_quarantine"] = {
                "registries": registries,
                "declared_variables": len(declared),
                "arm_declared_env_reads": sorted(
                    bound.get("env_reads") or ()),
                "scrubbed": quarantine_env(present & inherited),
                "set_by_the_arm_while_binding": arm_set,
                "watchlist": env_watchlist_snapshot(declared)}
            if arm_set:
                print(f"[child {spec['key']}] the arm set these itself while "
                      f"binding, left as its own declared behaviour: "
                      f"{sorted(arm_set)}")
            child_build(spec, facts, bound, result)
            child_measure(spec, facts, bound, result)
            close = bound.get("close")
            if close is not None:
                close()
    except harness.Defect as exc:
        result["defect"] = str(exc)
        code = DEFECT_EXIT
        print(f"[child {spec['key']}] DEFECT: {exc}")
    with open(spec["result"], "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True, default=str)
    return code


def probe_main(out_path):
    """The device facts, from a process that is allowed to have the device."""
    try:
        facts = gates.device_facts()
    except Exception as exc:
        facts = {"error": f"{type(exc).__name__}: {exc}"}
    with open(out_path, "w") as fh:
        json.dump(facts, fh, indent=1, sort_keys=True)
    print(f"[probe] {facts}")
    return 0


# ---------------------------------------------------------------------------
# The Parent: Orchestration, Gates, And The Store
# ---------------------------------------------------------------------------
def spawn(argv, log_path, timeout=None):
    """Run one child and keep its whole log. Returns its exit code.

    A child's output is written where the run can be read afterwards, and
    its last lines are printed, so a failure is visible without hunting for
    a file.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(f"[spawn] {' '.join(argv[1:])}")
    sys.stdout.flush()
    try:
        done = subprocess.run([sys.executable] + argv, capture_output=True,
                              text=True, timeout=timeout)
        output = (done.stdout or "") + (done.stderr or "")
        code = done.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        output = stdout + f"\n[spawn] TIMED OUT after {timeout}s\n"
        code = DEFECT_EXIT
    with open(log_path, "w") as fh:
        fh.write(output)
    for line in [x for x in output.splitlines()
                 if x.startswith(("[child", "[probe", "Traceback"))][-6:]:
        print(f"  | {line}")
    return code


def run_child(facts, run_dir, name, spec, timeout=None):
    """Write a child's spec, run it, read its result. Defects propagate."""
    children = os.path.join(run_dir, "children")
    os.makedirs(children, exist_ok=True)
    spec = dict(spec)
    spec["result"] = os.path.join(children, f"{name}.result.json")
    spec["outputs_dir"] = os.path.join(run_dir, "outputs")
    spec["facts"] = facts.portable()
    spec_path = os.path.join(children, f"{name}.spec.json")
    with open(spec_path, "w") as fh:
        json.dump(spec, fh, indent=1, sort_keys=True, default=str)
    log_path = os.path.join(run_dir, "logs", f"{name}.log")
    code = spawn([os.path.join(HERE, "run.py"), "--child", spec_path],
                 log_path, timeout)
    if not os.path.exists(spec["result"]):
        raise harness.Defect(
            f"the child for {spec['key']} exited {code} without writing a "
            f"result. Its log is {log_path}")
    with open(spec["result"]) as fh:
        result = json.load(fh)
    if result.get("defect"):
        raise harness.Defect(f"{spec['key']}: {result['defect']}")
    if code == DEFECT_EXIT:
        raise harness.Defect(f"the child for {spec['key']} exited on a defect "
                             f"without naming it; its log is {log_path}")
    return result


def bound_view(key, result):
    """A child's bind report, in the shape the source and mirror gates read.

    The mirror hook ran in the child. Here it is a thunk over the verdict
    the child reported, so the gate scores one verdict in one place and no
    hook is ever called twice.
    """
    bound = dict(result.get("bound") or {})
    mirror = result.get("mirror") or {}
    if mirror.get("declared"):
        bound["verify_mirror"] = (
            lambda m=mirror: (bool(m.get("ok")), m.get("detail")))
    profile = key.split("@", 1)[1] if "@" in key else None
    bound["config_origin"] = (config.PROFILES.get(profile) or {}).get("origin")
    return bound


def base_fields(facts, arm, profile, bound, batch, routing, tier, role):
    """The fingerprint every cell of this configuration carries.

    The role is part of the fingerprint, so the opening and the closing
    reading of the control are two identities rather than one address
    written twice.
    """
    return {
        "arm": arm, "shape_id": facts.shape_id, "batch": int(batch),
        "routing": routing, "tree": bound.get("tree") or "",
        "tree_sha": bound.get("tree_sha") or "",
        "source_hash": bound.get("source_hash") or "",
        "profile": profile,
        "config_origin": config.PROFILES[profile]["origin"],
        "flags": dict(bound.get("flags") or {}), "frame": facts.frame,
        "frame_options": dict(facts.frame_options),
        "draw_seed": None, "replay_steps": None,
        "iters": int(config.TIERS[tier]["iters"]), "repeats": 1,
        "warmup": int(config.WARMUP), "tier": tier, "role": role,
        "session_id": facts.session_id, "session_drift_us": None,
        "env_hash": facts.env_hash, "device": facts.device or "",
        "input_hashes": None,
        "status": "failed", "program_us": None, "per_step_us": None,
        "kernel_self_us": None, "refusal": None,
        "refusal_attribution": None,
        "wall_us": None, "window_wall_us": None, "wall_over_program": None,
        "coverage": None, "derivation": None, "executions": None,
        "per_step_basis": None, "kit_source_hash": kit_source_hash(),
    }


def cell_from(facts, arm, profile, bound, measurement):
    """One measurement a child reported, stamped and validated here."""
    fields = base_fields(facts, arm, profile, bound, measurement["batch"],
                         measurement["routing"], measurement["tier"],
                         measurement["role"])
    for field in ("draw_seed", "replay_steps", "repeats", "iters", "status",
                  "program_us", "per_step_us", "kernel_self_us", "refusal",
                  "refusal_attribution", "wall_us", "window_wall_us",
                  "wall_over_program", "coverage", "derivation", "executions",
                  "per_step_basis"):
        fields[field] = measurement.get(field)
    if measurement["routing"] == "replay":
        fields["input_hashes"] = dict(facts.input_hashes or {})
    return cells.make_cell(**fields)


def refusal_cells(facts, arm, profile, batch, entry, bound, refusal,
                  attribution, roles=("measure",)):
    """A point a child refused, as one cell per routing and role.

    A refusal is data. The table has a row there, it says what the arm said
    and whose limit it was, and nobody has to remember that a gap in a table
    was a refusal rather than a run that was never launched.

    A refused replay cell ran no steps, and cells.py will not accept an
    empty replay_steps, so it carries the sentinel pair NO_STEPS. It is a
    sentinel and not a capture: a refused cell that named real steps would
    be claiming to have replayed calls it never touched.
    """
    tier = entry.get("tier", facts.manifest["tier"])
    out = []
    for role in roles:
        for routing in entry.get("routings") or ["random"]:
            fields = base_fields(facts, arm, profile, bound, batch, routing,
                                 tier, role)
            fields.update({
                "draw_seed": (None if routing == "replay" else
                              harness.draw_seed(facts.seed, "random", batch)),
                "replay_steps": (NO_STEPS if routing == "replay" else None),
                "input_hashes": (dict(facts.input_hashes or {})
                                 if routing == "replay" else None),
                "status": "refused", "refusal": refusal,
                "refusal_attribution": attribution})
            out.append(cells.make_cell(**fields))
    return out


def quarantine_roots(entry):
    """Where a child looks for the registries it has to quarantine, BEFORE it
    binds anything.

    A tree can only be parsed if it can be named, and at spawn time the names
    available are the manifest's tree for this arm, the kit's own checkout, and
    the fetched trees under arms/_sources, which is where the pinned arms bind.
    An arm that resolves some other tree is covered after bind, by the second
    quarantine, which is early enough because a tree's registry resolves every
    variable lazily.
    """
    roots = []
    if entry.get("tree"):
        roots.append(entry["tree"])
    origin = origin_kit()
    roots.append(origin)
    fetched = os.path.join(origin, "arms", "_sources")
    if os.path.isdir(fetched):
        for name in sorted(os.listdir(fetched)):
            path = os.path.join(fetched, name)
            if os.path.isdir(path):
                roots.append(path)
    return roots


def late_inputs(facts, plan):
    """Every file this plan's children will open late, for the inputs gate.

    That is the recorded capture, needed by every replay point, plus
    anything a manifest lists under "inputs". They are gathered here, in the
    parent, and proven before a child exists, because a file a child opens
    at cell forty is a file the session bet on at minute zero.
    """
    required = []
    replay = [p for p in plan if p["routing"] == "replay"]
    if replay:
        required.append({
            "label": f"recorded capture for {facts.shape_id}",
            "path": facts.capture_path(),
            "needed_by": [f"{p['key']}:B{p['batch']}" for p in replay],
            "min_records": len(replay),
            "batches": sorted({int(p["batch"]) for p in replay})})
    for extra in facts.manifest.get("inputs") or ():
        path = extra.get("path")
        if path and not os.path.isabs(path):
            path = os.path.join(origin_kit(), path)
        required.append({"label": extra.get("label") or extra.get("path"),
                         "path": path,
                         "needed_by": [extra.get("needed_by") or "the run"],
                         "min_records": extra.get("min_records") or 1})
    return required


def capture_verifier(facts, batches):
    """A verify hook for the inputs gate: the capture must serve the plan.

    It parses the recording once, at session open, and asks
    harness.check_capture the same questions a replay cell would ask at
    measurement time: right expert width, totals that match the selection
    width, no expert holding more rows than there are tokens, and enough
    non-degenerate serving calls at every count the plan replays. What a
    child would discover per point is discovered here, once.
    """
    def verify(path):
        by_count = harness.load_capture(path)
        harness.check_capture(by_count, facts.shape, batches)
        serving = {}
        for batch in batches:
            records = by_count.get(int(batch)) or []
            keep = [r for r in records
                    if not harness.is_degenerate(r, facts.shape["top_k"])]
            serving[str(batch)] = len(keep)
            if len(keep) < int(config.REPLAY_STEPS_PER_CELL):
                raise harness.CaptureError(
                    f"{batch} tokens has {len(keep)} serving calls past the "
                    f"degeneracy rule and a replay cell measures "
                    f"{config.REPLAY_STEPS_PER_CELL}")
        return {"records": sum(len(v) for v in by_count.values()),
                "token_counts": sorted(by_count),
                "serving_calls_per_batch": serving}
    return verify


def quarantine_summary(result):
    """What one child's two quarantine passes did, for a cell's basis."""
    pre = result.get("isolation") or {}
    post = result.get("post_bind_quarantine") or {}
    scrubbed = dict(pre.get("quarantine") or {})
    scrubbed.update(post.get("scrubbed") or {})
    return {
        "declared_variables": max(int(pre.get("declared_variables") or 0),
                                  int(post.get("declared_variables") or 0)),
        "registries": [r.get("registry") for r in
                       (pre.get("registries") or [])
                       + (post.get("registries") or [])],
        "arm_declared_env_reads": post.get("arm_declared_env_reads") or [],
        "scrubbed": scrubbed,
        "watchlist": post.get("watchlist") or pre.get("watchlist") or {},
    }


def child_order(plan):
    """The children this run spawns, in session order.

    The opening control first, then one child per measuring configuration in
    the order the manifest lists them, then the closing control. Each entry
    is (child name, registry key, role).
    """
    order = []
    for role in ("control-open", "measure", "control-close"):
        for key in dict.fromkeys(p["key"] for p in plan
                                 if p["role"] == role):
            order.append((f"{key}.{role}", key, role))
    return order


def run(manifest_path, run_dir, deviceless, only=None, batches=None,
        child_timeout=None, keep_traces=False):
    """The sequence in this file's docstring, in that order."""
    manifest = read_manifest(manifest_path)
    shape = config.SHAPES[manifest["shape_id"]]
    plan = resolve_plan(manifest, only=only, batches=batches)
    frame = dict(config.FRAMES[manifest["frame"]])
    cache = isolate_compile_cache(manifest["frame"], frame)
    # The frame's runtime half, set once here so every child inherits it.
    libtpu = apply_libtpu_args(frame)
    if deviceless:
        deviceless_env(shape["ep"])
    os.makedirs(run_dir, exist_ok=True)
    report = {"manifest": manifest_path, "run_dir": run_dir,
              "trace_retention": ("every capture kept (--keep-traces)"
                                  if keep_traces else
                                  "each capture dropped once its figures, "
                                  "census and basis were read; kept windows: "
                                  + ", ".join(n for n, _ in TRACE_KEEP)),
              "started": datetime.datetime.now().isoformat(timespec="seconds"),
              "deviceless": bool(deviceless),
              "selection": {"only": sorted(only or ()),
                            "batches": sorted(int(b) for b in
                                              (batches or ()))},
              "frame": manifest["frame"], "frame_options": frame,
              "frame_compiler_options": gates.frame_compiler_options(frame),
              "frame_libtpu_init_args": gates.frame_libtpu_args(frame),
              "libtpu_init_args_applied": libtpu,
              "compile_cache": cache, "plan": plan, "gates": {},
              "children": {}, "basis": {}}
    written = []
    refused_gates = []

    def gate(name, result):
        ok, detail = result
        report["gates"][name] = {"ok": bool(ok), "detail": detail}
        print(f"[gate] {name}: {'pass' if ok else 'REFUSE'}")
        if not ok:
            refused_gates.append(name)
        return ok

    # 3. The device, from a child, because the parent must not hold it.
    device_info = None
    if not deviceless:
        probe_path = os.path.join(run_dir, "children", "probe.result.json")
        os.makedirs(os.path.dirname(probe_path), exist_ok=True)
        spawn([os.path.join(HERE, "run.py"), "--probe", probe_path],
              os.path.join(run_dir, "logs", "probe.log"), child_timeout)
        if os.path.exists(probe_path):
            with open(probe_path) as fh:
                device_info = json.load(fh)
        else:
            device_info = {"error": "the probe child wrote no result"}
    facts = Facts(manifest, run_dir, deviceless,
                  device=("host platform, no device (deviceless build map)"
                          if deviceless
                          else (device_info or {}).get("fingerprint")),
                  keep_traces=keep_traces)
    facts.numerics = numerics_plan(manifest, plan)
    report["session_id"] = facts.session_id
    report["device"] = facts.device
    report["numerics_plan"] = facts.numerics
    # A comparison that cannot happen says so out loud. The manifest names a
    # count and a routing for the numerics comparison; if fewer than two
    # measure points in the resolved plan run there, no comparison exists and
    # the gate would simply not appear, which reads as one that passed.
    if facts.numerics:
        comparable = [p for p in plan if p["role"] == "measure"
                      and int(p["batch"]) == facts.numerics["batch"]
                      and p["routing"] == facts.numerics["routing"]]
        if len(comparable) < 2:
            report["numerics_note"] = (
                f"the numerics comparison names {facts.numerics['batch']} "
                f"tokens at {facts.numerics['routing']} routing, and this "
                f"plan runs {len(comparable)} measure point(s) there, so no "
                f"comparison exists. Name a count and routing the arms run.")
            print(f"[numerics] not comparable in this plan: "
                  f"{report['numerics_note']}")
            facts.numerics = None

    gate("environment", gates.environment(
        env_lock_path=os.path.join(HERE, config.ENV_LOCK_FILE),
        require_device=not deviceless, frame=frame,
        allow_flags=((DEVICELESS_ALLOWED_FLAG,) if deviceless else ()),
        device_info=device_info))

    # Every file the children will open late, proven before one exists.
    required = late_inputs(facts, plan)
    replay_batches = sorted({int(p["batch"]) for p in plan
                             if p["routing"] == "replay"})
    inputs_ok, inputs_detail = gates.manifest_inputs(
        required, verify=(capture_verifier(facts, replay_batches)
                          if replay_batches else None))
    gate("manifest_inputs", (inputs_ok, inputs_detail))
    facts.input_hashes = gates.input_hashes(inputs_detail)
    report["input_hashes"] = facts.input_hashes

    # A gate that failed here stops the session BEFORE any child spawns:
    # otherwise a missing input is discovered one point at a time, with a
    # child per configuration already paid for.
    if refused_gates:
        report["refused_gates"] = refused_gates
        print(f"[run] refused at session open, no child spawned: "
              f"{refused_gates}")
        return finish(facts, report, written, refused_gates, code=2)

    # 4. One child per configuration, in session order.
    entries = plan_entries(manifest, plan)
    arms, results, results_by_key, build = {}, {}, {}, {}
    opening, closing = {}, {}
    for name, key, role in child_order(plan):
        entry = entries[key]
        points = [p for p in plan if p["key"] == key and p["role"] == role]
        here = sorted({int(p["batch"]) for p in points})
        spec = {"mode": "configuration", "key": key, "arm": entry["arm"],
                "profile": entry["profile"], "entry": entry,
                "manifest": manifest_path, "build_batches": here,
                "tree_roots": quarantine_roots(entry),
                "points": ([] if deviceless else
                           [{"role": p["role"], "routing": p["routing"],
                             "batch": int(p["batch"]), "tier": p["tier"]}
                            for p in points]),
                "numerics": facts.numerics}
        result = run_child(facts, run_dir, name, spec, child_timeout)
        results[name] = result
        results_by_key.setdefault(key, result)
        report["children"][name] = {
            "bind": result["bind"], "mirror": result["mirror"],
            "build": result["build"],
            "measured": len(result["measurements"]),
            "device": (result.get("device") or {}).get("fingerprint")}
        if key not in arms:
            arms[key] = bound_view(key, result)
        # G5: the frame the child resolved has to be the frame the cells will
        # claim. Two processes reading one config file is still two reads.
        child_frame = result.get("resolved_frame")
        if child_frame is not None and child_frame != frame:
            raise harness.Defect(
                f"the child for {key} resolved frame {child_frame} and this "
                f"session stamps {frame}: a cell may not claim a frame its "
                f"measuring process did not use")
        # The device has to be the same device for the whole session.
        seen = (result.get("device") or {}).get("fingerprint")
        if seen and facts.device and seen != facts.device:
            raise harness.Defect(
                f"the child for {key} saw device {seen!r} and this session is "
                f"stamped {facts.device!r}: the device changed under the run")
        for batch_text, outcome in sorted(result["build"].items()):
            build[(entry["arm"], entry["profile"], int(batch_text))] = outcome
        if not result["bind"]["ok"]:
            for batch in here:
                build[(entry["arm"], entry["profile"], batch)] = {
                    "status": "refused",
                    "refusal": result["bind"]["refusal"],
                    "attribution": result["bind"]["attribution"]}
        for measurement in result["measurements"]:
            cell = cell_from(facts, entry["arm"], entry["profile"], arms[key],
                             measurement)
            written.append(cell)
            # WHERE THE FIGURE'S BASIS IS KEPT. The cell schema has no field
            # for the window's execution count, so the basis is recorded in
            # the report against the cell's own content address. Give the
            # schema a field and this moves onto the cell.
            report["basis"][cell["cell_id"]] = {
                "executions": measurement.get("executions"),
                "expected_executions": measurement.get("expected_executions"),
                "executions_per_call": measurement.get("executions_per_call"),
                "coverage": measurement.get("coverage"),
                "derivation": measurement.get("derivation"),
                "anchor_ops_per_call": measurement.get("ops_per_call"),
                "programs_in_window": measurement.get("programs_in_window"),
                "program_census": measurement.get("program_census"),
                "trace_retention": measurement.get("trace_retention"),
                "wall_us": measurement.get("wall_us"),
                "window_wall_us": measurement.get("window_wall_us"),
                "wall_minus_program_us":
                    measurement.get("wall_minus_program_us"),
                "wall_over_program": measurement.get("wall_over_program"),
                "materialized": measurement.get("materialized"),
                "reading": measurement.get("basis"),
                # A capture is an input pin: the recording a replay cell was
                # measured against is named by its own hash, so two cells read
                # from two recordings can never be read as one.
                "input_hashes": (dict(facts.input_hashes)
                                 if cell["routing"] == "replay" else None),
                # What the measuring child's environment held, and what was
                # taken out of it: a cell's flags are what shaped the program
                # only if nothing else could.
                "quarantine": quarantine_summary(result),
                "frame_environment": result.get("frame_environment")}
            if measurement["role"] == "control-open":
                opening[int(measurement["batch"])] = cell
            elif measurement["role"] == "control-close":
                closing[int(measurement["batch"])] = cell
        if role == "measure":
            by_point = result.get("ops_by_point") or (
                {"first": result["ops"]} if result.get("ops") else {})
            for batch_text, point_ops in sorted(by_point.items()):
                gate(f"window_census:{key}:B{batch_text}",
                     gates.window_census(
                         key, (result["bound"] or {}).get("contract") or {},
                         point_ops))

    # 5. The gates, over what the children reported.
    #
    # A configuration that did not bind has no source to hash and no mirror to
    # check, and its refusal is already a cell with an attribution. Feeding it
    # to these gates would fail the whole run over a refusal the run already
    # recorded, which is the opposite of treating a refusal as data. It is
    # named here and in the report instead.
    not_bound = {
        key: ((results_by_key.get(key) or {}).get("bind") or {}).get("refusal")
        for key in arms
        if not ((results_by_key.get(key) or {}).get("bind") or {}).get("ok")}
    for key, refusal in sorted(not_bound.items()):
        print(f"[bind] {key} did not bind, so the source and mirror gates "
              f"have nothing to check: {str(refusal)[:120]}")
    report["not_bound"] = not_bound
    checkable = {k: v for k, v in arms.items() if k not in not_bound and v}
    gate("source_hash", gates.source_hash(
        checkable, pins_path=os.path.join(HERE, config.PINS_FILE),
        our_tree=HERE))
    gate("selector_mirror", gates.selector_mirror(checkable))
    gate("build_map", gates.build_map(
        plan, build, declared_points(manifest),
        manifest.get("expected_refusals") or ()))
    gate("build_map_scope_control", gates.prove_plan_scope())
    report["build_map"] = {gates.show(k): v for k, v in build.items()}
    report["build_map_note"] = (
        "built and refused verdicts are UNFRAMED lowerings for the device: "
        "the build map applies no compiler options and no runtime flags, so a "
        "verdict speaks about shapes and kernels rather than about the frame. "
        "Accepted as declared; the frame's compile happens in the measured "
        "window.")
    prediction = report["gates"]["build_map"]["detail"].get("prediction") or {}
    for line in prediction.get("as_predicted") or ():
        print(f"[prediction] as pre-registered: {line}")
    for line in prediction.get("misses") or ():
        print(f"[prediction] MISS: {line}")
    for line in prediction.get("not_predicted") or ():
        print(f"[prediction] refusal that was not pre-registered: {line}")

    # A refused point becomes cells for every role and routing it would have
    # run, so a refusal is never an empty square anywhere.
    roles_by_key = {}
    for point in plan:
        roles_by_key.setdefault(point["key"], [])
        if point["role"] not in roles_by_key[point["key"]]:
            roles_by_key[point["key"]].append(point["role"])
    for (arm, profile, batch), outcome in sorted(build.items()):
        if outcome["status"] != "refused":
            continue
        key = registry_key(arm, profile)
        refused_cells = refusal_cells(
            facts, arm, profile, batch, entries[key], arms.get(key) or {},
            outcome["refusal"], outcome.get("attribution") or "harness",
            roles=tuple(roles_by_key.get(key) or ("measure",)))
        # A REFUSED CELL GETS A BASIS TOO. What environment the refusal
        # happened in is part of what it means, and a reader comparing a
        # refusal against a later success needs both sides described.
        child = results_by_key.get(key) or {}
        for cell in refused_cells:
            report["basis"][cell["cell_id"]] = {
                "quarantine": quarantine_summary(child),
                "frame_environment": child.get("frame_environment"),
                "frame_environment_injected_by_imports":
                    child.get("frame_environment_injected_by_imports"),
                "input_hashes": (dict(facts.input_hashes)
                                 if cell["routing"] == "replay" else None)}
        written.extend(refused_cells)

    # 6. The deliberate-breakage control, through a child and the parent's own
    # cell path, which is the path every real cell takes.
    smallest = min(int(p["batch"]) for p in plan)

    def run_injected(bound):
        injection = bound.get("injection", "raises")
        spec = {"mode": "injected", "key": "deliberate_breakage",
                "arm": "deliberate_breakage", "profile": "upstream-default",
                "entry": {"arm": "deliberate_breakage",
                          "profile": "upstream-default",
                          "routings": ["random"], "batches": [smallest],
                          "tier": "screen"},
                "manifest": manifest_path, "build_batches": [],
                "tree_roots": [], "injection": injection,
                "points": [{"role": "measure", "routing": "random",
                            "batch": smallest, "tier": "screen"}],
                "numerics": None}
        result = run_child(facts, run_dir, f"breakage-{injection}", spec,
                           child_timeout)
        if not result["measurements"]:
            return None
        return cell_from(
            facts, "deliberate_breakage", "upstream-default",
            bound_view("deliberate_breakage@upstream-default", result),
            result["measurements"][0])

    gate("deliberate_breakage", gates.deliberate_breakage(run_injected))

    if refused_gates:
        report["refused_gates"] = refused_gates
        return finish(facts, report, written, refused_gates, code=2)
    if deviceless:
        unbound = sorted(report.get("not_bound", ()))
        report["note"] = ("deviceless phase: the gates that need no device "
                          "ran, every configuration built its own points in "
                          "its own child, and the refusals they found are the "
                          "only cells")
        if unbound:
            report["note"] += (". Arms that never bound: " + ", ".join(unbound)
                               + ". A configuration that cannot bind proves "
                               "nothing, so this run exits nonzero.")
            return finish(facts, report, written, refused_gates, code=2)
        return finish(facts, report, written, refused_gates, code=0)

    # 7. Controls, then the drift onto every cell.
    control_ok, control_detail = gates.controls(opening, closing)
    gate("controls", (control_ok, control_detail))
    drift = gates.drift_map(control_detail)
    for cell in written:
        cell["session_drift_us"] = drift.get(int(cell["batch"]))
    report["batches_without_drift"] = sorted(
        {int(c["batch"]) for c in written if c["session_drift_us"] is None})

    # 8. Numerics, from the arrays the children saved.
    if facts.numerics:
        import numpy as np
        outputs = {}
        for result in results.values():
            if result.get("output") and os.path.exists(result["output"]):
                outputs[result["key"]] = np.load(result["output"])
        report["numerics_outputs"] = sorted(outputs)
        reference_key = facts.numerics["reference_key"]
        reference = outputs.get(reference_key)
        for key in sorted(outputs):
            if key == reference_key:
                continue
            gate(f"numerics:{key}", gates.numerics(
                key, outputs[key], reference, reference_key))

    failed = [c for c in written if c["status"] == "failed"]
    code = 2 if refused_gates else (3 if failed else 0)
    return finish(facts, report, written, refused_gates, code)


def finish(facts, report, written, refused_gates, code):
    """Write the cells and the report, then say what happened.

    Every cell is revalidated here, after the drift backfill, so the cell_id
    each one carries is the hash of the fingerprint that was actually
    stored. Nothing downstream identifies a cell any other way.

    A DEVICELESS PASS NEVER TOUCHES THE MANIFEST'S STORE. Its cells are
    refusals found while building on a host, and the store is the record of
    device sessions: without this rule a pre-flight writes host-platform
    refusals into the store and they have to be deleted by hand. A deviceless
    pass's cells stay in its own run directory, where they are a diagnostic,
    and the store is written by device sessions alone.
    """
    # The drift backfill changed a fingerprint field after the basis index
    # was keyed, so every re-derived cell_id is re-mapped here: an orphaned
    # basis is coverage, walls and quarantine records nobody can reach.
    remap, revalidated = {}, []
    for c in written:
        old_id = dict(c).get("cell_id")
        cell = cells.make_cell(**dict(c))
        if old_id and old_id != cell["cell_id"]:
            remap[old_id] = cell["cell_id"]
        revalidated.append(cell)
    written = revalidated
    if remap and isinstance(report.get("basis"), dict):
        report["basis"] = {remap.get(k, k): v
                           for k, v in report["basis"].items()}
    store = facts.manifest["cells"]
    if not os.path.isabs(store):
        store = os.path.join(origin_kit(), store)
    run_copy = os.path.join(facts.run_dir, os.path.basename(store))
    if facts.deviceless:
        report["cells_store"] = run_copy
        report["store_note"] = (f"deviceless: the manifest's store {store} "
                                f"was not written; these cells live in the "
                                f"run directory only")
        store = None
    elif code == 2:
        # A session that failed a GATE leaves its cells in its own run
        # directory: the shared store holds gate-clean sessions, because
        # nothing downstream reads gate verdicts before reading cells. A
        # session with refused or harness-failed CELLS and clean gates is
        # stored: those cells are results, not session defects.
        report["cells_store"] = run_copy
        report["store_note"] = (f"exit {code}: the manifest's store {store} "
                                f"was not written; these cells live in the "
                                f"run directory only")
        store = None
    else:
        os.makedirs(os.path.dirname(store) or ".", exist_ok=True)
        cells.write_cells(store, written)
        report["cells_store"] = store
    cells.write_cells(run_copy, written)
    report["cells_written"] = len(written)
    report["cells_run_copy"] = run_copy
    report["cell_ids"] = [c["cell_id"] for c in written]
    report["exit_code"] = code
    report["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(facts.run_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True, default=str)
    print(f"[run] {len(written)} cells to {store or run_copy}"
          f"{' (run directory only)' if store is None else ''}; "
          f"gates refused: {refused_gates or 'none'}; exit {code}")
    return code


# ---------------------------------------------------------------------------
# The Controls On The Controls
# ---------------------------------------------------------------------------
def prove_controls():
    """Every negative control in this kit, run deviceless and in process.

    A gate that cannot fail proves nothing, so each control is exercised
    both ways: against a path that behaves and against one that does not.
    This is the gate-level proof and it needs no manifest; a real run
    exercises the same controls through the child boundary. A gate whose
    negative controls did not fire is refused.
    """
    problems = []

    # The breakage control, against a path that records the injected failure
    # and refuses the assertion, which is what measure_point does.
    def catching_path(bound):
        try:
            bound["call"](None, None)
        except AssertionError as exc:
            raise harness.Defect(f"assertion in the injected arm: {exc}"
                                 ) from exc
        except Exception as exc:
            return {"status": "failed",
                    "refusal": f"{type(exc).__name__}: {exc}",
                    "refusal_attribution": harness.attribution_for(exc),
                    "program_us": None, "per_step_us": None,
                    "kernel_self_us": None}
        raise AssertionError("the injected arm did not fail at all")

    # A path that swallows both, which is the defect the control exists for.
    def swallowing_path(bound):
        try:
            bound["call"](None, None)
        except BaseException:
            pass
        return {"status": "ok", "refusal": None, "refusal_attribution": None,
                "program_us": 123.0, "per_step_us": [123.0],
                "kernel_self_us": 100.0}

    caught_ok, caught = gates.deliberate_breakage(catching_path)
    swallowed_ok, swallowed = gates.deliberate_breakage(swallowing_path)
    print(f"[proof] breakage control, catching path: ok={caught_ok}")
    print(f"[proof]   attribution recorded: "
          f"{caught.get('cell_attribution')!r}; assertion raised: "
          f"{caught.get('assertion_raised')!r}")
    print(f"[proof] breakage control, swallowing path: ok={swallowed_ok}")
    for problem in swallowed.get("problems") or []:
        print(f"[proof]   fired: {problem}")
    if not caught_ok:
        problems.append(f"the breakage gate refuses a correct path: "
                        f"{caught.get('problems')}")
    if swallowed_ok:
        problems.append("the breakage gate passes a path that turned a failed "
                        "cell into a number")

    scope_ok, scope = gates.prove_plan_scope()
    print(f"[proof] plan-scope control: ok={scope_ok} "
          f"outside={scope.get('outside_manifest')}")
    if not scope_ok:
        problems.append(f"the plan-scope control did not hold: "
                        f"{scope.get('problems')}")

    # RETENTION, BOTH WAYS. Dropping a capture is destructive, so the control
    # that matters is the one proving it never touches a window the proofs
    # re-derive from. Exercised on a stand-in capture, not a real one.
    import tempfile
    with tempfile.TemporaryDirectory() as scratch:
        def stand_in(name):
            payload = os.path.join(scratch, name, "plugins", "profile", "p")
            os.makedirs(payload, exist_ok=True)
            with open(os.path.join(payload, "host.xplane.pb"), "wb") as fh:
                fh.write(b"x" * 4096)
            return os.path.join(scratch, name)

        anchor = TRACE_KEEP[0][0]
        kept_reason = trace_keep_reason(anchor)
        ordinary = window_name("some_arm@some-profile", "measure", "replay",
                               8192)
        drop_reason = trace_keep_reason(ordinary)
        print(f"[proof] retention keep-list: the anchor window is kept "
              f"({bool(kept_reason)}), an ordinary window is not "
              f"({drop_reason is None})")
        if not kept_reason:
            problems.append(f"the keep-list does not protect {anchor}, the "
                            f"window the basis proofs re-derive from")
        if drop_reason is not None:
            problems.append("the keep-list protects every window, so the "
                            "retention control cannot free anything")
        dropped = prune_trace(stand_in("ordinary"), ordinary)
        left = os.path.exists(os.path.join(scratch, "ordinary", "plugins"))
        record = os.path.exists(os.path.join(scratch, "ordinary",
                                            "retention.json"))
        print(f"[proof] retention drop: freed {dropped['freed_bytes']} bytes, "
              f"capture gone={not left}, reason recorded={record}")
        if left or not record or dropped["freed_bytes"] != 4096:
            problems.append("dropping a read capture did not remove it and "
                            "leave the reason in its place")
        # And the keep path leaves the file alone: the control fails if a
        # kept window is ever handed to the pruner.
        keeper = stand_in("anchor")
        if trace_keep_reason(anchor) is None:
            prune_trace(keeper, anchor)
        if not os.path.exists(os.path.join(keeper, "plugins", "profile", "p",
                                           "host.xplane.pb")):
            problems.append("a kept window's capture was removed anyway")
        print("[proof] retention keep: the anchor window's capture is "
              "untouched")

    for problem in problems:
        print(f"[proof] FAILED: {problem}")
    if problems:
        return 4
    print("[proof] every control fires when it should and passes when it "
          "should")
    return 0


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="run.py", description="The one driver of the MoE benchoff.")
    parser.add_argument("--manifest", help="path to the run's manifest json")
    parser.add_argument("--run-dir", default=None,
                        help="where the snapshot, children, traces and report "
                             "land; results/<run id> by default")
    parser.add_argument("--deviceless", action="store_true",
                        help="the pre-flight pass: every gate that needs no "
                             "device, and every configuration builds its own "
                             "points in its own child. No cell is measured.")
    parser.add_argument("--only", default=None,
                        help="comma-separated arms to run, a subset of the "
                             "manifest's. Narrows the resolved plan, which is "
                             "what every gate sees.")
    parser.add_argument("--batches", default=None,
                        help="comma-separated token counts to run, a subset "
                             "of the manifest's")
    parser.add_argument("--child-timeout", type=float, default=None,
                        help="seconds before a child is treated as hung. No "
                             "limit by default: a heavy tier's window is as "
                             "long as it is, and a number invented here would "
                             "kill honest runs.")
    parser.add_argument("--child", default=None,
                        help="internal: run one configuration from the spec "
                             "file named here")
    parser.add_argument("--probe", default=None,
                        help="internal: write the device facts to the path "
                             "named here")
    parser.add_argument("--keep-traces", action="store_true",
                        help="keep every raw capture. By default a capture is "
                             "dropped once its figures, program census and "
                             "basis have been read out of it, because one "
                             "point's capture is 130 MB and up and a full "
                             "disk kills the session, not the point. Windows "
                             "the proofs re-derive from are kept either way.")
    parser.add_argument("--prove-controls", action="store_true",
                        help="run every negative control in the kit and exit "
                             "non-zero if any of them cannot fire")
    args = parser.parse_args(argv)
    if args.child:
        return child_main(args.child)
    if args.probe:
        return probe_main(args.probe)
    if args.prove_controls:
        return prove_controls()
    if not args.manifest:
        parser.error("--manifest is required")
    only = [s.strip() for s in (args.only or "").split(",") if s.strip()]
    batches = [int(s) for s in (args.batches or "").split(",") if s.strip()]
    manifest_path = os.path.abspath(args.manifest)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.abspath(args.run_dir or os.path.join(
        results_root(), run_id))
    snapshot_and_reexec(argv, manifest_path, run_dir)
    try:
        return run(manifest_path, run_dir, args.deviceless, only, batches,
                   args.child_timeout, keep_traces=args.keep_traces)
    except harness.Defect as exc:
        print(f"[run] DEFECT: {exc}")
        return DEFECT_EXIT
    except harness.Refusal as exc:
        print(f"[run] REFUSE: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
