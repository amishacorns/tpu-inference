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

"""The gates: ten checks, each traceable to a failure that happened.

Every gate in this file is a function returning (ok, detail). ok is a bool
and nothing else; detail is a json-serializable dict naming what was
compared, what was found, and, when a gate fails, both sides of the
disagreement. No gate prints a verdict and no gate exits: run.py decides
what a failure stops.

The ten, and where each one lives:

  1. environment        here, environment(): versions, device, and the
                        frame isolation check, because the environment is
                        where a compiler frame silently grows
  2. manifest inputs    here, manifest_inputs(): every file the manifest
                        names, present, readable, hashed and able to serve
                        the plan, BEFORE any child spawns. Without it a run
                        dies one replay point at a time on a capture that
                        is not at the path the run resolves
  3. source hash        here, source_hash()
  4. selector mirror    here, selector_mirror()
  5. build map          here, build_map(), over the RESOLVED plan, with
                        prove_plan_scope() as its negative control
  6. deliberate breakage here, deliberate_breakage(), two injections: a
                        failure that must become a cell, and an assertion
                        that must not
  7. controls           here, controls()
  8. window census      here, window_census()
  9. numerics           here, numerics()
 10. re-derivation     tools_rederive.py, not here. Re-deriving the
                        document from the cells has to run without the
                        measurement kit's cooperation, and without a
                        device, so it reads the stores and nothing else.

IMPORT SAFE BEFORE JAX, for the same reason harness.py is: run.py imports
this module while it is still deciding the compiler frame and the
deviceless flags, and both are read at `import jax`.
"""

import gzip
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import subprocess
import sys

import config

GATES = (
    ("environment", "gates.environment"),
    ("manifest inputs", "gates.manifest_inputs"),
    ("source hash", "gates.source_hash"),
    ("selector mirror", "gates.selector_mirror"),
    ("build map", "gates.build_map"),
    ("deliberate breakage", "gates.deliberate_breakage"),
    ("controls", "gates.controls"),
    ("window census", "gates.window_census"),
    ("numerics", "gates.numerics"),
    ("re-derivation", "tools_rederive.py"),
)

# The packages env.lock has to pin and this gate has to check: env.lock pins
# jax, jaxlib, libtpu and numpy, and the kit asserts them and the device
# before any cell runs.
PINNED_PACKAGES = ("jax", "jaxlib", "libtpu", "numpy")

# Channels through which the environment reaches the compiler. An xla option
# arriving this way extends or overrides the frame a cell claims to have
# compiled under, which is why the frame is content rather than a name: a name
# cannot show that two arms compiled the same way. Each name, and why it is a
# channel:
FRAME_HAZARD_ENV = {
    "XLA_FLAGS": "xla options read at backend initialization and applied to "
                 "every compile in the process",
    "LIBTPU_INIT_ARGS": "runtime initialization flags, several of which are "
                        "xla_tpu compile options. This is the frame's second "
                        "half, so its whole value is compared against what "
                        "the frame declares, token for token",
    "JAX_DISABLE_JIT": "removes compilation, so no frame applies to anything",
}

# Variables that change how a program is built or where it runs and that no
# frame declares. Any value at all refuses the run: unlike the two channels
# above there is no legitimate content for them here, so there is nothing to
# compare and nothing to allow.
FRAME_FORBIDDEN_ENV = {
    "JAX_DEFAULT_MATMUL_PRECISION": "changes the precision every matmul "
                                    "compiles at, which is the measurement",
    "JAX_NUMPY_DTYPE_PROMOTION": "changes what dtype an operation produces, "
                                 "so two arms could compute in different "
                                 "widths under one frame name",
    "JAX_ENABLE_X64": "doubles the width of every default dtype",
    "JAX_NUMPY_RANK_PROMOTION": "changes broadcasting, so a program can be "
                                "built differently without any code changing",
    "JAX_DEFAULT_DEVICE": "moves work off the device the session asserted",
    "TPU_LIBRARY_PATH": "selects another libtpu than the one env.lock pins, "
                        "which is the runtime the whole frame is defined "
                        "against",
}


def frame_libtpu_args(frame):
    """The runtime half of a frame, as a list of tokens."""
    return list((frame or {}).get("libtpu_init_args") or ())


def frame_compiler_options(frame):
    """The jit half of a frame, as the option dict jax is given."""
    return dict((frame or {}).get("compiler_options") or {})

# The relative-L2 figure the numerics gate reports, and the named bar it is
# read against. The bar is a REPORT threshold and never gates a cell: arms
# use different arithmetic contracts and different accumulation orders, so
# they differ in the last bits by construction, and a run stopped on that
# would be a run stopped on arithmetic that is working.
#
# The bar is the named bf16-class tolerance, 2e-3.
REPORT_TOLERANCE_NAME = "bf16-class"
REPORT_TOLERANCE = 2e-3

# What the deliberate-breakage control raises, so the gate can recognize its
# own injection rather than crediting an unrelated failure.
BREAKAGE_MARK = "deliberate breakage control: this cell is designed to fail"


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------
def file_sha256(path):
    """The sha256 of the file's content. A gzipped file hashes as its
    uncompressed stream, so a capture's hash is stable across compression."""
    if str(path).endswith(".gz"):
        h = hashlib.sha256()
        with gzip.open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    """sha256 of one file's bytes, hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest(paths, root):
    """The one source-hash rule, used by the arms and re-derived by the gate.

    The digest is the sha256 of one sha256sum-format line per file, sorted
    by the file's path relative to root:

        <sha256><two spaces><relative path><newline>

    The format is the pin file's own
    (pins.json in this directory, which carries a sha256 for every fetched file),
    so a reviewer can rebuild the digest with sha256sum and sort and get the
    same string without running any of this code.
    """
    lines = sorted(f"{file_sha256(p)}  {os.path.relpath(p, root)}\n"
                   for p in paths)
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def rel_l2(a, b):
    """Relative L2 of a candidate against a reference: ||a-b|| / ||b||.

    One definition of this comparator, used everywhere a candidate array is
    checked against a reference array.
    """
    import numpy as np
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _normalize_package(name):
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def read_env_lock(path):
    """The pinned versions in an env.lock, as {package: version}.

    The file is pip freeze output. Lines that are comments, blank, or pip
    options are ignored; a line that is not a plain `name==version` pin is
    reported as unpinned rather than parsed loosely.
    """
    pinned, loose = {}, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            if "==" in line:
                name, _, version = line.partition("==")
                pinned[_normalize_package(name)] = version.strip()
            else:
                loose.append(line)
    return pinned, loose


def installed_versions(packages):
    out = {}
    for package in packages:
        try:
            out[_normalize_package(package)] = metadata.version(package)
        except metadata.PackageNotFoundError:
            out[_normalize_package(package)] = None
    return out


def pip_freeze():
    """This interpreter's pip freeze, verbatim, as one string."""
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pip freeze failed: {result.stderr.strip()}")
    return result.stdout


def env_hash():
    """sha256 of this interpreter's pip freeze, stamped on every cell."""
    return hashlib.sha256(pip_freeze().encode()).hexdigest()


def _kind_family(kind):
    """A device kind reduced to its generation family, for comparison.

    The runtime and the configuration spell the same chip two ways: captured
    device kinds carry "TPU7x" while config.DEVICE_REQUIRED says "TPU v7".
    Comparison therefore drops spaces and the "v" some surfaces write
    before the generation number, and asks whether the required family is a
    prefix of the reported one, so "TPU v7" accepts "TPU7x" and refuses
    "TPU6e".
    """
    return re.sub(r"v(?=\d)", "", str(kind).lower().replace(" ", ""))


def device_facts():
    """What the backend reports, as data. Touches the device.

    Called in a process that is allowed to initialize the backend, which is
    a measuring child and never the orchestrating parent.
    """
    import jax
    devices = jax.devices()
    kinds = sorted({str(d.device_kind) for d in devices})
    return {"kinds": kinds, "chips": len(devices),
            "hosts": jax.process_count(),
            "fingerprint": (f"{'+'.join(kinds)} chips={len(devices)} "
                            f"hosts={jax.process_count()} "
                            f"jax={jax.__version__}")}


def device_fingerprint():
    """The device string stamped on every cell. Touches the device."""
    return device_facts()["fingerprint"]


def frame_fingerprint(frame_options):
    """A short hash of a resolved option dict, for cache isolation and logs.

    Two frames that differ in one option get different digests, which is
    what the compile cache is keyed on so a shared cache can never serve a
    binary compiled under other options.
    """
    payload = json.dumps(frame_options or {}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def frame_environment(names=None):
    """The compile-relevant environment, by name, as it stands right now.

    Snapshotted before a tree is imported and again after, so what the
    launching shell contributed can be told apart from what the code under
    measurement did to its own process.
    """
    watched = set(names or ()) | set(FRAME_HAZARD_ENV) | set(
        FRAME_FORBIDDEN_ENV)
    return {name: os.environ[name] for name in sorted(watched)
            if name in os.environ}


def frame_injection(before, after, frame):
    """What appeared in the compile environment while code was imported.

    Returns (problems, injected). A token that arrives during an import is
    the imported code acting on its own process: the serving tree's
    env_override prepends to LIBTPU_INIT_ARGS and appends an ISA flag to
    XLA_FLAGS, and jax's own plugin sets TPU_LIBRARY_PATH and a launch-barrier
    token. Those are recorded, not refused, because they are properties of the
    code being measured rather than of the shell that launched it, and they
    land after backend initialization here in any case.

    What IS refused: an injected token that gives a DIFFERENT value to a flag
    the frame declares. That is the frame's own content being changed under
    its name, which no record can make honest.
    """
    problems, injected = [], {}
    declared = {}
    for token in frame_libtpu_args(frame):
        name, _, value = token.partition("=")
        declared[name] = value
    for name in sorted(set(before) | set(after)):
        was, now = before.get(name), after.get(name)
        if was == now:
            continue
        injected[name] = {"before": was, "after": now}
        if name != "LIBTPU_INIT_ARGS":
            continue
        for token in (now or "").split():
            if token in frame_libtpu_args(frame):
                continue
            flag, _, value = token.partition("=")
            if flag in declared and value != declared[flag]:
                problems.append(
                    f"an import set {flag}={value!r} in LIBTPU_INIT_ARGS and "
                    f"this frame declares {flag}={declared[flag]!r}. The "
                    f"frame's own content cannot be changed under its name.")
    return problems, injected


def frame_conflicts(frame, allow_flags=()):
    """Environment settings that would extend or override the frame.

    Returns (problems, observed). The frame a cell stamps has to be the whole
    truth about how it compiled, on BOTH its halves.

    LIBTPU_INIT_ARGS is compared whole. It is the frame's runtime half, so
    what it may hold is exactly what the frame declares: every token is
    checked, not only the ones that begin with --xla, because a prefix test
    lets everything else ride through. A duplicate of a token the frame
    already names is accepted as the no-op it is, since the serving tree's
    env_override prepends one of them at import time without
    deduplicating, and a missing token is a problem too: a run whose runtime
    half was cleared by something is not running the frame it claims.

    XLA_FLAGS holds no part of any frame here: the jit half reaches the
    compiler as options on the call, so an xla option in that variable is
    contamination. allow_flags names the prefixes a phase legitimately sets,
    which is only ever the deviceless host-platform device count.

    Everything in FRAME_FORBIDDEN_ENV refuses on any value at all.
    """
    problems, observed = [], {}
    allow = tuple(allow_flags)
    declared = frame_libtpu_args(frame)
    for name in sorted(FRAME_FORBIDDEN_ENV):
        if name in os.environ:
            observed[name] = os.environ[name]
            problems.append(
                f"{name}={os.environ[name]!r} is set, and no frame declares "
                f"it: {FRAME_FORBIDDEN_ENV[name]}. Unset it and measure "
                f"again.")
    for name in sorted(os.environ):
        if name not in FRAME_HAZARD_ENV and not name.startswith("XLA_"):
            continue
        value = os.environ[name]
        observed[name] = value
        if name == "JAX_DISABLE_JIT":
            if value.strip().lower() not in ("", "0", "false"):
                problems.append(f"JAX_DISABLE_JIT={value!r} removes "
                                f"compilation, so the frame this run stamps "
                                f"would describe nothing")
            continue
        if name == "LIBTPU_INIT_ARGS":
            present = value.split()
            extra = sorted(set(present) - set(declared))
            missing = sorted(set(declared) - set(present))
            duplicates = sorted({t for t in present
                                 if present.count(t) > 1})
            observed["LIBTPU_INIT_ARGS_duplicates"] = duplicates
            if extra:
                problems.append(
                    f"LIBTPU_INIT_ARGS carries {extra}, which this frame does "
                    f"not declare. The frame's runtime half is "
                    f"{declared or 'empty'}, and a token outside it changes "
                    f"the device compile under a frame name that does not "
                    f"mention it.")
            if missing:
                problems.append(
                    f"LIBTPU_INIT_ARGS is missing {missing}, which this frame "
                    f"declares. Something cleared or replaced the runtime "
                    f"half after this run set it.")
            continue
        offending = [t for t in value.split()
                     if t.startswith("--xla") and not t.startswith(allow)]
        if offending:
            problems.append(
                f"{name} carries compiler options {offending}, which extend "
                f"or override the frame this run compiles under "
                f"({sorted(frame_compiler_options(frame))}). "
                f"{FRAME_HAZARD_ENV.get(name, 'reaches the compiler')}.")
    if declared and "LIBTPU_INIT_ARGS" not in os.environ:
        problems.append(
            f"this frame declares a runtime half ({declared}) and "
            f"LIBTPU_INIT_ARGS is not set at all, so the flags the serving "
            f"deployment initializes libtpu with are absent from this run.")
    return problems, observed


# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
def environment(env_lock_path=None, require_device=True, frame=None,
                allow_flags=(), device_info=None):
    """The interpreter, the device and the compiler frame are the pinned ones.

    env.lock is asserted, not advisory. Every package in PINNED_PACKAGES has
    to be present in it and installed at exactly the pinned version. A
    missing env.lock fails the gate, because "the versions were whatever
    was installed" is not a fingerprint.

    The frame is checked here too, because the environment is where a frame
    silently grows or silently loses a half: the jit options must not be
    extended through XLA_FLAGS, and LIBTPU_INIT_ARGS must hold exactly the
    runtime flags the frame declares, no more and no fewer.

    require_device is False only for the deviceless phase, where the build
    map runs on the host platform and there is no device to fingerprint.
    allow_flags names the flags that phase legitimately sets.

    device_info is how the device facts arrive when this gate runs in a
    process that must not touch the device. The driver measures in child
    processes, one per source tree, and a parent that initialized the
    backend to fingerprint it would hold the chips its own children need.
    So a child reports {"kinds": [...], "chips": n, "hosts": n,
    "fingerprint": str} or {"error": str}, and the comparison against
    config.DEVICE_REQUIRED still happens here, in one place. Passing None
    with require_device True means this process does the fingerprinting
    itself.
    """
    path = env_lock_path or config.ENV_LOCK_FILE
    detail = {"env_lock": path, "checked_packages": list(PINNED_PACKAGES),
              "require_device": bool(require_device),
              "frame": dict(frame or {}),
              "frame_compiler_options": frame_compiler_options(frame),
              "frame_libtpu_init_args": frame_libtpu_args(frame),
              "frame_fingerprint": frame_fingerprint(frame)}
    frame_problems, observed = frame_conflicts(frame, allow_flags)
    detail["frame_environment"] = observed
    if not os.path.exists(path):
        detail["problems"] = frame_problems + [
            f"{path} does not exist, so no version pin was asserted. "
            f"Generate it with pip freeze from the venv that runs the "
            f"session."]
        return False, detail
    pinned, loose = read_env_lock(path)
    detail["loose_lines"] = loose
    installed = installed_versions(PINNED_PACKAGES)
    detail["installed"] = installed
    detail["pinned"] = {k: pinned.get(k) for k in
                        (_normalize_package(p) for p in PINNED_PACKAGES)}
    problems = list(frame_problems)
    for package in PINNED_PACKAGES:
        key = _normalize_package(package)
        want, got = pinned.get(key), installed.get(key)
        if want is None:
            problems.append(f"{package} is not pinned in {path}")
        elif got is None:
            problems.append(f"{package} is pinned at {want} but is not "
                            f"installed in {sys.executable}")
        elif got != want:
            problems.append(f"{package} is pinned at {want} but {got} is "
                            f"installed")
    if require_device:
        facts = device_info
        if facts is None:
            try:
                facts = device_facts()
            except Exception as exc:
                facts = {"error": f"{type(exc).__name__}: {exc}"}
        detail["device"] = facts
        detail["device_fingerprint"] = facts.get("fingerprint")
        if facts.get("error"):
            problems.append(f"the device could not be fingerprinted: "
                            f"{facts['error']}")
        else:
            kinds = list(facts.get("kinds") or ())
            want = config.DEVICE_REQUIRED
            family = _kind_family(want["kind"])
            if not kinds or not all(_kind_family(k).startswith(family)
                                    for k in kinds):
                problems.append(f"the run requires {want['kind']!r} but the "
                                f"backend reports {kinds}")
            if int(facts.get("chips", 0)) != int(want["chips"]):
                problems.append(f"the run requires {want['chips']} chips but "
                                f"{facts.get('chips')} are visible")
            if int(facts.get("hosts", 0)) != int(want["hosts"]):
                problems.append(f"the run requires {want['hosts']} host(s) "
                                f"but the backend reports "
                                f"{facts.get('hosts')}")
    detail["problems"] = problems
    return not problems, detail


# ---------------------------------------------------------------------------
# 2. Manifest Inputs
# ---------------------------------------------------------------------------
def manifest_inputs(required, verify=None):
    """Every file the manifest names is present and sound, at session open.

    THE FAILURE THIS CLOSES. Without this gate a session can open, pass every
    other check, spawn its children, and then fail at every replay point one
    at a time because the recorded capture is not at the path the run
    resolves. The attribution still holds and the cells are still honest, and
    the whole session is still lost: an input that children read LATE has to
    be proven at open, before any child exists.

    A CAPTURE IS AN INPUT PIN. It is hashed here and the hash is reported, so
    a cell measured against one recording can never be read as a cell
    measured against another, exactly as a source file's hash pins an arm.

    required is a list of dicts, each naming one file:
        label      what it is, in words
        path       the resolved absolute path children will open
        needed_by  the plan points that will read it
        min_records optional floor on what verify reports

    verify is optional and receives the path. It returns a dict of facts (a
    record count under "records", anything else it wants to record) or
    raises. Its raising is a refusal of the session: a capture that cannot
    serve the plan is not better for being present.
    """
    detail = {"inputs": {}, "problems": []}
    for item in required or ():
        label = item.get("label") or item.get("path")
        entry = {"path": item.get("path"),
                 "needed_by": list(item.get("needed_by") or ())}
        path = item.get("path")
        if not path:
            detail["problems"].append(f"{label} names no path at all")
            detail["inputs"][label] = entry
            continue
        if not os.path.exists(path):
            entry["exists"] = False
            detail["problems"].append(
                f"{label} is not at {path}, and "
                f"{len(entry['needed_by'])} point(s) of this plan will read "
                f"it. Put the file there or point the manifest at the file.")
            detail["inputs"][label] = entry
            continue
        entry["exists"] = True
        entry["bytes"] = os.path.getsize(path)
        try:
            with open(path, "rb") as fh:
                fh.read(1)
        except OSError as exc:
            entry["readable"] = False
            detail["problems"].append(f"{label} at {path} cannot be read: "
                                      f"{exc}")
            detail["inputs"][label] = entry
            continue
        entry["readable"] = True
        entry["sha256"] = file_sha256(path)
        if verify is not None:
            try:
                facts = verify(path) or {}
                entry.update(facts)
            except Exception as exc:
                entry["verified"] = False
                detail["problems"].append(
                    f"{label} at {path} cannot serve this plan: "
                    f"{type(exc).__name__}: {exc}")
                detail["inputs"][label] = entry
                continue
            entry["verified"] = True
            floor = int(item.get("min_records") or 1)
            records = entry.get("records")
            if records is None:
                detail["problems"].append(
                    f"{label} was verified without reporting a record count, "
                    f"so nothing shows it holds anything")
            elif int(records) < floor:
                detail["problems"].append(
                    f"{label} holds {records} record(s), under the {floor} "
                    f"this plan needs")
        detail["inputs"][label] = entry
    return not detail["problems"], detail


def input_hashes(detail):
    """{label: sha256} from a manifest_inputs detail, for stamping."""
    return {label: entry.get("sha256")
            for label, entry in (detail.get("inputs") or {}).items()
            if entry.get("sha256")}


# ---------------------------------------------------------------------------
# The Canonical Serving Settings, Parsed
# ---------------------------------------------------------------------------
# The serving-path comparison has to build the layer under the settings the
# deployment actually serves with. Those settings live in one authority as a
# tuple of tokens, and they are PARSED out of it, never retyped here: a
# restated serve setting is a second source that can drift from the first, and
# the whole point of the comparison is that it does not.
#
# The authority is named by the manifest as an input (label and path), so the
# manifest-inputs gate hashes it at session open and every cell that depends
# on it carries that hash. When the authority is not in a checkout, the parse
# says so and the comparison it feeds refuses rather than assuming values.
SERVE_TOKENS_SYMBOL = "CANONICAL_SERVE_TOKENS"


def serve_tokens(path, symbol=SERVE_TOKENS_SYMBOL):
    """The canonical serve tokens, parsed from their own authority.

    Returns a list of records, one per token, each carrying what the token is
    and where it came from:

        kind        environment | argument | additional-config
        name        the variable or argument name
        value       the value as written, verbatim
        token       the token verbatim
        line        the line in the source the token sits on

    An environment token is NAME=value. An argument token starts with a dash
    and is what the serve command line carries. The one JSON blob is the
    additional-config argument, kept whole.

    Nothing is imported: the authority is read as text and walked with ast, so
    parsing it cannot execute it and cannot depend on its imports.
    """
    import ast
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"the canonical serving settings are not at {path}, so no serving "
            f"comparison can be built from them")
    source = open(path).read()
    tree = ast.parse(source)
    node = None
    for candidate in ast.walk(tree):
        targets = []
        if isinstance(candidate, ast.Assign):
            targets = [getattr(t, "id", None) for t in candidate.targets]
        elif isinstance(candidate, ast.AnnAssign):
            targets = [getattr(candidate.target, "id", None)]
        if symbol in targets:
            node = candidate.value
            break
    if node is None or not isinstance(node, (ast.Tuple, ast.List)):
        raise ValueError(
            f"{path} has no {symbol} tuple to parse, so the serving settings "
            f"this comparison needs have moved or been renamed")
    records = []
    for element in node.elts:
        if not isinstance(element, ast.Constant) or not isinstance(
                element.value, str):
            continue
        token = element.value
        line = element.lineno
        if token.startswith("{"):
            records.append({"kind": "additional-config", "name":
                            "--additional_config", "value": token,
                            "token": token, "line": line})
        elif token.startswith("-"):
            name, sep, value = token.partition("=")
            records.append({"kind": "argument", "name": name,
                            "value": value if sep else "", "token": token,
                            "line": line})
        else:
            name, sep, value = token.partition("=")
            records.append({"kind": "environment", "name": name,
                            "value": value if sep else "", "token": token,
                            "line": line})
    return records


def serve_token_review(records, registry_names=()):
    """Which parsed settings the quarantine registry knows, and which it does
    not.

    A setting the registry declares can be applied for a serving-path
    lowering and taken away again, because the tree itself says it reads it.
    A setting the registry does NOT declare is reported by name: it may be a
    variable another layer of the stack reads, or a stale token, and either
    way applying it silently would put a value into a measured program that
    no registry accounts for.
    """
    declared = set(registry_names or ())
    detail = {"tokens": len(records), "environment": {}, "arguments": {},
              "unknown_to_the_registry": [], "additional_config": None,
              "source_lines": {}}
    for record in records:
        detail["source_lines"][record["name"]] = record["line"]
        if record["kind"] == "environment":
            detail["environment"][record["name"]] = record["value"]
            if record["name"] not in declared:
                detail["unknown_to_the_registry"].append({
                    "name": record["name"], "value": record["value"],
                    "line": record["line"],
                    "note": "the tree's own environment registry does not "
                            "declare this, so nothing here may apply it "
                            "silently"})
        elif record["kind"] == "argument":
            detail["arguments"][record["name"]] = record["value"]
        else:
            detail["additional_config"] = record["value"]
    return detail


def structural_diff(label_left, left, label_right, right, allowlist=()):
    """Two lowered programs compared on structure, naming both sides.

    left and right are harness.lowered_census results. Compared: the custom
    call targets and their counts, the collectives and their counts, the
    function count, and the operation census by kind. A difference is reported
    as a row naming what each side had, so a reader sees the pair rather than
    a verdict.

    allowlist entries are (subject, reason) pairs, where subject is a custom
    call target, a collective name or an operation name. A difference in an
    allowlisted subject is reported with its reason and does not fail: an
    allowlist entry is a claim that the difference does not change what the
    layer computes, and it carries the argument for that claim in the output.
    """
    allowed = dict(allowlist or ())
    detail = {"left": label_left, "right": label_right, "rows": [],
              "allowed": [], "problems": []}

    def compare(section, left_counts, right_counts):
        for subject in sorted(set(left_counts) | set(right_counts)):
            here, there = left_counts.get(subject, 0), right_counts.get(
                subject, 0)
            if here == there:
                continue
            row = {"section": section, "subject": subject,
                   label_left: here, label_right: there}
            if subject in allowed:
                row["allowed_because"] = allowed[subject]
                detail["allowed"].append(row)
            else:
                detail["rows"].append(row)
                detail["problems"].append(
                    f"{section} {subject}: {label_left} has {here}, "
                    f"{label_right} has {there}")

    compare("custom call", left.get("custom_calls") or {},
            right.get("custom_calls") or {})
    compare("collective", left.get("collectives") or {},
            right.get("collectives") or {})
    compare("operation", left.get("operations") or {},
            right.get("operations") or {})
    if left.get("functions") != right.get("functions"):
        detail["problems"].append(
            f"function count: {label_left} has {left.get('functions')}, "
            f"{label_right} has {right.get('functions')}")
    return not detail["problems"], detail


# ---------------------------------------------------------------------------
# 3. Source Hash
# ---------------------------------------------------------------------------
# pins.json, beside config.py, is the pin store. Its shape:
#
#   {"<pin name>": {"commit": "<full commit>",
#                   "root": "<directory the paths are relative to>",
#                   "files": {"<relative path>": "<sha256>"}}}
#
# An arm that binds source outside this branch names its pin entry, and
# every file it executed has to be in that entry at the recorded hash. An
# arm that binds this branch's own tree names no pin: its files are hashed
# and RECORDED on the cell instead, because pinning a moving branch against
# itself proves nothing (config.py:272-276).
def source_hash(arms, pins_path=None, our_tree=None):
    """Every arm executed the source its cells will claim it executed.

    arms is {registry key: bound dict}, the key being the arm and the
    profile it runs. Each bound dict carries source_hash and may carry
    sources (the files it executed), pin (a pins.json entry name) and
    namespace (the top-level module it resolves). An arm that carries
    sources has its digest re-derived here with source_digest, so the value
    stamped on its cells is checked rather than trusted; an arm that carries
    a namespace has the loaded module's location checked against its tree.
    """
    path = pins_path or config.PINS_FILE
    detail = {"pins": path, "arms": {}, "problems": []}
    store = {}
    if os.path.exists(path):
        with open(path) as fh:
            store = json.load(fh)
    else:
        detail["pins_missing"] = True
    for name, bound in sorted(arms.items()):
        entry = {"claimed": bound.get("source_hash"),
                 "tree": bound.get("tree"), "tree_sha": bound.get("tree_sha")}
        sources = list(bound.get("sources") or [])
        pin_name = bound.get("pin")
        entry["pin"] = pin_name
        entry["source_count"] = len(sources)
        if not bound.get("source_hash"):
            detail["problems"].append(f"{name} reported no source_hash")
        if not sources:
            entry["note"] = ("the arm declared no source files, so only its "
                             "own digest is stamped and nothing was "
                             "re-derived")
            detail["problems"].append(
                f"{name} declared no source files, so its source_hash cannot "
                f"be checked against anything")
            detail["arms"][name] = entry
            continue
        missing = [p for p in sources if not os.path.exists(p)]
        if missing:
            detail["problems"].append(f"{name} names source files that do "
                                      f"not exist: {missing}")
            detail["arms"][name] = entry
            continue
        root = bound.get("tree") or our_tree or os.path.dirname(
            os.path.abspath(__file__))
        entry["root"] = root
        # THE SOURCES HAVE TO BE UNDER THE TREE THE CELL WILL NAME. An arm
        # that hashes files from one place and stamps a tree from another
        # describes a measurement that did not happen. The live venv holds an
        # editable install of the serving package at its own commit, which is
        # not the pinned tree, so this is the check that keeps an arm from
        # measuring the venv's copy and citing the pin.
        outside = [p for p in sources
                   if os.path.relpath(p, root).startswith(os.pardir)]
        if outside:
            detail["problems"].append(
                f"{name} executed files outside the tree it stamps "
                f"({root}): {outside}")
        per_file = {os.path.relpath(p, root): file_sha256(p) for p in sources}
        entry["files"] = per_file
        # WHERE THE MODULE ACTUALLY CAME FROM. An arm that declares the
        # top-level module it resolves has it checked here: the loaded
        # module's file must be under the arm's tree. Without the
        # declaration the gate says so rather than assuming it was fine.
        #
        # The FILE is a fact gathered where the import happened, which is the
        # child process that bound the tree, and it arrives as namespace_file.
        # Only when this gate runs in the same process that imported does it
        # look the module up itself: asking this process about a module
        # another process imported would report "not loaded" for a perfectly
        # clean arm.
        namespace = bound.get("namespace")
        entry["namespace"] = namespace
        if namespace:
            loaded = bound.get("namespace_file")
            if loaded is None:
                module = sys.modules.get(namespace)
                loaded = getattr(module, "__file__", None) if module else None
            entry["namespace_file"] = loaded
            if loaded is None:
                detail["problems"].append(
                    f"{name} declares it resolves {namespace!r} and no file "
                    f"was reported for it, so nothing shows which tree its "
                    f"code came from")
            elif os.path.relpath(loaded, root).startswith(os.pardir):
                detail["problems"].append(
                    f"{name} stamps tree {root} but {namespace!r} is loaded "
                    f"from {loaded}, which is outside it: the arm measured "
                    f"another checkout, most likely the environment's own "
                    f"installed package")
        else:
            entry["namespace_unchecked"] = (
                "the arm declares no top-level module, so the gate cannot "
                "show which checkout its imports resolved from")
        derived = source_digest(sources, root)
        entry["derived"] = derived
        if derived != bound.get("source_hash"):
            detail["problems"].append(
                f"{name} stamps source_hash {bound.get('source_hash')} but "
                f"its files digest to {derived}")
        if pin_name:
            pinned = (store.get(pin_name) or {})
            entry["pin_commit"] = pinned.get("commit")
            if not pinned:
                detail["problems"].append(
                    f"{name} names pin {pin_name!r}, which is not in {path}")
            else:
                pinned_files = pinned.get("files") or {}
                for rel, got in sorted(per_file.items()):
                    want = pinned_files.get(rel)
                    if want is None:
                        detail["problems"].append(
                            f"{name} executed {rel}, which pin {pin_name!r} "
                            f"does not pin")
                    elif want != got:
                        detail["problems"].append(
                            f"{name} file {rel} is {got} on disk but pin "
                            f"{pin_name!r} records {want}")
        else:
            entry["recorded_not_pinned"] = True
        detail["arms"][name] = entry
    return not detail["problems"], detail


# ---------------------------------------------------------------------------
# 3. Selector Mirror
# ---------------------------------------------------------------------------
def selector_mirror(arms):
    """An arm measured at its owner's selected configuration proves the mirror.

    An arm whose config origin is owner-selector reproduces another
    project's configuration choice, and the flags it stamps are that
    project's answer rather than ours. Such an arm has to carry a
    verify_mirror hook and the hook has to pass, or the cell cannot claim
    the owner's name. Arms at a configuration this project chose may carry
    the hook and it is run when they do.

    The hook takes no arguments and returns either (ok, detail) or a truthy
    value. A raised exception is drift, not an error to route around.
    """
    detail = {"arms": {}, "problems": []}
    for name, bound in sorted(arms.items()):
        hook = bound.get("verify_mirror")
        origin = bound.get("config_origin")
        required = origin == "owner-selector"
        entry = {"declared": hook is not None, "required": required,
                 "config_origin": origin}
        if hook is None:
            if required:
                detail["problems"].append(
                    f"{name} stamps config_origin owner-selector but declares "
                    f"no verify_mirror hook, so nothing shows its flags are "
                    f"the owner's selection rather than ours")
            detail["arms"][name] = entry
            continue
        try:
            result = hook()
        except Exception as exc:
            entry["ok"] = False
            entry["detail"] = f"{type(exc).__name__}: {exc}"
            detail["problems"].append(f"{name} mirror check raised: "
                                      f"{type(exc).__name__}: {exc}")
            detail["arms"][name] = entry
            continue
        if isinstance(result, tuple) and len(result) == 2:
            ok, mirror_detail = result
        else:
            ok, mirror_detail = bool(result), None
        entry["ok"] = bool(ok)
        entry["detail"] = mirror_detail
        if not ok:
            detail["problems"].append(
                f"{name} mirror check reports drift from the owner's "
                f"selector: {mirror_detail}")
        detail["arms"][name] = entry
    return not detail["problems"], detail


# ---------------------------------------------------------------------------
# 4. Build Map
# ---------------------------------------------------------------------------
# THE INPUT IS THE RESOLVED PLAN, never a set of names read back out of the
# configuration. plan is the exact list of cells about to run, after every
# command-line selector has been applied, each entry a dict carrying at
# least arm, batch, routing, profile and tier. declared is the set the
# manifest itself allows. A point in the plan that the manifest does not
# declare fails this gate, which is what keeps a command-line selector from
# adding a point no gate was given.
def plan_key(point):
    """A point's identity: the arm, the profile it runs, and the count.

    The profile is part of it because one arm runs under more than one
    configuration in a manifest: a control and a table row can be the same
    arm at two different profiles, one as-served and one at upstream
    defaults. Keying on the arm alone would collapse the two.
    """
    return (point["arm"], point["profile"], int(point["batch"]))


def show(key):
    return f"{key[0]}@{key[1]}:B{key[2]}"


def build_map(plan, results, declared, expected_refusals=()):
    """Every point about to run either builds or refuses, deviceless, first.

    results is {(arm, profile, batch): {"status": "built" | "refused", ...}}
    from run.py's build map. A point that is missing, or that carries any
    other status, fails the gate: a run that discovers at measurement time
    that an arm cannot build has already spent the session.

    expected_refusals is the manifest's pre-registration, each entry an
    {arm, batch, attribution, reason} dict. It is scored here and reported,
    never gated: a refusal that was predicted and did not happen is good
    news about the kernel and bad news about the prediction, and stopping a
    session over it would trade a measurement session for a manifest edit.
    Every miss is named in the detail so the manifest can be corrected on
    the record.
    """
    allowed = {(a, p, int(b)) for a, p, b in declared}
    keys = [plan_key(p) for p in plan]
    detail = {"planned": len(plan), "built": 0, "refused": 0,
              "declared": len(allowed), "problems": [], "refusals": {},
              "outside_manifest": [], "prediction": {}}
    for point in plan:
        key = plan_key(point)
        if key not in allowed:
            detail["outside_manifest"].append(show(key))
            detail["problems"].append(
                f"{show(key)} is in the resolved plan and the manifest does "
                f"not declare it. A selector may narrow a run; it may not add "
                f"a point that no gate was given.")
    for key in keys:
        got = results.get(key)
        if not got:
            detail["problems"].append(f"{show(key)} is in the plan but was "
                                      f"never built or refused")
            continue
        status = got.get("status")
        if status == "built":
            detail["built"] += 1
        elif status == "refused":
            detail["refused"] += 1
            detail["refusals"][show(key)] = {
                "refusal": got.get("refusal"),
                "attribution": got.get("attribution")}
        else:
            detail["problems"].append(
                f"{show(key)} reports build status {status!r}, which is "
                f"neither built nor refused")
    extra = [k for k in results if k not in set(keys)]
    if extra:
        detail["problems"].append(f"the build map holds points the plan does "
                                  f"not: {sorted(map(str, extra))}")
    if not plan:
        detail["problems"].append("the build map was asked to check an empty "
                                  "plan")
    # Score the pre-registration.
    refused_by_arm = {(k[0], k[2]): v for k, v in results.items()
                      if v.get("status") == "refused"}
    built_by_arm = {(k[0], k[2]) for k, v in results.items()
                    if v.get("status") == "built"}
    misses, matches = [], []
    for expected in expected_refusals or ():
        pair = (expected.get("arm"), int(expected.get("batch")))
        got = refused_by_arm.get(pair)
        if got is None:
            if pair in built_by_arm:
                misses.append(
                    f"{pair[0]} at {pair[1]} tokens was pre-registered to "
                    f"refuse ({expected.get('reason')}) and it built")
            else:
                misses.append(
                    f"{pair[0]} at {pair[1]} tokens was pre-registered to "
                    f"refuse and is not in this run's plan at all")
        elif (expected.get("attribution")
              and got.get("attribution") != expected.get("attribution")):
            misses.append(
                f"{pair[0]} at {pair[1]} tokens refused as "
                f"{got.get('attribution')!r}, pre-registered as "
                f"{expected.get('attribution')!r}: {got.get('refusal')}")
        else:
            matches.append(f"{pair[0]} at {pair[1]} tokens refused as "
                           f"pre-registered ({expected.get('reason')})")
    unexpected = [show(k) for k, v in results.items()
                  if v.get("status") == "refused"
                  and (k[0], k[2]) not in {(e.get("arm"), int(e.get("batch")))
                                           for e in expected_refusals or ()}]
    detail["prediction"] = {"expected": len(expected_refusals or ()),
                            "as_predicted": matches, "misses": misses,
                            "not_predicted": unexpected}
    return not detail["problems"], detail


def prove_plan_scope():
    """Negative control on the scope check above.

    A plan naming a point the manifest does not declare has to fail
    build_map. Without this control the scope check could be silently
    inert, and an inert scope check lets a selector put an ungated arm in a
    table.

    Returns (ok, detail) where ok means the control fired.
    """
    declared = [("declared_arm", "ours-served", 64)]
    inside = [{"arm": "declared_arm", "profile": "ours-served", "batch": 64}]
    outside = inside + [{"arm": "arm_the_manifest_never_declared",
                         "profile": "ours-served", "batch": 64}]
    results = {("declared_arm", "ours-served", 64): {"status": "built"},
               ("arm_the_manifest_never_declared", "ours-served", 64):
                   {"status": "built"}}
    inside_ok, _ = build_map(
        inside, {("declared_arm", "ours-served", 64): {"status": "built"}},
        declared)
    outside_ok, outside_detail = build_map(outside, results, declared)
    detail = {"declared_plan_passes": bool(inside_ok),
              "outside_plan_refused": not outside_ok,
              "outside_manifest": outside_detail.get("outside_manifest"),
              "problems": []}
    if not inside_ok:
        detail["problems"].append("the scope check refuses a plan that is "
                                  "inside the manifest, so it cannot be used")
    if outside_ok:
        detail["problems"].append("the scope check passed a plan holding an "
                                  "arm the manifest never declared")
    return not detail["problems"], detail


# ---------------------------------------------------------------------------
# 5. Deliberate Breakage
# ---------------------------------------------------------------------------
def broken_arm(kind="raises"):
    """A bound arm designed to fail, in the shape bind returns.

    Its call fails on the first invocation, which is inside the warm-up, so
    the failure happens where a real arm's failure would happen and travels
    the same path.

    kind="raises"   an ordinary exception out of the arm's own code, which
                    is a refusal and belongs on a cell.
    kind="asserts"  a bare AssertionError, which is not a refusal at all: it
                    says something the instrument believed impossible
                    happened, and it has to fail the run rather than be
                    booked as a point the kernel could not do.
    """
    def call(x, gating):
        if kind == "asserts":
            assert False, BREAKAGE_MARK
        raise RuntimeError(BREAKAGE_MARK)

    return {
        "call": call,
        "contract": {"anchor": "no_such_op", "router_in_window": False,
                     "dispatch_collectives": (), "combine_kind": "none",
                     "gating_consumed": ()},
        "source_hash": "0" * 64,
        "tree_sha": "0" * 40,
        "tree": os.path.dirname(os.path.abspath(__file__)),
        "flags": {"deliberate_breakage": True},
        "config_origin": "ours-tuned",
        "sources": [],
        # The kind travels on the dict so a runner that measures in another
        # process can build the same injection there: a control that only
        # works in the parent proves nothing about the path the cells take.
        "injection": kind,
        "note": f"{BREAKAGE_MARK} (gates.broken_arm, {kind})",
    }


def deliberate_breakage(run_one_cell):
    """A cell built to fail must be recorded as failed, and an assertion must
    not be recorded at all.

    run_one_cell takes a bound arm and gives back the cell run.py would have
    written for it, or raises when the run must fail. Two injections, both
    required.

    The first raises an ordinary exception. It has to come back as a cell
    with status failed, the injection's own text, an attribution, and no
    numbers: a measurement path that swallows a failure, or turns one into a
    number, cannot be trusted with the cells that matter.

    The second asserts. It must NOT come back as a cell: the path has to
    raise, so the run fails. Booking an assertion failure as a benign
    refusal publishes a harness artefact as a kernel's refusal.

    THE RULE THIS CONTROL ENFORCES: a run whose deliberately broken
    measurement still exits zero is refused. Silent failure has to be
    impossible, and the only way to show that is to break a measurement on
    purpose and watch the path react.
    """
    detail = {"injected": BREAKAGE_MARK, "problems": []}
    try:
        cell = run_one_cell(broken_arm("raises"))
    except Exception as exc:
        detail["problems"].append(
            f"the measurement path raised out of an ordinary arm failure "
            f"instead of recording a failed cell: {type(exc).__name__}: "
            f"{exc}")
        cell = None
    detail["cell_status"] = (cell or {}).get("status")
    detail["cell_refusal"] = (cell or {}).get("refusal")
    detail["cell_attribution"] = (cell or {}).get("refusal_attribution")
    if cell is None:
        pass
    elif not cell:
        detail["problems"].append("the measurement path produced no cell")
    else:
        if cell.get("status") not in ("failed", "refused"):
            # Kernel-attributed exceptions book as refused (a result), and
            # harness-attributed ones as failed (a gap); either way the
            # injected cell must NOT read as ok. The check below on the
            # number fields is the one that catches a swallowed failure.
            detail["problems"].append(
                f"the injected cell was recorded with status "
                f"{cell.get('status')!r}. A cell designed to fail that does "
                f"not read as failed or refused means a real failure can be "
                f"recorded as a number.")
        if BREAKAGE_MARK not in str(cell.get("refusal")):
            detail["problems"].append(
                f"the injected cell failed for a reason that is not the "
                f"injection: {cell.get('refusal')!r}. The control proves "
                f"nothing until it fails for its own reason.")
        if not cell.get("refusal_attribution"):
            detail["problems"].append(
                "the injected cell carries no refusal attribution, so a "
                "reader cannot tell whose limit it was")
        for field in ("program_us", "kernel_self_us", "per_step_us"):
            if cell.get(field) is not None:
                detail["problems"].append(
                    f"the injected cell carries {field}, so a failed "
                    f"measurement still produced a number")
    try:
        assertion_cell = run_one_cell(broken_arm("asserts"))
        detail["assertion_cell_status"] = (assertion_cell or {}).get("status")
        detail["problems"].append(
            f"an assertion failure came back as a cell with status "
            f"{(assertion_cell or {}).get('status')!r} and attribution "
            f"{(assertion_cell or {}).get('refusal_attribution')!r}. An "
            f"assertion is a broken instrument, not a point the kernel "
            f"cannot do, and it has to fail the run.")
    except Exception as exc:
        detail["assertion_raised"] = f"{type(exc).__name__}: {exc}"
    return not detail["problems"], detail


# ---------------------------------------------------------------------------
# 6. Controls
# ---------------------------------------------------------------------------
def controls(opening, closing):
    """What the machine did to the same arm between the session's ends.

    opening and closing are {batch: cell}. The same control arm is measured
    first and last at the same token counts, and the difference is the
    session's own drift at that count. Every cell in the session is stamped
    with it, so a reader can see how much of a gap between two arms could
    be the hour rather than the code.

    This gate computes and reports. It does not fail a run on the size of
    the drift, because no bar for that has been measured here yet, and a bar
    invented in this file would be a number nobody can trace. It fails when
    a control is missing, is not an ok cell, or does not pair.
    """
    detail = {"batches": {}, "problems": []}
    if not opening or not closing:
        detail["problems"].append(
            f"the session needs both controls: opening has "
            f"{len(opening or {})} cells, closing has {len(closing or {})}")
    for batch in sorted(set(opening) | set(closing), key=lambda b: int(b)):
        first, last = opening.get(batch), closing.get(batch)
        if first is None or last is None:
            detail["problems"].append(f"the control at {batch} tokens ran "
                                      f"{'first' if first else 'last'} only")
            continue
        if first.get("status") != "ok" or last.get("status") != "ok":
            detail["problems"].append(
                f"the control at {batch} tokens is not two ok cells: opening "
                f"{first.get('status')}, closing {last.get('status')}")
            continue
        drift = float(last["program_us"]) - float(first["program_us"])
        detail["batches"][str(batch)] = {
            "opening_us": float(first["program_us"]),
            "closing_us": float(last["program_us"]),
            "drift_us": drift,
            "drift_fraction_of_opening": drift / float(first["program_us"]),
        }
    return not detail["problems"], detail


def drift_map(control_detail):
    """{batch: drift microseconds} from a controls() detail, for backfill."""
    return {int(b): v["drift_us"]
            for b, v in (control_detail.get("batches") or {}).items()}


# ---------------------------------------------------------------------------
# 7. Window Census
# ---------------------------------------------------------------------------
# The stage vocabulary and the classifier are one definition, shared by every
# reader of a trace. Forked copies drift, and once they do, per-stage
# attribution differs between two harnesses reading the same trace.
STAGES = ("router", "topk_ag", "hidden_ag", "dispatch_gather", "gmm1", "gmm2",
          "combine_gather", "combine_reduce", "combine_rs", "glue")

# What a contract's combine_kind may say, and the stage that shows it in the
# trace. The vocabulary is the classifier's, so a contract cannot name a
# combine the census has no way to see.
COMBINE_KINDS = {
    "reduce_scatter": "combine_rs",
    "all_reduce": "combine_reduce",
    "gather_reduce": "combine_gather",
    "none": None,
}

# Collective kinds a contract may declare for dispatch, as the profiler's own
# category names.
COLLECTIVE_KINDS = ("all-gather", "all-reduce", "reduce-scatter",
                    "collective-permute")

# Which collective categories a declared combine legitimately accounts for, so
# the census does not report a combine's own collective as undeclared dispatch.
COMBINE_COLLECTIVES = {
    # A declared reduce-scatter owns an all-reduce too: XLA lowers a
    # psum_scatter as an all-reduce followed by a device-id dynamic-slice on
    # this mesh, so the decomposed form is the declared combine, not a
    # stranger. In the trace it reads as %all-reduce to bf16[64,4096]
    # followed by a device-id slice to bf16[8,4096].
    "reduce_scatter": ("reduce-scatter", "all-reduce"),
    "all_reduce": ("all-reduce",),
    "gather_reduce": (),
    "none": (),
}


def classify_op(name, tf_name, category, expr=""):
    """One profiler row onto one stage of the layer.

    The name-scope tests are there because the one-hot permute path's
    dispatch and combine are plain dot fusions whose operation names say
    nothing about routing.
    """
    n, t, e = str(name), str(tf_name), str(expr)
    if (n.startswith("gmm_v2") or "gmm_v2" in t
            or n.startswith("gmm_fused") or "gmm_fused" in t):
        return "gmm2" if "act_None" in (n + t) else "gmm1"
    if "ragged_gather_reduce" in n + t:
        return "combine_gather"
    if "ragged_gather" in n + t:
        return "dispatch_gather"
    if "ring_reduce_scatter" in n + t or "hier_rs_kernel" in n + t:
        return "combine_rs"
    if "_process_tokens_locally" in t or re.search(r"one_hot|onehot", t,
                                                   re.I):
        return "dispatch_gather"
    if re.search(r"moe_gmm_local|expert_parallel_gmm", t) and re.search(
            r"dot|einsum|combine", t, re.I):
        return "combine_gather"
    is_collective = (str(category) in ("all-reduce", "all-gather",
                                       "reduce-scatter", "collective")
                     or re.match(r"(all-reduce|all-gather|reduce-scatter"
                                 r"|collective-permute)", n))
    if is_collective:
        if "all-gather" in n or str(category) == "all-gather":
            if ("sharding_constraint" in t
                    or re.search(r"topk|top_k", t, re.I)
                    or re.match(r"s32|u32|i32", e)):
                return "topk_ag"
            return "hidden_ag"
        if "reduce-scatter" in n or str(category) == "reduce-scatter":
            return "combine_rs"
        return "combine_reduce"
    if re.search(r"top_k|top-k|sort|softmax|argsort|one_hot|one-hot|cumsum"
                 r"|iota|bincount", t, re.I):
        return "router"
    return "glue"


def window_census(arm, contract, ops):
    """What the timed window holds against what the arm said it holds.

    The contract an arm declares at bind time is a claim about its own
    program: whether the router runs inside the window, which collectives
    the dispatch pays for, and how the combine ends. This reads the trace's
    own operation names and compares. A mismatch names both sides, because
    the failure this closes is a table whose rows measure different programs
    and say so nowhere.
    """
    detail = {"arm": arm, "declared": dict(contract), "problems": []}
    stage_us = {s: 0.0 for s in STAGES}
    stage_ops = {s: [] for s in STAGES}
    categories = {}
    for row in ops:
        stage = classify_op(row.get("hlo_op_name"), row.get("tf_op_name"),
                            row.get("category"),
                            row.get("hlo_op_expression", ""))
        stage_us[stage] += float(row.get("us_per_execution", 0.0))
        stage_ops[stage].append(row.get("hlo_op_name"))
        category = str(row.get("category") or "")
        if category in COLLECTIVE_KINDS:
            categories[category] = categories.get(category, 0) + 1
    detail["observed_stage_us"] = stage_us
    detail["observed_collectives"] = categories

    declared_router = bool(contract.get("router_in_window"))
    observed_router = stage_us["router"] > 0.0
    detail["router_in_window_observed"] = observed_router
    if declared_router != observed_router:
        detail["problems"].append(
            f"{arm} declares router_in_window={declared_router} but the "
            f"window {'holds' if observed_router else 'holds no'} router "
            f"operations ({sorted(set(stage_ops['router']))[:6]})")

    declared_collectives = tuple(contract.get("dispatch_collectives") or ())
    unknown = [c for c in declared_collectives if c not in COLLECTIVE_KINDS]
    if unknown:
        detail["problems"].append(
            f"{arm} declares dispatch collectives the census cannot see: "
            f"{unknown}. The vocabulary is {list(COLLECTIVE_KINDS)}.")
    for kind in declared_collectives:
        if kind in COLLECTIVE_KINDS and kind not in categories:
            detail["problems"].append(
                f"{arm} declares a {kind} in dispatch, and the window holds "
                f"none: observed collectives are "
                f"{sorted(categories) or 'none'}")
    combine_owns = COMBINE_COLLECTIVES.get(contract.get("combine_kind")) or ()
    collective_ops = sorted(set(
        stage_ops["hidden_ag"] + stage_ops["topk_ag"]
        + stage_ops["combine_rs"] + stage_ops["combine_reduce"]))
    for kind in sorted(categories):
        if kind not in declared_collectives and kind not in combine_owns:
            detail["problems"].append(
                f"{arm} declares neither a {kind} in dispatch nor a combine "
                f"that owns one, and the window holds {categories[kind]} of "
                f"them ({collective_ops[:6]})")

    declared_combine = contract.get("combine_kind")
    if declared_combine not in COMBINE_KINDS:
        detail["problems"].append(
            f"{arm} declares combine_kind {declared_combine!r}, which is not "
            f"one of {list(COMBINE_KINDS)}")
    else:
        stage = COMBINE_KINDS[declared_combine]
        observed = {k: stage_us[v] for k, v in COMBINE_KINDS.items()
                    if v and stage_us[v] > 0.0}
        detail["combine_observed"] = observed
        if stage is None:
            if observed:
                detail["problems"].append(
                    f"{arm} declares no combine, and the window holds "
                    f"{sorted(observed)}")
        elif stage_us[stage] <= 0.0:
            if (declared_combine == "reduce_scatter"
                    and stage_us["combine_reduce"] > 0.0):
                # The decomposed lowering: the reduce-scatter arrived as an
                # all-reduce plus a device-id slice, which the classifier
                # books under combine_reduce. The combine is the declared
                # one, in its decomposed form, and the census records that
                # instead of refusing it.
                detail["reduce_scatter_decomposed"] = (
                    "the declared reduce-scatter lowered as an all-reduce "
                    "with a device-id dynamic-slice; the combine time sits "
                    "under combine_reduce in this census")
            else:
                detail["problems"].append(
                    f"{arm} declares a {declared_combine} combine, and the "
                    f"window holds "
                    f"{sorted(observed) or 'no combine operations'}")

    anchor = contract.get("anchor")
    if not anchor:
        detail["problems"].append(f"{arm} declares no anchor operation, so "
                                  f"nothing normalizes its window")
    return not detail["problems"], detail


# ---------------------------------------------------------------------------
# 8. Numerics
# ---------------------------------------------------------------------------
def numerics(arm, candidate, reference, reference_arm):
    """Every arm computes the same layer, on the inputs the cells were run on.

    The figure is the relative L2 of the arm's output against the production
    arm's, on identical inputs, and it is stamped rather than argued. The
    named bar is a report threshold: arms differ in the last bits by
    construction, so being above it is a finding to read, not a run to stop.

    What does fail here: a shape that does not match, an output that is not
    finite, or a comparison that could not be made at all. Those are not
    tolerance questions.
    """
    import numpy as np
    detail = {"arm": arm, "reference_arm": reference_arm,
              "tolerance_name": REPORT_TOLERANCE_NAME,
              "report_threshold": REPORT_TOLERANCE,
              "threshold_is_gating": False, "problems": []}
    if candidate is None or reference is None:
        side = "the arm" if candidate is None else f"{reference_arm}"
        detail["problems"].append(f"no comparison exists: {side} produced no "
                                  f"output to compare")
        return False, detail
    got = np.asarray(candidate)
    ref = np.asarray(reference)
    detail["shape"] = list(got.shape)
    detail["reference_shape"] = list(ref.shape)
    if got.shape != ref.shape:
        detail["problems"].append(
            f"{arm} output shape {got.shape} does not match {reference_arm}'s "
            f"{ref.shape}, so the two are not the same layer")
        return False, detail
    got64 = got.astype(np.float64)
    ref64 = ref.astype(np.float64)
    detail["not_finite_candidate"] = int((~np.isfinite(got64)).sum())
    detail["not_finite_reference"] = int((~np.isfinite(ref64)).sum())
    if detail["not_finite_candidate"] or detail["not_finite_reference"]:
        detail["problems"].append(
            f"{arm} comparison holds values that are not finite: "
            f"{detail['not_finite_candidate']} in the arm, "
            f"{detail['not_finite_reference']} in {reference_arm}")
        return False, detail
    value = rel_l2(got64, ref64)
    detail["rel_l2"] = value
    detail["largest_element_difference"] = float(np.max(np.abs(got64 - ref64)))
    detail["above_report_threshold"] = value > REPORT_TOLERANCE
    return True, detail


# ---------------------------------------------------------------------------
# 12. The Serving Path Agreement
# ---------------------------------------------------------------------------
# WHAT THIS GATE VERIFIES, AND WHAT IT DOES NOT. The comparison's production
# row is a layer built by this kit, and a reader is entitled to know it is the
# layer the deployment serves. Two things are checked, deviceless:
#
#   1. The DECISIONS the serving dispatch makes. Every keyword it passes to
#      the layer entry is taken out of its own source, each value's origin is
#      classified from that source, and the ones the settings decide are
#      resolved by the tree's own environment registry under the canonical
#      serve tokens. The arm's stamped keywords are compared against them and
#      a difference names both sides.
#   2. The LAYER PROGRAM. The arm's framed lowering is censused against the
#      entry invoked with the dispatch-resolved keywords, and the two
#      structures are compared operation by operation.
#
# What it does not verify: operand preparation above the layer. The serving
# adapter is a torch surface that needs a layer object, a quantization method
# instance, a forward context and the model wrapper's context, which is a
# serving boot and not a lowering, so its own program cannot be built here.
# What that code contributes is the preparation of the operands the layer
# receives, and that is covered instead by the window census on the device:
# every program executed inside a measured window is counted and named, so
# work the serving path performs around the layer cannot hide inside a cell.
#
# One more limit, stated because the check looks stronger than it is:
# scatter_results and the mesh topology are ONE decision upstream
# (interface/moe.py:110-111 sets scatter_results = is_attn_dp(mesh)), and the
# arm derives its topology from the same flag for that reason. Checking them
# against each other proves the coupling runs in the right direction and that
# the axis override really produces attention data-parallelism. It cannot
# prove the deployment chose that form; the serve tokens are what say so.
DISPATCH_FILE = os.path.join("tpu_inference", "layers", "common", "moe.py")
DISPATCH_FUNCTION = "moe_apply"
LAYER_ENTRY = "fused_moe_func"


def _origin_of(node, locals_seen):
    """Where one keyword's value comes from, read off the tree's own ast."""
    import ast
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "envs":
            return "environment", node.attr
        return node.value.id, None
    if isinstance(node, ast.Name):
        if node.id in locals_seen:
            return "derived", None
        return "local", None
    if isinstance(node, ast.Call):
        target = node.func
        if (isinstance(target, ast.Attribute) and target.attr == "get"
                and isinstance(target.value, ast.Name)):
            return target.value.id, None
        return "call", None
    if isinstance(node, ast.Constant):
        return "literal", None
    return "expression", None


def _env_read_in(node):
    """The environment variable an expression reads, if it reads one."""
    import ast
    for inner in ast.walk(node):
        if (isinstance(inner, ast.Attribute)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "envs"):
            return inner.attr
    return None


def _is_plain_env_read(expression):
    """Whether an expression is a bare env read and nothing else."""
    return bool(re.fullmatch(r"envs\.\s*[A-Z0-9_]+",
                             (expression or "").strip()))


def _kwargs_key_in(node):
    """The kwargs key an expression pops or gets, if it does either."""
    import ast
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call):
            continue
        target = inner.func
        if (isinstance(target, ast.Attribute)
                and target.attr in ("pop", "get")
                and isinstance(target.value, ast.Name)
                and "kwargs" in target.value.id
                and inner.args
                and isinstance(inner.args[0], ast.Constant)):
            return inner.args[0].value
    return None


def dispatch_keywords(tree, dispatch_file=DISPATCH_FILE,
                      function=DISPATCH_FUNCTION, entry=LAYER_ENTRY):
    """Every keyword the serving dispatch passes to the layer entry.

    Read out of the bound tree by an ast walk, never restated here: the call
    to the entry inside the dispatch function is located and each keyword is
    returned with the expression the tree wrote for it, the line it is on, and
    where that value comes from.

        origin  environment  the tree's own env registry decides it, and
                             the environment variable's name is given
                derived      a local the dispatch computes, its assignment
                             returned as the tree wrote it
                <name>       an attribute or a .get on that object, so the
                             serving runner or the layer decides it
                literal      a constant in the dispatch

    Refuses when the file, the function or the call is not there: a gate that
    cannot find its authority has to say so, not pass.
    """
    import ast
    path = os.path.join(tree, dispatch_file)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"the serving dispatch {dispatch_file} is not in {tree}, so what "
            f"it passes to {entry} cannot be read and the comparison would "
            f"be against an assumption")
    with open(path) as fh:
        text = fh.read()
    module = ast.parse(text, filename=path)
    target = None
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            target = node
            break
    if target is None:
        raise ValueError(
            f"{dispatch_file} in {tree} has no {function}, so the keywords it "
            f"passes to {entry} cannot be read")
    assignments = {}
    for node in ast.walk(target):
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name):
                    assignments[name.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target,
                                                            ast.Name):
            if node.value is not None:
                assignments[node.target.id] = node
    call = None
    for node in ast.walk(target):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == entry):
            call = node
            break
    if call is None:
        raise ValueError(
            f"{function} in {tree}/{dispatch_file} does not call {entry}, so "
            f"this tree's dispatch does not reach the layer entry the arm "
            f"binds and the two are not the same path")
    # The dispatch's own kwargs dictionary, wherever this module builds it:
    # a keyword the dispatch pops from it is decided by whatever that literal
    # holds, and when that is a setting the chain ends at a setting.
    supplied = {}
    for node in ast.walk(module):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        written = " ".join(ast.get_source_segment(text, t) or ""
                           for t in targets)
        if "extra_backend_kwargs" not in written:
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                supplied[key.value] = {
                    "expression": ast.get_source_segment(text, value),
                    "line": value.lineno,
                    "environment_variable": _env_read_in(value)}
    keywords = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        origin, variable = _origin_of(keyword.value, assignments)
        record = {"expression": ast.get_source_segment(text, keyword.value),
                  "line": keyword.value.lineno, "origin": origin,
                  "environment_variable": variable}
        node = keyword.value
        # THE CHAIN IS FOLLOWED, NOT GUESSED. A dispatch computes some of
        # these through a local or two, and each hop is the tree's own line:
        # the hops are recorded so a reader can walk the same path.
        chain, seen = [], set()
        while (isinstance(node, ast.Name) and node.id in assignments
               and node.id not in seen and len(chain) < 4):
            seen.add(node.id)
            assigned = assignments[node.id]
            chain.append({"expression": ast.get_source_segment(text, assigned),
                          "line": assigned.lineno})
            found = _env_read_in(assigned.value)
            if found and not record["environment_variable"]:
                record["environment_variable"] = found
            key = _kwargs_key_in(assigned.value)
            if key is not None:
                record["kwargs_key"] = key
            if isinstance(assigned.value, ast.Name):
                node = assigned.value
                continue
            node = None
            for inner in ast.walk(assigned.value):
                if (isinstance(inner, ast.Name) and inner.id in assignments
                        and inner.id not in seen):
                    node = inner
                    break
            if node is None:
                break
        if chain:
            record["derivation"] = chain[0]["expression"]
            record["derivation_line"] = chain[0]["line"]
            record["chain"] = chain
        key = record.get("kwargs_key") or _kwargs_key_in(keyword.value)
        if key is not None:
            record["kwargs_key"] = key
            entry_default = supplied.get(key)
            if entry_default:
                record["supplied_by_dispatch"] = entry_default
                if entry_default["environment_variable"] and not record[
                        "environment_variable"]:
                    record["environment_variable"] = entry_default[
                        "environment_variable"]
                record["untransformed"] = _is_plain_env_read(
                    entry_default["expression"])
        if origin == "environment":
            record["untransformed"] = True
        keywords[keyword.arg] = record
    return {"file": path, "function": function, "entry": entry,
            "line": call.lineno, "keywords": keywords,
            "dispatch_kwargs": supplied}


def dispatch_environment(tree, dispatch, tokens):
    """The values the tree's own registry gives those keywords.

    The canonical serve tokens are put in the environment, the tree's env
    registry is imported from the tree, and every environment variable the
    dispatch reads is read through it. The registry decides, not this file:
    each variable's default and its parse are the tree's own, so a variable
    the tokens do not set comes back as the value serving would run with
    anyway, marked as coming from the default.

    The environment is restored before returning. Reading a setting must not
    change the process that read it.
    """
    wanted, values = {}, {}
    for name, record in (dispatch.get("keywords") or {}).items():
        variable = record.get("environment_variable")
        if variable:
            wanted[name] = variable
    if not wanted:
        return {"values": values, "refusal": None}
    settings = {t["name"]: t["value"] for t in tokens
                if t.get("kind") == "environment"}
    previous, restore = {}, []
    for variable in sorted(set(wanted.values())):
        previous[variable] = os.environ.get(variable)
        if variable in settings:
            os.environ[variable] = str(settings[variable])
            restore.append(variable)
    removed = sys.modules.pop("tpu_inference.envs", None)
    added = tree not in sys.path
    if added:
        sys.path.insert(0, tree)
    try:
        import importlib
        registry = importlib.import_module("tpu_inference.envs")
        for name, variable in sorted(wanted.items()):
            values[name] = {
                "environment_variable": variable,
                "value": getattr(registry, variable),
                "set_by_tokens": variable in settings,
                "token_value": settings.get(variable)}
        refusal = None
    except Exception as exc:
        refusal = (f"the tree's own environment registry could not be read "
                   f"({type(exc).__name__}: {exc}), so what the serving "
                   f"dispatch resolves cannot be established and the "
                   f"comparison would be against an assumption")
    finally:
        if added:
            try:
                sys.path.remove(tree)
            except ValueError:
                pass
        sys.modules.pop("tpu_inference.envs", None)
        if removed is not None:
            sys.modules["tpu_inference.envs"] = removed
        for variable in restore:
            if previous[variable] is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous[variable]
    return {"values": values, "refusal": refusal}


SERVING_AGREEMENT_SCOPE = (
    "This gate verifies the decisions the serving dispatch makes and the "
    "layer program they produce. It does not verify the preparation of the "
    "operands above the layer: that code is a torch surface needing a "
    "serving boot, and what it contributes is covered by the window census "
    "on the device, which counts and names every program executed inside a "
    "measured window.")


def serving_path_agreement(tree, tokens, arm_flags, arm_label="the arm",
                           dispatch_label="the serving dispatch",
                           attn_dp_from_mesh=None, arm_census=None,
                           entry_census=None, allowlist=(),
                           entry_defaults=None):
    """The twelfth gate: the arm is the layer the deployment serves.

    arm_flags is what the arm stamped as sent to the layer entry
    (provenance.keywords_sent). attn_dp_from_mesh, when given, is what the
    tree's own is_attn_dp returned for the arm's mesh. arm_census and
    entry_census, when both given, are compared with structural_diff.

    Every row names both sides. A keyword the dispatch decides from the
    settings and the arm does not send is a problem, because the arm would
    then run the entry's default while serving runs the setting; a keyword
    whose values differ is a problem; a structural difference in the two
    programs is a problem unless the allowlist carries a reason for it.
    """
    dispatch = dispatch_keywords(tree)
    resolved = dispatch_environment(tree, dispatch, tokens)
    detail = {"scope": SERVING_AGREEMENT_SCOPE, "tree": tree,
              "dispatch": {"file": dispatch["file"], "line": dispatch["line"],
                           "function": dispatch["function"],
                           "entry": dispatch["entry"]},
              "settings_decided": [], "serving_decided": [],
              "program": None, "problems": []}
    if resolved["refusal"]:
        detail["problems"].append(resolved["refusal"])
        return False, detail
    sent = dict(arm_flags or {})
    for name, record in sorted((dispatch.get("keywords") or {}).items()):
        origin = record["origin"]
        if name in (resolved["values"] or {}):
            here = resolved["values"][name]
            row = {"keyword": name, dispatch_label: here["value"],
                   arm_label: sent.get(name, "(not sent)"),
                   "environment_variable": here["environment_variable"],
                   "set_by_tokens": here["set_by_tokens"],
                   "expression": record["expression"],
                   "line": record["line"]}
            if record.get("derivation") and not record.get("untransformed"):
                row["derivation"] = record["derivation"]
                # A keyword the dispatch computes from a setting cannot be
                # compared value to value: the derivation is the dispatch's
                # own and evaluating it here would restate it. The setting it
                # reads is reported, and the arm's keyword beside it, for a
                # reader to see the pair.
                row["compared"] = False
                detail["settings_decided"].append(row)
                continue
            row["compared"] = True
            detail["settings_decided"].append(row)
            if name not in sent:
                # The arm sends only what its profile names, and a keyword it
                # omits runs the entry's own default. That is agreement when
                # the default is what the setting resolves to, and a real
                # difference when it is not, so the entry's declared defaults
                # decide which of the two this is.
                defaults = entry_defaults or {}
                default = defaults.get(name, "(default unknown)")
                row[arm_label] = f"(not sent, entry default {default!r})"
                if name in defaults and default == here["value"]:
                    row["agrees_by_default"] = True
                else:
                    detail["problems"].append(
                        f"{name}: {dispatch_label} passes "
                        f"{here['value']!r} from "
                        f"{here['environment_variable']} and {arm_label} does "
                        f"not send the keyword, so the arm runs the entry's "
                        f"default {default!r} where serving runs the setting")
            elif sent[name] != here["value"]:
                detail["problems"].append(
                    f"{name}: {dispatch_label} passes {here['value']!r} "
                    f"(from {here['environment_variable']}) and {arm_label} "
                    f"sends {sent[name]!r}")
        else:
            detail["serving_decided"].append(
                {"keyword": name, "origin": origin,
                 "expression": record["expression"], "line": record["line"],
                 arm_label: sent.get(name, "(not sent)")})
    if attn_dp_from_mesh is not None:
        coupled = bool(sent.get("scatter_results"))
        detail["topology"] = {
            "is_attn_dp(mesh) on the arm's mesh": bool(attn_dp_from_mesh),
            f"scatter_results {arm_label} sends": coupled,
            "why they are one decision": SERVING_COUPLING}
        if bool(attn_dp_from_mesh) != coupled:
            detail["problems"].append(
                f"scatter_results: {arm_label} sends {coupled!r} and the "
                f"tree's own is_attn_dp returns {bool(attn_dp_from_mesh)!r} "
                f"for the mesh the arm built, so the arm's topology and its "
                f"combine are the incoherent mixture unmodified serving "
                f"cannot produce")
    if arm_census is not None and entry_census is not None:
        same, program = structural_diff(
            dispatch_label + " entry, dispatch-resolved keywords",
            entry_census, arm_label, arm_census, allowlist=allowlist)
        detail["program"] = program
        if not same:
            detail["problems"].extend(program["problems"])
    return not detail["problems"], detail


SERVING_COUPLING = (
    "the serving adapter sets scatter_results from is_attn_dp(mesh) "
    "(tpu_inference/layers/vllm/interface/moe.py:110-111), so the topology "
    "and the scattered combine are one decision, and the arm derives its "
    "topology from the same flag for that reason")
