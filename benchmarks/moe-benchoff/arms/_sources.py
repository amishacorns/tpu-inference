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

"""Externals, trees and the operand spine every arm shares.

Nothing here measures anything. This file answers the questions an arm must
not answer twice: where an external source came from and whether it is the
pinned one, which tree an arm bound, how a host array reaches a device, and
what the one weight master is.

No value in this file is a configuration choice. Shapes arrive from
config.SHAPES through the caller, pins from pins.json, flag values from the
profile, and the weight master is derived from the shape being measured.

REFUSALS ARE HARNESS'S CLASSES, NOT A SECOND SET. harness.py already
defines the one refusal hierarchy and run.py attributes a refusal by its
TYPE (harness.attribution_for). A second hierarchy defined here would be a
different set of classes under the same names, so an arm's refusal would
stop being recognised and would be booked as a defect. This module
re-exports them and defines none of its own.

NO MTIME AND NO DIRECTORY ORDER anywhere in this package. An external
resolves by pinned path and sha256; a worktree resolves by commit; a
package copy carries exactly the files the pin names.
"""

import ast
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The kit whose modules and pins THIS process reads: a snapshot copies every
# top-level module, the pins and the arm modules into the run directory, and
# those copies are the code the run executes (run.py:446-470).
KIT = os.path.dirname(HERE)
# WHERE THE FETCHED CHECKOUTS LIVE, which is not the same question. The
# snapshot copies the arm modules and NOT the checkouts beside them: they are
# pinned by content hash rather than copied, because the pins are the record.
# So the cache resolves from the editable kit the run was launched from, which
# run.py exports as BENCHOFF_KIT_ORIGIN (run.py:481, :490), and falls back to
# this module's own location only when nothing exported it. An arm resolving
# from its module location would re-fetch tens of megabytes into every run
# directory and would bind a checkout no pin describes.
ORIGIN = os.environ.get("BENCHOFF_KIT_ORIGIN") or KIT
CACHE = os.path.join(ORIGIN, "arms", "_sources")
if KIT not in sys.path:
    sys.path.insert(0, KIT)

import config  # noqa: E402  the one place a value may exist
import gates  # noqa: E402  the source-digest rule the gate re-derives
import harness  # noqa: E402  the one refusal hierarchy

Refusal = harness.Refusal
KernelRefusal = harness.KernelRefusal
HarnessRefusal = harness.HarnessRefusal
ShapeConstraintRefusal = harness.ShapeConstraintRefusal
FetchRefusal = harness.FetchRefusal
OutOfScopeRefusal = harness.OutOfScopeRefusal


def owner_refusal(exc):
    """An owner's own refusal, kept verbatim and attributed to the kernel."""
    return KernelRefusal(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Modules: from a file, from a tree.
# ---------------------------------------------------------------------------
def module_from_file(path, key):
    """One file as a module, under a key that cannot shadow another arm's."""
    if key in sys.modules:
        return sys.modules[key]
    if not os.path.exists(path):
        raise FetchRefusal(f"no source file at {path}")
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def assert_under(module, expect):
    """Where a module resolved from, or a refusal to build a mixed source."""
    f = os.path.realpath(getattr(module, "__file__", "") or "")
    if not f.startswith(os.path.realpath(expect)):
        raise AssertionError(
            f"PROVENANCE VIOLATION: {module.__name__} resolved from {f!r}, "
            f"expected under {expect!r} -- refusing to build a mixed-source "
            f"arm")
    return f


def import_from_tree(tree, dotted, top="tpu_inference"):
    """A module from one named tree, refusing a tree already loaded."""
    tree = os.path.realpath(tree)
    for name in [m for m in list(sys.modules)
                 if m == top or m.startswith(top + ".")]:
        f = os.path.realpath(getattr(sys.modules[name], "__file__", "") or "")
        if f and not f.startswith(tree):
            raise AssertionError(
                f"PROVENANCE VIOLATION: {name} is already loaded from {f}, "
                f"which is not under this arm's tree {tree}. Run this arm in "
                f"its own process; refusing to build.")
    if tree not in sys.path:
        sys.path.insert(0, tree)
    mod = importlib.import_module(dotted)
    assert_under(mod, tree)
    return mod


def own_tree():
    """The tree this kit ships in, which is what our own arms bind.

    Read from the editable kit rather than from a snapshot copy: a run
    directory can sit outside the repository, and a snapshot's own location
    answers a different question than "which checkout is this kit part of".
    """
    return _run(["git", "-C", ORIGIN, "rev-parse", "--show-toplevel"])


def tree_sha(tree):
    """The revision an arm is bound to, read from the tree itself."""
    try:
        return _run(["git", "-C", tree, "rev-parse", "HEAD"])
    except (RuntimeError, OSError):
        return "unknown"


# ---------------------------------------------------------------------------
# Reading a value out of a source file without importing it. An owner's
# tuned table is data inside a module their package cannot be imported
# without, so it is parsed rather than executed.
# ---------------------------------------------------------------------------
def _assign(path, name):
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [getattr(t, "id", None) for t in node.targets]
        elif isinstance(node, ast.AnnAssign):
            names = [getattr(node.target, "id", None)]
        else:
            continue
        if name in names:
            return node
    raise AssertionError(f"{path} has no assignment to {name}")


def read_literal(path, name):
    """A literal a source file declares: their table, their constant."""
    return ast.literal_eval(_assign(path, name).value)


def read_call_kwargs(path, name):
    """The keyword literals of a constructor call a source file declares."""
    node = _assign(path, name).value
    if not isinstance(node, ast.Call):
        raise AssertionError(f"{name} in {path} is not a constructor call")
    return {k.arg: ast.literal_eval(k.value) for k in node.keywords}


def read_call_literals(path, func_name):
    """Every call to a named function in a file, as its literal arguments.

    Used to read an owner's own call-site defaults -- the value their code
    passes when the environment says nothing -- rather than restating them.
    """
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == func_name):
            try:
                out.append(tuple(ast.literal_eval(a) for a in node.args))
            except ValueError:
                continue
    if not out:
        raise AssertionError(f"{path} holds no literal call to {func_name}")
    return out


def signature_defaults(fn, names):
    """An owner's own parameter defaults, read off their signature.

    A default this kit never passes still decides the program, so a cell
    records it. Reading it here rather than writing it down means a pin that
    moves the default moves the stamp with it.
    """
    import inspect
    params = inspect.signature(fn).parameters
    missing = [n for n in names if n not in params]
    if missing:
        raise AssertionError(
            f"{getattr(fn, '__qualname__', fn)} has no parameters {missing}: "
            f"the stamp would record a default that does not exist")
    return {n: params[n].default for n in names}


def read_only_int(path, name):
    """The one integer in an assignment, refusing if it is not the only one."""
    node = _assign(path, name)
    found = [n.value for n in ast.walk(node)
             if isinstance(n, ast.Constant) and isinstance(n.value, int)
             and not isinstance(n.value, bool)]
    if len(found) != 1:
        raise AssertionError(
            f"{name} in {path} carries {len(found)} integers ({found}); this "
            f"mirror reads one and refuses to guess which")
    return found[0]


# ---------------------------------------------------------------------------
# Externals: fetched at their pin, verified file by file.
# ---------------------------------------------------------------------------
def load_pins():
    with open(os.path.join(KIT, config.PINS_FILE)) as f:
        return json.load(f)


def _run(args):
    out = subprocess.run(args, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def _patched(root, entry):
    """A pinned checkout with the pin's own patch applied, if it names one.

    A tile pin is the released package at its commit plus one hand-written
    block-size entry, and the diff that writes that entry ships beside the
    pins under pins_patches/. THE PINNED PER-FILE SHA256S ARE THE PATCHED
    HASHES, so the patch is applied here, before fetch_and_verify hashes
    anything: a pin whose patch did not run would fail its own verification.

    The checkout is reset to the pinned commit first, so a cache directory
    reused across runs carries the patch exactly once and never twice.
    """
    rel = entry.get("patch_file")
    if not rel:
        return root
    diff = os.path.join(KIT, rel)
    if not os.path.exists(diff):
        raise FetchRefusal(
            f"pin {entry['commit']} names patch_file {rel!r}, which is not "
            f"at {diff}: the pinned hashes are the patched ones, so the pin "
            f"cannot be satisfied without it")
    _run(["git", "-C", root, "checkout", "-q", "--force", "--detach",
          entry["commit"]])
    _run(["git", "-C", root, "apply", "--whitespace=nowarn", diff])
    return root


def _checkout(name, entry, allow_env_root=True):
    """The pinned checkout for one external.

    The kit's own checkout under _sources/<name> at the pinned commit comes
    first. An environment-named clone is a convenience for a run that
    already has one and is used only when it is allowed, never for a mirror
    check. Either way the pin decides: every file is hashed below.
    """
    root = os.path.join(CACHE, name)
    if os.path.isdir(os.path.join(root, ".git")):
        if _run(["git", "-C", root, "rev-parse", "HEAD"]) == entry["commit"]:
            return _patched(root, entry)
    if allow_env_root:
        env = os.environ.get("BENCHOFF_SRC_" + name.upper())
        if env:
            return os.path.realpath(env)
    os.makedirs(root, exist_ok=True)
    if not os.path.isdir(os.path.join(root, ".git")):
        _run(["git", "init", "-q", root])
        _run(["git", "-C", root, "remote", "add", "origin", entry["repo"]])
    _run(["git", "-C", root, "fetch", "--depth", "1", "origin",
          entry.get("fetch_ref") or entry["commit"]])
    _run(["git", "-C", root, "checkout", "-q", "--detach", entry["commit"]])
    return _patched(root, entry)


def fetch_and_verify(pins=None, names=None, allow_env_root=True):
    """Every external at its pin, with every named file's hash checked.

    Returns {name: {"root", "commit", "package_root", "files"}}, where files
    maps the pinned relative path to its verified sha256. A hash that does
    not match raises with the pin printed: a source that is not the pinned
    one is not a source this benchmark measures.
    """
    pins = load_pins() if pins is None else pins
    out = {}
    for name in sorted(pins):
        if names is not None and name not in names:
            continue
        entry = pins[name]
        root = _checkout(name, entry, allow_env_root)
        checked = {}
        for rel in sorted(entry["files"]):
            want, path = entry["files"][rel], os.path.join(root, rel)
            if not os.path.exists(path):
                raise FetchRefusal(
                    f"pin {name} {entry['commit']} names {rel}, which is not "
                    f"in {root}")
            got = sha256_file(path)
            if got != want:
                raise FetchRefusal(
                    f"PIN MISMATCH {name} {rel}\n  pinned repo "
                    f"{entry['repo']}\n  pinned commit {entry['commit']}\n"
                    f"  pinned sha256 {want}\n  file sha256   {got}\n"
                    f"  file {path}")
            checked[rel] = got
        res = {"root": root, "commit": entry["commit"], "files": checked,
               "package_root": None}
        if entry.get("package_dir"):
            res["package_root"] = _importable_copy(root, entry)
        out[name] = res
    return out


def sha256_file(path):
    """The gate's own file hash, so one rule answers everywhere."""
    return gates.file_sha256(path)


def _importable_copy(root, entry):
    """The pinned package files under an importable name of their own.

    A released kernel package inside another project's namespace cannot be
    imported beside the tree an arm binds without one shadowing the other. A
    copy under its own name removes the question. Exactly the files the pin
    names are copied, so nothing unpinned can be imported from it.

    The copy lives INSIDE the pinned checkout, so the source gate's namespace
    check sees the loaded package under the tree the cell names rather than
    beside it (gates.py:839-872). A run pointed at its own clone by the
    environment gains one ignored directory there.
    """
    parent = os.path.join(root, "_pkg")
    dest = os.path.join(parent, entry["package_name"])
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    for rel in sorted(entry["files"]):
        if not rel.startswith(entry["package_dir"] + "/"):
            continue
        inner = os.path.relpath(rel, entry["package_dir"])
        target = os.path.join(dest, inner)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(os.path.join(root, rel), target)
        if sha256_file(target) != entry["files"][rel]:
            raise FetchRefusal(
                f"the importable copy of {entry['package_name']} does not "
                f"match the pin at {rel}")
    return parent


def _has_commit(tree, commit):
    """Whether a checkout holds one commit's object, without fetching."""
    try:
        return _run(["git", "-C", tree, "cat-file", "-t", commit]) == "commit"
    except (RuntimeError, OSError):
        return False


def tree_worktree(upstream_pin):
    """A worktree of THIS repository at a pin, created once and reused.

    The production arm binds unmodified upstream, which is a revision of the
    repository this kit ships in rather than an external project. Resolution
    is by commit and by nothing else.

    A CLONE THAT DOES NOT HOLD THE COMMIT FETCHES IT. A shallow clone, or a
    clone of a fork whose history does not reach upstream main, has no
    object to make a worktree from, and a kit that could only run where the
    commit happened to be local would not run from a fresh clone at all. So
    the commit is looked for in this checkout first, and when it is not here
    the pins.json entry naming it is fetched like any other external. The
    entry pins no per-file hashes because the arm binds the whole tree
    rather than a named set of files; the commit is what decides it, and
    git refuses a checkout of any other.
    """
    path = os.path.join(CACHE, "tree-" + upstream_pin[:12])
    if os.path.isdir(path):
        at = _run(["git", "-C", path, "rev-parse", "HEAD"])
        if at != upstream_pin:
            raise HarnessRefusal(
                f"{path} is at {at}, not at the pin {upstream_pin}")
        return path
    os.makedirs(CACHE, exist_ok=True)
    try:
        tree = own_tree()
    except (RuntimeError, OSError):
        tree = None
    if tree and _has_commit(tree, upstream_pin):
        _run(["git", "-C", tree, "worktree", "add", "--detach", path,
              upstream_pin])
        return path
    entry = load_pins().get(config.UPSTREAM_PIN_NAME)
    if entry is None:
        raise FetchRefusal(
            f"this checkout does not hold {upstream_pin} and "
            f"{config.PINS_FILE} has no {config.UPSTREAM_PIN_NAME!r} entry, "
            f"so the production arm has no tree to bind")
    if entry["commit"] != upstream_pin:
        raise HarnessRefusal(
            f"config.UPSTREAM_PIN is {upstream_pin} and the "
            f"{config.UPSTREAM_PIN_NAME!r} pin is {entry['commit']}: the two name "
            f"different trees and nothing here may choose between them")
    return _checkout(config.UPSTREAM_PIN_NAME, entry)


def tree_for_arm(pin_name, needs, tree_path=None):
    """The tree an arm of this branch binds: this checkout, or its own pin.

    A kit sitting on the tree that carries the kernel measures THAT tree, so
    own_tree() is preferred and a live checkout is what the cell records.

    A CLONE THAT DOES NOT CARRY THE KERNEL FETCHES ITS PIN. These arms
    measure work that is not on upstream main, so a clone of upstream has no
    module for them to import and binding it would fail at the import rather
    than say why. When any path in `needs` is absent from this checkout the
    arm falls back to its own pins.json entry, fetched and hash verified
    exactly like an external, and binds that checkout instead. `needs` are
    repository-relative paths the arm imports.

    Returns (tree, fetched), where fetched is the fetch_and_verify record
    when the pin was taken and None when the live tree served. A caller that
    took the pin names it on its bound dict, so the source gate checks the
    files it executed against the pinned hashes.
    """
    if tree_path:
        return os.path.realpath(tree_path), None
    try:
        tree = own_tree()
    except (RuntimeError, OSError):
        tree = None
    if tree and all(os.path.exists(os.path.join(tree, n)) for n in needs):
        return os.path.realpath(tree), None
    fetched = fetch_and_verify(names=[pin_name])[pin_name]
    return os.path.realpath(fetched["root"]), fetched


# ---------------------------------------------------------------------------
# Operands. Every layout conversion an arm's own form needs happens on the
# host, before placement, so a timed window never contains a cast or a
# transpose that belongs to the benchmark rather than to the kernel.
# ---------------------------------------------------------------------------
_WEIGHTS = {}


def block_scaled(shape):
    """Whether this shape's weights carry a scale per contraction block.

    The four-bit float form and nothing else. It is asked by dtype rather
    than by a field of its own because the dtype IS the question every
    kernel here asks: the fused expert-parallel layer derives its weight
    format from the weight element type and refuses anything outside its
    table (kernels/fused_moe/v2/host.py weight_format_of_dtype).
    """
    import jax.numpy as jnp
    return jnp.dtype(shape["weight_dtype"]) == jnp.dtype(jnp.float4_e2m1fn)


def scale_shapes(shape):
    """The two scale tables' shapes, answered from the shape alone.

    [experts, blocks, 1, channels] either way: one block for the
    per-channel forms, one per config.WEIGHT_SCALE_BLOCK contraction rows
    for the four-bit form. A deviceless build check needs the layout and
    not the values, and the master draws tens of gigabytes, so the layout
    is derived here and the master derives its own from this.
    """
    E, H, IM = shape["experts"], shape["hidden"], shape["intermediate"]
    if not block_scaled(shape):
        return (E, 1, 1, 2 * IM), (E, 1, 1, H)
    b = config.WEIGHT_SCALE_BLOCK
    for name, k in (("hidden", H), ("intermediate", IM)):
        if k % b:
            raise ShapeConstraintRefusal(
                f"a four-bit weight carries one scale per {b} contraction "
                f"rows and this shape's {name} is {k}, which is not a whole "
                f"number of them; both matmuls block at the one width")
    return (E, H // b, 1, 2 * IM), (E, IM // b, 1, H)


def _block_quantize(base, experts, wdt, block):
    """One expert-stacked weight slab and its per-contraction-block scales.

    base is the [contraction, channels] float32 draw the per-channel path
    rolls, and the return is the slab [experts, contraction, channels] in
    `wdt` with the scale table [experts, blocks, 1, channels] float32.

    The arithmetic is the serving weight loader's own
    (layers/common/quantization quantize_tensor, reached from
    process_weights/moe_weights.py quantize_moe_weights): a block's scale
    is its largest magnitude divided by the format's peak, and the stored
    value is the row divided by that scale. The draw is divided by the
    square root of the contraction width first, which is the same
    reciprocal square root the per-channel path carries in its scale, so a
    dequantized row lands near unit variance here too. A block whose
    values are all zero has no scale to divide by and stores zeros.
    """
    import jax.numpy as jnp
    import numpy as np
    K, N = base.shape
    nb = K // block
    peak = float(jnp.finfo(wdt).max)
    w = np.empty((experts, K, N), dtype=wdt)
    s = np.empty((experts, nb, 1, N), dtype=np.float32)
    for e in range(experts):
        rolled = (np.roll(base, e, axis=0) * (K ** -0.5)).reshape(nb, block, N)
        scale = np.abs(rolled).max(axis=1) / peak            # [blocks, N]
        usable = scale > 0
        inv = np.where(usable, 1.0 / np.where(usable, scale, 1.0), 0.0)
        w[e] = np.clip(rolled * inv[:, None, :], -peak,
                       peak).reshape(K, N).astype(wdt)
        s[e] = scale[:, None, :]
    return w, s


def weight_master(shape):
    """The one synthetic weight source for a shape, memoized per process.

    The served form of the shape, in the shape's own weight element type.

    EIGHT BITS, ONE SCALE PER OUTPUT CHANNEL. The scale is the reciprocal
    square root of each matmul's contraction width, so a dequantized row
    lands near unit variance at any shape; a reciprocal and a square root
    are the only arithmetic here and neither is a tuned value.

    FOUR BITS, ONE SCALE PER CONTRACTION BLOCK. The same draw, divided by
    that same reciprocal square root and then quantized block by block
    along the contraction axis, config.WEIGHT_SCALE_BLOCK rows to a block:
    a block's scale is its largest magnitude divided by the format's own
    peak and the stored value is the row divided by that scale, which is
    the serving weight loader's arithmetic and again holds a dequantized
    row near unit variance. A largest magnitude, a reciprocal and a square
    root are the whole of it, and the format's peak is read off the format.

    Both forms hand back the same four keys in the same layout,
    [experts, blocks, 1, channels] for a scale table, so an arm places one
    or the other without asking which it got; scale_shapes answers the
    block count for a build check that must not draw the values.

    The draw is seeded from the shape itself and every expert is the same
    standard-normal block rolled by its own index, which gives every expert
    distinct weights without drawing tens of gigabytes. Nothing in a
    matmul's timing depends on the values, and every arm in every process
    derives the same master, which is what the numerics gate needs.
    """
    key = json.dumps(shape, sort_keys=True)
    if key in _WEIGHTS:
        return _WEIGHTS[key]
    import jax.numpy as jnp
    import numpy as np
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    E, H, IM = shape["experts"], shape["hidden"], shape["intermediate"]
    wdt = jnp.dtype(shape["weight_dtype"])
    base13 = rng.standard_normal((H, 2 * IM), dtype=np.float32)
    base2 = rng.standard_normal((IM, H), dtype=np.float32)
    if block_scaled(shape):
        # Asked first, so a shape the block width does not divide is refused
        # before either slab is allocated.
        scale_shapes(shape)
        b = config.WEIGHT_SCALE_BLOCK
        w13, s13 = _block_quantize(base13, E, wdt, b)
        w2, s2 = _block_quantize(base2, E, wdt, b)
        out = dict(w13=w13, w2=w2, s13=s13, s2=s2)
    else:
        w13 = np.empty((E, H, 2 * IM), dtype=wdt)
        w2 = np.empty((E, IM, H), dtype=wdt)
        for e in range(E):
            w13[e] = np.roll(base13, e, axis=0).astype(wdt)
            w2[e] = np.roll(base2, e, axis=0).astype(wdt)
        out = dict(w13=w13, w2=w2,
                   s13=np.full((E, 1, 1, 2 * IM), H ** -0.5, np.float32),
                   s2=np.full((E, 1, 1, H), IM ** -0.5, np.float32))
    _WEIGHTS[key] = out
    return out


def make_mesh(axes, ep):
    """The bench mesh: the expert-parallel axis last, the rest of size one."""
    import jax
    import numpy as np
    from jax.sharding import Mesh
    devs = jax.devices()
    if len(devs) < ep:
        raise HarnessRefusal(f"need {ep} devices, have {len(devs)}")
    dims = (1,) * (len(axes) - 1) + (ep,)
    return Mesh(np.array(devs[:ep]).reshape(dims), axes)


def put(a, shape, dtype, sharding):
    """One host array, converted on the host and placed, before the window."""
    import jax
    import jax.numpy as jnp
    arr = jnp.asarray(a, dtype=jnp.dtype(dtype))
    if arr.shape != tuple(shape):
        raise AssertionError(
            f"placed array has shape {arr.shape}, expected {tuple(shape)}")
    return jax.device_put(arr, sharding)


def spec(shape, dtype, sharding=None):
    """The abstract form of one operand, for a deviceless build check."""
    import jax
    import jax.numpy as jnp
    return jax.ShapeDtypeStruct(tuple(shape), jnp.dtype(dtype),
                                sharding=sharding)


def jit(fn):
    """The arm's own jit: one program per call, and no compiler options.

    THE COMPILER FRAME IS THE DRIVER'S. run.py wraps this in the one jit
    that carries config.FRAMES[frame], so an arm that named an option of its
    own would put one kernel's tuning into a comparison as if it were the
    serving frame.
    """
    import jax
    return jax.jit(fn)


def lower_for_device(jitted, *specs):
    """Trace and lower one program on this host, FOR THE DEVICE.

    A build check has to lower the program the measured run will compile,
    not a host-platform neighbour of it: a pallas kernel refuses to lower
    for the host platform at all ("only interpret mode is supported on CPU
    backend"), so a build map that lowered for the host would prove nothing
    about the arms whose kernels are pallas. jax's own two-step names the
    lowering platform, and the trace-time device probe run.py pins answers
    the kernels' own device questions.
    """
    return jitted.trace(*specs).lower(lowering_platforms=("tpu",))


def contract(anchor, router_in_window, dispatch_collectives, combine_kind,
             gating_consumed):
    """The window contract, in the census's own vocabulary."""
    for kind in dispatch_collectives:
        if kind not in gates.COLLECTIVE_KINDS:
            raise AssertionError(
                f"{kind!r} is not a collective the census can see: "
                f"{gates.COLLECTIVE_KINDS}")
    if combine_kind not in gates.COMBINE_KINDS:
        raise AssertionError(
            f"{combine_kind!r} is not one of {list(gates.COMBINE_KINDS)}")
    return {"anchor": anchor, "router_in_window": bool(router_in_window),
            "dispatch_collectives": tuple(dispatch_collectives),
            "combine_kind": combine_kind,
            "gating_consumed": tuple(gating_consumed)}


def source_hash(paths, root):
    """The gate's own digest rule, never a second one (gates.py:161-176)."""
    return gates.source_digest(paths, root)


def owner_device_key(root, rel):
    """The device kind an owner's tuned table is keyed on, by their own rule.

    Their function is loaded from their pinned file and called, so the key is
    theirs rather than a restatement of theirs.

    THE FALLBACK IS FOR ONE CASE ONLY, and the narrowing is the point. Their
    rule refuses a host platform outright, which is every deviceless build,
    and there the key is the device this kit requires and the measured run
    asserts. Their rule ALSO refuses a TPU generation it does not recognise,
    and swallowing that would key a v7 table on a v6e box: a configuration
    nobody selected, published under their name. So the fallback is taken
    only when the backend reports no TPU at all, and their refusal is
    re-raised otherwise.
    """
    mod = module_from_file(os.path.join(root, rel), "sgl_jax_jax_utils")
    try:
        return mod.get_device_name(), "their rule, on this device"
    except Exception as exc:
        kinds = gates.device_facts()["kinds"]
        if any("TPU" in k for k in kinds):
            raise KernelRefusal(
                f"their device rule refuses this backend, which reports "
                f"{kinds}: {type(exc).__name__}: {exc}. Their tuned table is "
                f"keyed on a kind their rule recognises, so nothing here can "
                f"choose a key for it.") from exc
        return (config.DEVICE_REQUIRED["kind"],
                f"deviceless: the backend reports {kinds}, their rule refuses "
                f"a host platform, so the key is config.DEVICE_REQUIRED, the "
                f"device the run asserts before any cell")
