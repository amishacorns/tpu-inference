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

"""Pull request 3040's fused kernel, with the combine folded into it.

The released package at its pin, entered at its own layer entry point
(fused_moe_rs.py:449 fused_moe_func_rs), on the operands every other arm
here takes: one weight master, per-expert eight-bit weights, one scale per
output channel, tokens and router logits sharded over the expert axis.

THE ROUTER IS INSIDE THE WINDOW, and that is why the logits go in
POSITIONALLY: the package's entry takes gating_output as its eighth
positional operand and routes in-program. Handing it a precomputed
selection would be a different program under the same name.

UNTUNED BY THE PACKAGE'S OWN STATEMENT: its notes say the device-specific
block-size tables were removed for release and the chooser returns
caller-supplied defaults. Every number this arm produces is an untuned
number and the artifact has to say so.

NOT DEPENDENCY FREE: the package imports two modules from a serving tree
(fused_moe_rs.py:28-29), so a tree is on the path even though the kernels are
its own. What it reads there is the expert and MLP axis names, at import
(:35-36), so the attention-DP override the production arms install cannot
reach this arm.

THE PACKAGE'S SECOND KERNEL IS DELIBERATELY NOT BOUND: its own notes call it
an instructional reference rather than a complete layer, and the pin carries
only the files this arm imports.
"""

import functools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import _sources  # noqa: E402

REQUIRES_EXTERNAL = True

_PIN = "pr3040"
_DATA, _MODEL = "data", "model"
_TOKENS, _LOGITS, _SCALES = "bfloat16", "float32", "float32"
_ACT, _SCORING = "silu", "softmax"
_SERVING_IMPORT = "tpu_inference.layers.common.sharding"
_SWITCHES = ("TPU_MOE_ENABLE_SP", "TPU_MOE_FP8_OUTPUT_COMM")


def _layer(x, logits, w13, w2, s13, s2, *, entry, mesh, topk, fp8_comm,
           exit_spec):
    import jax
    from jax.sharding import NamedSharding
    out = entry(x, w13, w2, s13, s2, None, None, logits,
                topk=topk, renormalize=True, mesh=mesh, activation=_ACT,
                scoring_fn=_SCORING, fp8_post_gather=fp8_comm)
    # The exit is the layout their program actually ends on: owner-rank rows
    # with their sequence-parallel switch on, replicated with it off
    # (fused_moe_rs.py:366-368). Constraining a replicated exit to the
    # sharded layout would add a scatter their program does not perform.
    return jax.lax.with_sharding_constraint(
        out, NamedSharding(mesh, exit_spec))


def _restore_env(name, before):
    """One environment variable back to what it held, unset included."""
    if before is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = before


def bind(tree_path, shape, profile_flags=None):
    # THE TUNED VARIANT'S TWO SETTINGS, both their own. Their sequence
    # parallel switch is an environment read at trace time
    # (fused_moe_rs.py:79, :365), so a tuned cell sets it here at bind: the
    # value is then the arm's own declared act, recorded as arm-set by the
    # post-bind quarantine, never something the shell happened to hold.
    # Their FP8 combine is a static keyword of the entry (fused_moe_rs.py:466,
    # :27) and only engages with sequence parallel off (:370-371).
    #
    # THE OVERRIDE IS RESTORED WHEN THE BIND DOES NOT COMPLETE. The variable
    # is process global, and a bind that raises past the set -- a pin
    # mismatch, an axis-name leak, one of their own guards -- would otherwise
    # leave it in force for whatever that process binds next, which publishes
    # a configuration nobody named under another arm's name. A bind that
    # COMPLETES keeps it, because the program it hands back has not been
    # traced yet and their switch is read when it is.
    tuned = dict(profile_flags or {})
    sp_override = tuned.pop("TPU_MOE_ENABLE_SP", None)
    before = os.environ.get("TPU_MOE_ENABLE_SP")
    if sp_override is not None:
        os.environ["TPU_MOE_ENABLE_SP"] = str(sp_override)
    bound = False
    try:
        result = _bind(tree_path, shape, tuned, sp_override)
        bound = True
        return result
    finally:
        if sp_override is not None and not bound:
            _restore_env("TPU_MOE_ENABLE_SP", before)


def _bind(tree_path, shape, tuned, sp_override):
    fp8_comm = bool(tuned.pop("fp8_post_gather", False))
    # A tuned cell may name a sibling pin: the same package at the same
    # commit plus a hand-written block-size entry, standing in for the
    # device tables the release removed. The pin decides the bytes either
    # way; the cell's flags carry which pin it was.
    pin_name = str(tuned.pop("pin", _PIN))
    fetched = _sources.fetch_and_verify(names=[pin_name])[pin_name]
    tree = os.path.realpath(tree_path or _sources.own_tree())
    # The serving tree first: the package's module-level imports need it.
    sh = _sources.import_from_tree(tree, _SERVING_IMPORT)
    if fetched["package_root"] not in sys.path:
        sys.path.insert(0, fetched["package_root"])
    import importlib
    name = _sources.load_pins()[pin_name]["package_name"]
    pkg = importlib.import_module(name)
    _sources.assert_under(pkg, fetched["package_root"])
    frs = importlib.import_module(name + ".fused_moe_rs")
    root = fetched["root"]
    sources = [os.path.join(root, rel) for rel in sorted(fetched["files"])]
    E, H, IM = shape["experts"], shape["hidden"], shape["intermediate"]
    # The scale layout the general path hands the same kernels, read off the
    # master rather than written down: [experts, blocks, 1, channels], one
    # block for the per-channel forms and one per contraction block for a
    # four-bit shape.
    s13_shape, s2_shape = _sources.scale_shapes(shape)
    # WHICH AXIS NAMES THIS ARM RESOLVED, and whether anything overrode them.
    # The package reads the expert and MLP-data axes at import
    # (fused_moe_rs.py:35-36) and resolves the expert axis per mesh at :360;
    # it never reads the attention-data axis, which is the one the pair-family
    # arms override, so no topology of ours can reach it. The override
    # table is recorded anyway rather than asserted empty, because an
    # owner-default arm has to be able to SHOW that rather than claim it.
    axis_reads = {"EXPERT": frs.EXPERT, "MLP_DATA": frs.MLP_DATA}
    overrides = dict(getattr(sh.ShardingAxisName, "_overrides", {}) or {})
    leaked = sorted(k for k in overrides if k in axis_reads)
    if leaked:
        raise _sources.HarnessRefusal(
            f"the axis names this package reads are overridden in this "
            f"process ({leaked}), so the arm would measure a topology its own "
            f"configuration does not name. Bind it in its own process.")
    state = {}

    def topo():
        if "mesh" not in state:
            from jax.sharding import NamedSharding
            from jax.sharding import PartitionSpec as P
            mesh = _sources.make_mesh((_DATA, _MODEL), shape["ep"])
            # The exit their program ends on follows their own switch, read
            # after any tuned override was set: owner-rank rows with
            # sequence parallel on, replicated with it off.
            exit_spec = (P(_MODEL, None) if frs.enabled_tpu_sp() else P())
            state.update(mesh=mesh, experts=NamedSharding(mesh, P(_MODEL)),
                         tok=NamedSharding(mesh, P(_MODEL, None)),
                         run=_sources.jit(functools.partial(
                             _layer, entry=pkg.fused_moe_func_rs, mesh=mesh,
                             topk=shape["top_k"], fp8_comm=fp8_comm,
                             exit_spec=exit_spec)))
        return state

    def weights():
        if "w" not in state:
            st, w = topo(), _sources.weight_master(shape)
            e = st["experts"]
            state["w"] = (
                _sources.put(w["w13"], (E, H, 2 * IM),
                             shape["weight_dtype"], e),
                _sources.put(w["w2"], (E, IM, H), shape["weight_dtype"], e),
                _sources.put(w["s13"], s13_shape, _SCALES, e),
                _sources.put(w["s2"], s2_shape, _SCALES, e))
        return state["w"]

    def prepare(case):
        st, b = topo(), int(case.batch)
        return (_sources.put(case.x, (b, H), _TOKENS, st["tok"]),
                {"logits": _sources.put(case.logits, (b, E), _LOGITS,
                                        st["tok"])})

    def call(x, gating, *w):
        # Weights as jit arguments, never closed over (the constant-bake
        # failure; see production_pair.call).
        return topo()["run"](x, gating["logits"], *w)

    def build_check(batch):
        st = topo()
        b, e = int(batch), st["experts"]
        try:
            _sources.lower_for_device(
                st["run"],
                _sources.spec((b, H), _TOKENS, st["tok"]),
                _sources.spec((b, E), _LOGITS, st["tok"]),
                _sources.spec((E, H, 2 * IM), shape["weight_dtype"], e),
                _sources.spec((E, IM, H), shape["weight_dtype"], e),
                _sources.spec(s13_shape, _SCALES, e),
                _sources.spec(s2_shape, _SCALES, e))
        except _sources.Refusal:
            raise
        except Exception as exc:
            # The package's own guards refuse here: the row-count ceiling
            # (fused_moe_rs.py:122-129), the selections-per-block multiple
            # (:478) and the scoring function it accepts (:69-74). Their
            # words, attributed to the kernel.
            raise _sources.owner_refusal(exc) from exc

    return {
        "call": call, "prepare": prepare, "operands": weights,
        "build_check": build_check,
        "contract": _sources.contract(
            # The package's own kernel scope name.
            ("gmm_v2_fused_rs", "gmm_fused_rs", "fused_rs"), True,
            # The caller-side all-gather at the kernel driver's shard-map
            # boundary.
            ("all-gather",),
            # No post-kernel combine. The exchange is per row, inside the
            # kernel, which is the design claim.
            "none", ("logits",)),
        "source_hash": _sources.source_hash(sources, root),
        "tree_sha": fetched["commit"], "tree": root, "pin": pin_name,
        # The top-level module this arm resolves: the released package under
        # its own importable name, whose copy lives inside the pinned
        # checkout so the source gate sees it under the tree the cell
        # names. The serving tree's own namespace is a dependency of the
        # package, not the source this arm measures.
        "namespace": name, "sources": sources,
        # The environment this arm's own code reads: the package's two
        # switches, which change WHICH PROGRAM it is, and the variable that
        # can name an already-present checkout. The driver quarantines
        # these per cell, and declaring them here is what makes the
        # quarantine cover the switches that decide which program runs.
        "env_reads": list(_SWITCHES) + ["BENCHOFF_SRC_" + pin_name.upper()],
        "flags": dict(tuned, **{
            "fp8_post_gather_requested": fp8_comm,
            "tpu_moe_enable_sp_override": sp_override,
            "sequence_parallel_resolved": bool(frs.enabled_tpu_sp()),
            "fp8_output_comm_resolved": bool(
                frs._enable_fp8_output_comm_from_env()),
            "axis_names_read": {k: list(v) if isinstance(v, tuple) else v
                                for k, v in axis_reads.items()},
            "topology_overrides_in_force": sorted(overrides),
            "env_switches": {k: os.environ.get(k, "<unset>")
                             for k in _SWITCHES}}),
        "provenance": {
            "entry": "fused_moe_rs.py:449 fused_moe_func_rs",
            "main_kernel": "gmm_fused_rs_nodedup.py gmm_v2_fused_rs",
            "second_kernel_not_bound": "the package's own notes call it an "
                                       "instructional reference rather than a "
                                       "complete layer",
            "untuned": "the package's notes: the device-specific tuned "
                       "block-size tables were removed for release and the "
                       "chooser returns caller-supplied defaults. Every "
                       "number from this arm is an untuned number",
            "serving_tree_for_two_imports": tree,
            "sequence_parallel_is": (
                "WHAT IT CHANGES ON THIS MESH, which is not what it changes "
                "in general. The switch reads the MLP-data axis, and on this "
                "one-by-eight mesh that axis has extent one, so the driver's "
                "token in_spec is a replicated one either way "
                "(fused_moe_rs.py:366-367: P(MLP_DATA) with the switch on, "
                "P() with it off) and tokens enter REPLICATED in both. What "
                "the switch moves here is the EXIT: on, the output spec is "
                "the combined data-and-expert axes, so the kernel's scattered "
                "exit ends the layer on owner ranks; off, the output is "
                "replicated and an explicit gather runs in the body. On a "
                "mesh whose MLP-data axis has extent above one the switch "
                "would also decide the entry, and this arm's own numbers "
                "would then describe a different program."),
            "axis_names_read": {k: list(v) if isinstance(v, tuple) else v
                                for k, v in axis_reads.items()},
            "topology_overrides_in_force": sorted(overrides),
            "boundary_gather_is_ours": (
                "the all-gather this arm's contract declares is the CALLER'S: "
                "on this mesh the driver wants replicated tokens and this kit "
                "hands it the rank-sharded entry every other arm takes, so "
                "the boundary gathers them. A serving mesh with a real "
                "MLP-data extent would not pay it."),
            "entry_layout": {
                "x": f"P('{_MODEL}', None) {_TOKENS} [tokens, hidden]",
                "logits": f"P('{_MODEL}', None) {_LOGITS} [tokens, experts] "
                          f"passed positionally, so the package's own router "
                          f"runs inside the window"},
        },
    }
