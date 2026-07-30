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

"""The shipped fused expert-parallel kernel of pull request 3288.

The kernel package on this branch's own tree, entered at its own public
layer function with the operands and the capacity the SERVING ADAPTER
passes, so the cell measures the kernel as it is served rather than under
hand-picked sizes.

THE TREE IS THIS CHECKOUT WHERE THIS CHECKOUT CARRIES THE KERNEL, and the
arm's own pin where it does not. The kernel package is not on upstream main,
so a clone of upstream has no module for this arm to import; rather than
fail at the import, the arm falls back to its pins.json entry, fetched and
hash verified like any external, and binds that checkout. Preference stays
with the live tree, so a kit sitting on the serving tree measures that tree.

THE CAPACITY IS READ FROM THE TREE, not written here. It is the adapter's
own constant, tpu_inference/layers/common/moe_fused_ep.py _TILE_M, parsed
out of the adapter source rather than imported, so binding the kernel never
drags the whole serving adapter in. An unset environment therefore
reproduces the as-served configuration exactly, and a tree whose adapter
moves its constant moves this arm with it.

THE EXIT IS CONSTRAINED, on the owner-rank sharding at this kernel's own
axis, because every arm in the comparison has to end on the same layout. An
arm returning the layer's output bare leaves its exit layout to the
compiler, and two arms compared under two exit layouts are not comparable.
"""

import functools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import _sources  # noqa: E402

REQUIRES_EXTERNAL = False

_PIN = "fused_ep_v2"
_TOKENS, _LOGITS, _SCALES = "bfloat16", "float32", "float32"
_ACT = "silu"
_TOP = "tpu_inference"
_PKG = _TOP + ".kernels.fused_moe.v2"
_ADAPTER = "tpu_inference/layers/common/moe_fused_ep.py"
_CAPACITY_NAME = "_TILE_M"                      # moe_fused_ep.py:42
_FILES = ("tpu_inference/kernels/fused_moe/v2/layer.py",
          "tpu_inference/kernels/fused_moe/v2/kernel.py",
          "tpu_inference/kernels/fused_moe/v2/host.py",
          "tpu_inference/kernels/fused_moe/v2/router_ops.py")


def _layer(x, logits, w13, w2, s13, s2, *, pkg, mesh, axis, topk, capacity,
           weight_format):
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    # The adapter wraps the call in the kernel mesh's abstract mesh; mirror
    # it so the traced program is the served one.
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        out = pkg.fused_ep_moe_v2(x, w13, w2, s13, s2, logits, topk=topk,
                                  renormalize=True, mesh=mesh,
                                  capacity=capacity, act_fn=_ACT,
                                  weight_format=weight_format)
    return jax.lax.with_sharding_constraint(
        out, NamedSharding(mesh, P(axis, None)))


def bind(tree_path, shape, profile_flags=None):
    import jax.numpy as jnp
    tree, fetched = _sources.tree_for_arm(_PIN, _FILES, tree_path)
    flags = dict(profile_flags or {})
    pkg = _sources.import_from_tree(tree, _PKG)
    capacity = _sources.read_literal(os.path.join(tree, _ADAPTER),
                                     _CAPACITY_NAME)
    weight_format = pkg.weight_format_of_dtype(jnp.dtype(
        shape["weight_dtype"]))
    if weight_format is None:
        raise _sources.ShapeConstraintRefusal(
            f"the kernel takes weight formats {pkg.WEIGHT_FORMAT_NAMES} and "
            f"this shape's weights are {shape['weight_dtype']}")
    axis = pkg.AXIS
    sources = [os.path.join(tree, f) for f in _FILES]
    # Which scale-operand contract the bound tree carries: the landing branch
    # moved per-channel scales to their hardware layout, and its layer entry
    # refuses the old one by naming the accepted shape. One arm serves both
    # trees by reading the layer's own source for that refusal.
    # The marker is a SYMBOL the landing commit introduced, not an error
    # string: the layer's refusal text interpolates its wording at runtime,
    # so grepping for it found nothing in source (the first detection bug).
    with open(os.path.join(tree, _FILES[0])) as fh:
        two_d_scales = "act_scale_slab_rows" in fh.read()
    E, H, IM = shape["experts"], shape["hidden"], shape["intermediate"]

    # WHICH OF THE THREE SCALE LAYOUTS THIS CELL PLACES, decided by the
    # master's own block count and by the bound tree, never by a literal.
    # The master carries [E, blocks, 1, N] for either weight form. One block
    # is a per-channel table, which is the four-axis form the older entry
    # takes and the two-axis form the landing branch takes. More than one is
    # a scale per contraction block, which the layer entry takes in three
    # axes for every tree that has the four-bit form at all (layer.py's
    # per_contraction_block check: ndim 3, [E, blocks, N]). Only singleton
    # axes ever move; the values are the master's.
    def _scale_shape(master, n):
        blocks = master[1]
        if blocks > 1:
            return (E, blocks, n)
        return (E, n) if two_d_scales else (E, 1, 1, n)

    _m13, _m2 = _sources.scale_shapes(shape)
    s13_shape, s2_shape = _scale_shape(_m13, 2 * IM), _scale_shape(_m2, H)
    state = {}

    def topo():
        if "mesh" not in state:
            from jax.sharding import NamedSharding
            from jax.sharding import PartitionSpec as P
            mesh = _sources.make_mesh((axis,), shape["ep"])
            state.update(
                mesh=mesh, experts=NamedSharding(mesh, P(axis)),
                tok=NamedSharding(mesh, P(axis, None)),
                run=_sources.jit(functools.partial(
                    _layer, pkg=pkg, mesh=mesh, axis=axis,
                    topk=shape["top_k"], capacity=capacity,
                    weight_format=weight_format)))
        return state

    def weights():
        if "w" not in state:
            st, w = topo(), _sources.weight_master(shape)
            e = st["experts"]
            # The master serves scales in the four-axis layout; a contract
            # that drops a singleton axis wants the same values reshaped, and
            # the placement asserts shape rather than reshaping for us.
            import numpy as _np
            s13_src = _np.asarray(w["s13"]).reshape(s13_shape)
            s2_src = _np.asarray(w["s2"]).reshape(s2_shape)
            w13_src, w2_src = w["w13"], w["w2"]
            state["w"] = (
                _sources.put(w13_src, (E, H, 2 * IM),
                             shape["weight_dtype"], e),
                _sources.put(w2_src, (E, IM, H), shape["weight_dtype"], e),
                _sources.put(s13_src, s13_shape, _SCALES, e),
                _sources.put(s2_src, s2_shape, _SCALES, e))
        return state["w"]

    def prepare(case):
        st, b = topo(), int(case.batch)
        logits = case.logits
        return (_sources.put(case.x, (b, H), _TOKENS, st["tok"]),
                {"logits": _sources.put(logits, (b, E), _LOGITS, st["tok"])})

    def call(x, gating, *w):
        # Weights as jit arguments, never closed over (the constant-bake
        # failure; see production_pair.call).
        return topo()["run"](x, gating["logits"], *w)

    def build_check(batch):
        st = topo()
        b, e = int(batch), st["experts"]
        _sources.lower_for_device(
            st["run"],
            _sources.spec((b, H), _TOKENS, st["tok"]),
            _sources.spec((b, E), _LOGITS, st["tok"]),
            _sources.spec((E, H, 2 * IM), shape["weight_dtype"], e),
            _sources.spec((E, IM, H), shape["weight_dtype"], e),
            _sources.spec(s13_shape, _SCALES, e),
            _sources.spec(s2_shape, _SCALES, e))

    return {
        "call": call, "prepare": prepare, "operands": weights,
        "build_check": build_check,
        "contract": _sources.contract(
            # The kernel's own scope name, one program per call.
            ("fused_ep_moe_v2",), True,
            # The token all-gather and the routing-blob all-gather the
            # layer performs before the kernel.
            ("all-gather",),
            # No post-kernel combine exists. The weighted rows are pushed
            # to their token owners from inside the kernel, which is this
            # kernel's design claim.
            "none", ("logits",)),
        "source_hash": _sources.source_hash(sources, tree),
        # The commit the pin names when the pin was taken, the live tree's
        # own HEAD when it was not; and the pin name only in the first case,
        # so the source gate checks the executed files against the pinned
        # hashes exactly when the pinned checkout is what ran.
        "tree_sha": (fetched["commit"] if fetched
                     else _sources.tree_sha(tree)),
        "tree": tree, "pin": _PIN if fetched else None,
        # The top-level module this arm resolves, so the source gate can
        # show which checkout its imports came from and a per-configuration
        # child can assert one tree per namespace.
        "namespace": _TOP, "sources": sources,
        # The tree's own envs.py registry is quarantined wholesale by the
        # driver, so this arm declares no environment of its own. Naming the
        # kernel's switches here would cover only the ones a reader thought
        # of; the quarantine covers every variable the tree declares.
        "env_reads": [],
        "flags": dict(flags, capacity=capacity, two_d_scales=two_d_scales,
                      weight_format=str(weight_format)),
        "provenance": {
            "entry": "tpu_inference/kernels/fused_moe/v2/layer.py "
                     "fused_ep_moe_v2",
            "capacity": capacity,
            "capacity_read_from": f"{_ADAPTER} {_CAPACITY_NAME}",
            "capacity_is": "the adapter's own tile height, which is what "
                           "the serving path passes. A cell that ran another "
                           "value says so here",
            "block_and_stride": "left for the kernel to derive, which is "
                                "what the serving adapter does",
            "weight_format_derived_by": "the package's own "
                                        "weight_format_of_dtype",
            "axis": axis,
            "exit": f"P('{axis}', None) constrained, so this arm ends on "
                    f"the same owner-rank layout as every other arm",
            "entry_layout": {
                "x": f"P('{axis}', None) {_TOKENS} [tokens, hidden]",
                "logits": f"P('{axis}', None) {_LOGITS} [tokens, experts]"},
        },
    }
