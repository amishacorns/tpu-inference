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

# Derived from sglang-jax (Apache-2.0), see docstring citations.
"""What the two SGLang-JAX arms share, in one place so they cannot drift.

Both generations of their fused expert-parallel kernel take the same frame:
the same pinned clone, a copy import under a module key of their own, the
same two-axis mesh and sharding, the same per-expert weight split off this
kit's one weight master, the same precomputed selection, and the same
deviceless build check. What differs is the entry call, the selector, the
weight-scale layout and the window contract, and those stay in the arms.

The scale layout generalizes the two: a block count of one per contraction
axis IS the per-channel layout the second generation takes, and a count
above one repeats the same value across the blocks, which is what the first
generation needs and leaves the dequantized weights unchanged to the last
bit.
"""

import functools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import _sources  # noqa: E402

PIN = "sglang_jax"
RULE = "python/sgl_jax/srt/utils/jax_utils.py"
DATA, TENSOR = "data", "tensor"
TOKENS, IDX, WTS, SCALES = "bfloat16", "int32", "float32", "float32"
ACT = "silu"


def assert_weight_form(shape):
    """Refuse, before anything is fetched, a form neither generation has.

    Both of their kernels dequantize an EIGHT-BIT weight against one scale
    per output channel: the first generation tiles that one value across
    its own quantization blocks, which is why the scale_blocks argument
    below exists, and the second takes the per-channel table as it is.
    Neither carries a four-bit element type and neither takes a scale that
    varies ALONG the contraction, so a block-scaled shape is named and
    refused here. Tiling a block-scaled table the way the per-channel one
    is tiled would repeat a whole contraction block's scale onto rows it
    does not belong to and hand their kernel a table their loader never
    produces.
    """
    if _sources.block_scaled(shape):
        raise _sources.ShapeConstraintRefusal(
            f"their kernels take eight-bit expert weights with one scale "
            f"per output channel, and this shape's weights are "
            f"{shape['weight_dtype']} with one scale per contraction block: "
            f"neither generation of their kernel has a four-bit form")


def load(kernel_rel, module_key, pinned_only=False):
    """Their pinned kernel file, imported as a copy under its own key."""
    fetched = _sources.fetch_and_verify(
        names=[PIN], allow_env_root=not pinned_only)[PIN]
    root = fetched["root"]
    mod = _sources.module_from_file(os.path.join(root, kernel_rel), module_key)
    _sources.assert_under(mod, root)
    return fetched, root, mod


def binding(shape, make_layer, block_for, scale_blocks=(1, 1)):
    """The operands, the placement, the call and the build check.

    make_layer(mesh) returns the function to jit, taking the operands in the
    order below. block_for(batch) returns the block configuration the arm's
    own selector resolved. scale_blocks is (blocks on hidden, blocks on the
    per-expert width) for the weight-scale layout their kernel wants.
    """
    import numpy as np
    E = shape["experts"]
    H, IM = shape["hidden"], shape["intermediate"]
    wdt = shape["weight_dtype"]
    n1, n2 = scale_blocks
    state = {}

    def topo():
        if "mesh" not in state:
            from jax.sharding import NamedSharding
            from jax.sharding import PartitionSpec as P
            mesh = _sources.make_mesh((DATA, TENSOR), shape["ep"])
            state.update(mesh=mesh,
                         shard=NamedSharding(mesh, P((DATA, TENSOR))),
                         run=_sources.jit(make_layer(mesh)))
        return state

    def weights():
        if "w" not in state:
            st, w = topo(), _sources.weight_master(shape)
            s = st["shard"]

            def tile(src, blocks):
                return np.repeat(np.asarray(src), blocks, axis=1)
            state["w"] = (
                _sources.put(w["w13"][:, :, :IM], (E, H, IM), wdt, s),
                _sources.put(w["w2"], (E, IM, H), wdt, s),
                _sources.put(w["w13"][:, :, IM:], (E, H, IM), wdt, s),
                _sources.put(tile(w["s13"][:, :, :, :IM], n1), (E, n1, 1, IM),
                             SCALES, s),
                _sources.put(tile(w["s2"], n2), (E, n2, 1, H), SCALES, s),
                _sources.put(tile(w["s13"][:, :, :, IM:], n1), (E, n1, 1, IM),
                             SCALES, s))
        return state["w"]

    def prepare(case):
        st, b, k = topo(), int(case.batch), shape["top_k"]
        s = st["shard"]
        return (_sources.put(case.x, (b, H), TOKENS, s),
                {"idx": _sources.put(case.idx, (b, k), IDX, s),
                 "weights": _sources.put(case.weights, (b, k), WTS, s)})

    def call(x, gating, *w):
        # Weights as jit arguments, never closed over (the constant-bake
        # failure; see production_pair.call). The block configuration stays
        # in-call: it is a host-side value derived from the batch, not an
        # operand.
        return topo()["run"](x, gating["weights"], gating["idx"], *w,
                             block_for(x.shape[0]))

    def build_check(batch):
        st, b, k = topo(), int(batch), shape["top_k"]
        sp = functools.partial(_sources.spec, sharding=st["shard"])
        try:
            _sources.lower_for_device(
                st["run"],
                sp((b, H), TOKENS), sp((b, k), WTS), sp((b, k), IDX),
                sp((E, H, IM), wdt), sp((E, IM, H), wdt), sp((E, H, IM), wdt),
                sp((E, n1, 1, IM), SCALES), sp((E, n2, 1, H), SCALES),
                sp((E, n1, 1, IM), SCALES), block_for(b))
        except _sources.Refusal:
            raise
        except Exception as exc:
            # Their kernel refusing this count in its own words.
            raise _sources.owner_refusal(exc) from exc

    return {"call": call, "prepare": prepare, "operands": weights,
            "build_check": build_check}


def mirror(kernel_rel, module_key, fields, resolve, declared, bound,
           device_key, key_from, extra=None):
    """Their selector's answer, re-resolved from the FETCHED source at the pin.

    The convenience clone an environment may name is not consulted: a mirror
    checked against a path someone can edit proves nothing. The owner file
    hashes are in the result, so a moved pin reads differently from a drifted
    mirror. Three things are compared at every count: their own clamp against
    the second declaration of the part of it this kit restates, and the
    configuration the binding holds against the one the pinned source
    resolves.
    """
    pinned, proot, pmod = load(kernel_rel, module_key + "_pinned",
                               pinned_only=True)
    detail = {"pin": PIN, "commit": pinned["commit"],
              "file_sha256": pinned["files"], "device_key": device_key,
              "device_key_from": key_from, "per_batch": {}, "problems": []}
    for batch in sorted(bound):
        cfg, origin, raw = resolve(pmod, proot, batch)
        mine = {f: getattr(cfg, f) for f in fields}
        want = declared(raw, batch)
        detail["per_batch"][str(batch)] = {
            "origin": origin, "resolved": mine, "declared": want,
            "table_holds_key": origin.endswith("tuned")}
        for field, value in want.items():
            if mine[field] != value:
                detail["problems"].append(
                    f"at {batch} their clamp returns {field}={mine[field]} "
                    f"and the declared rule says {value}")
        if {f: getattr(bound[batch], f) for f in fields} != mine:
            detail["problems"].append(
                f"at {batch} the bound configuration is not the one the "
                f"pinned source resolves")
    if extra is not None:
        extra(proot, detail)
    return not detail["problems"], detail


def flags(fields, resolved, extra):
    """The resolved values in force, per count, plus the arm's own switches."""
    out = {
        "block_config_fields": list(fields),
        "block_config_by_batch": {
            str(b): [getattr(resolved[b][0], f) for f in fields]
            for b in sorted(resolved)},
        "block_config_origin_by_batch": {str(b): resolved[b][1]
                                         for b in sorted(resolved)},
    }
    out.update(extra)
    return out
