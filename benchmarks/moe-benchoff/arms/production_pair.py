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

"""The production pair path: two grouped matmuls, bound against upstream.

One MoE layer through the serving layer entry, on a tree that carries no
fused grouped matmul at all, so the plain call there IS the two-matmul pair
this comparison's production row means. The tree is a worktree of this
repository at config.UPSTREAM_PIN, made by _sources.tree_worktree when the
manifest names no tree of its own.

Only keywords the bound tree's own fused_moe_func signature has are sent. A
flag requested against a tree that has no such parameter refuses here,
because dropping it silently would publish one program's numbers under
another program's name. Flag values arrive in profile_flags and no value is
named in this file.

TWO FORMS, ONE BUILDER, AND THE PROFILE DECIDES WHICH. Attention
data-parallelism and the scattered combine are one decision upstream, not
two: its own interface sets scatter_results from is_attn_dp(mesh). So the
scatter flag decides the topology here, which gives exactly two coherent
forms -- replicated tokens with a psum combine and no routing all-gather, or
sharded tokens with the routing all-gather and a scattered combine -- and no
way to ask for a mixture of the two. The long comment in bind_family names
the gate at each tree.

This file carries the fused member of the same family too (fused_gmm): the
two differ only in which tree they bind and which expert compute that tree's
plain call is, and one builder keeps them from drifting apart.
"""

import functools
import inspect
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import _sources  # noqa: E402

REQUIRES_EXTERNAL = False

# The ported call pattern's own tokens, none of them a configuration value:
# the mesh axis names the bench mesh has always used (:1135), the element
# types the entry takes (:1142-1143, :1159-1160), and the activation and
# scoring the served layer runs (:1106).
_DATA, _MODEL = "data", "model"
_TOKENS, _LOGITS, _SCALES = "bfloat16", "float32", "float32"
_ACT, _SCORING = "silu", "softmax"
_TOP = "tpu_inference"
_ENTRY = _TOP + ".layers.common.fused_moe_gmm"
# The files a cell of this family executes: the layer entry plus the grouped
# matmul it imports (fused_moe_gmm.py:25 upstream, :25-29 where the tree also
# carries the fused pair).
_FILES = ("tpu_inference/layers/common/fused_moe_gmm.py",
          "tpu_inference/kernels/megablox/gmm_v2.py")
_FUSED_FILES = ("tpu_inference/kernels/megablox/gmm_fused.py",
                "tpu_inference/kernels/megablox/gmm_v2_fused_support.py")
_FUSED_MODULE = _TOP + ".kernels.megablox.gmm_fused"


def _layer(x, logits, w13, w2, s13, s2, *, fn, mesh, topk, kw, topology,
           exit_spec):
    import jax
    from jax.sharding import NamedSharding

    # THE TOPOLOGY IS SET AT TRACE TIME, not at bind time. The tree reads its
    # axis names while this function traces, and one process binds several
    # arms, so an arm that set them at bind time could have another arm's
    # topology in force by the time it retraced at a new count.
    topology()
    out = fn(x, w13, w2, s13, s2, None, None, logits,
             topk=topk, renormalize=True, mesh=mesh, use_ep=True,
             activation=_ACT, scoring_fn=_SCORING, **kw)
    # The exit each form actually ends on: under attention data-parallelism
    # the scattered combine leaves every rank holding its own rows, and this
    # asserts that layout; without it the psum combine leaves the result
    # replicated, and this says so instead of slicing a layout serving
    # would not slice.
    return jax.lax.with_sharding_constraint(
        out, NamedSharding(mesh, exit_spec))


def _keywords(fn, flags, fused, tree):
    """The keywords this tree accepts, and the refusals it forces."""
    params = inspect.signature(fn).parameters
    kw, refused = {}, {}
    for name in sorted(flags):
        if name in ("origin", "compiler_options"):
            continue
        if name in params:
            kw[name] = flags[name]
        elif flags[name]:
            raise _sources.HarnessRefusal(
                f"{name}={flags[name]!r} requested and this tree's "
                f"fused_moe_func has no {name} parameter, so the flag would "
                f"be dropped and the cell would carry a configuration it did "
                f"not run. Use a profile this tree's signature accepts, or "
                f"bind a tree that carries the parameter.")
        else:
            refused[name] = ("not in this tree's signature; the flag is off, "
                             "which is what the tree already does")
    if "use_fused_gmm" in params:
        kw["use_fused_gmm"] = fused
        return kw, refused
    try:
        _sources.import_from_tree(tree, _FUSED_MODULE)
        has_fused = True
    except ImportError:
        has_fused = False
    if has_fused != fused:
        raise _sources.HarnessRefusal(
            f"this tree's fused_moe_func has no use_fused_gmm parameter and "
            f"the tree {'carries' if has_fused else 'has no'} "
            f"kernels/megablox/gmm_fused, so its plain call is the "
            f"{'fused' if has_fused else 'unfused'} expert compute, while "
            f"this arm is the {'fused' if fused else 'unfused'} member. Bind "
            f"it to a tree whose plain call is its own program.")
    refused["use_fused_gmm"] = (
        "not in this tree's signature; the tree's plain call is the "
        + ("fused" if has_fused else "unfused") + " expert compute")
    return kw, refused


def bind_family(tree_path, shape, profile_flags, fused, expert_compute):
    tree = os.path.realpath(tree_path or _default_tree(fused))
    flags = dict(profile_flags or {})
    mod = _sources.import_from_tree(tree, _ENTRY)
    fn = mod.fused_moe_func
    kw, refused = _keywords(fn, flags, fused, tree)
    files = _FILES + (_FUSED_FILES if fused else ())
    sources = [os.path.join(tree, f) for f in files]
    E, H, IM = shape["experts"], shape["hidden"], shape["intermediate"]
    # The grouped matmuls take their scales in the master's own four-axis
    # layout, [experts, blocks, 1, channels], at every block count: one block
    # is the per-channel table this family has always placed, and a four-bit
    # shape's block-scaled table differs from it in that axis alone
    # (megablox/gmm_fused.py: "[groups, blocks, 1, N]" with nb dividing the
    # contraction). Read off the master rather than written down, so the two
    # cannot disagree about how many blocks there are.
    s13_shape, s2_shape = _sources.scale_shapes(shape)
    sh = _sources.import_from_tree(tree, _TOP + ".layers.common.sharding")
    # THE TOPOLOGY FOLLOWS THE PROFILE, AND THE COUPLING IS UPSTREAM'S OWN.
    # Upstream's interface sets scatter_results = is_attn_dp(mesh)
    # (layers/vllm/interface/moe.py:110-111), so attention data-parallelism
    # and the scattered combine are ONE decision. Deriving the topology from
    # the scatter flag reproduces that coupling instead of restating it, and
    # it is why no profile name is read here.
    #
    # What each form is, both re-read at the trees this family binds:
    #   scatter off -> the tree's own ATTN_DATA, which on this two-axis mesh
    #     is 'data' with extent one. The routing all-gather is gated on that
    #     extent being above one (upstream fused_moe_gmm.py:629, this branch
    #     :701), so it does not fire; is_attn_dp is false (:638, :710); the
    #     combine is the psum all-reduce (:325, :391) and the result is
    #     replicated. The shard_map takes tokens at P(MLP_DATA, None), which
    #     is 'data' and therefore replicated on this mesh (:703-706, :775-777),
    #     so a replicated entry needs no gather at the boundary either.
    #     Attention data-parallelism upstream is an opt-in serve argument
    #     rather than a code default, so this IS upstream's default form.
    #   scatter on -> ATTN_DATA overridden to both axes, extent eight: the
    #     routing all-gather fires, is_attn_dp is true, the combine is the
    #     psum_scatter onto owner ranks (:317), and tokens enter sharded, so
    #     the boundary gathers them for the same P(MLP_DATA, None) in_spec.
    #
    # A cell that gathered the routing while combining with an all-reduce is
    # a form unmodified upstream cannot produce, and deriving the topology
    # from the scatter flag is what keeps it from being asked for.
    attn_dp = bool(kw.get("scatter_results"))
    state = {}

    def topology():
        """The axis names this form runs under, and nothing sticky.

        reset() first (sharding.py:126-128) because the override is process
        global: without it an arm bound after a scattered-combine arm would
        inherit its topology and measure a form its own flags do not name.
        """
        sh.ShardingAxisName.reset()
        if attn_dp:
            sh.ShardingAxisName.override(ATTN_DATA=(_DATA, _MODEL))
        return {n: getattr(sh.ShardingAxisName, n)
                for n in ("ATTN_DATA", "MLP_DATA", "EXPERT")}

    def topo():
        if "mesh" not in state:
            from jax.sharding import NamedSharding
            from jax.sharding import PartitionSpec as P
            mesh = _sources.make_mesh((_DATA, _MODEL), shape["ep"])
            # The entry and the exit each form actually serves: sharded rows
            # under attention data-parallelism, replicated without it.
            entry = P(_MODEL, None) if attn_dp else P()
            state.update(mesh=mesh, experts=NamedSharding(mesh, P(_MODEL)),
                         tok=NamedSharding(mesh, entry),
                         run=_sources.jit(functools.partial(
                             _layer, fn=fn, mesh=mesh, topk=shape["top_k"],
                             kw=kw, topology=topology, exit_spec=entry)))
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
        st = topo()
        b = int(case.batch)
        # The logits are placed in float32, which is what the host router
        # that feeds the precomputed-routing arms reads (harness.host_top_k),
        # so an in-program router and a precomputed one route on identical
        # values. The cast is here, outside the window.
        return (_sources.put(case.x, (b, H), _TOKENS, st["tok"]),
                {"logits": _sources.put(case.logits, (b, E), _LOGITS,
                                        st["tok"])})

    def call(x, gating, *w):
        # The weights are jit ARGUMENTS, never closed over. A closed-over
        # array compiles into the program as a constant, which bloats the
        # executable and adds constant re-slicing per call that serving
        # never pays.
        return topo()["run"](x, gating["logits"], *w)

    def census_lowering(batch):
        """This arm's layer program, entered with its operands as arguments.

        The same jitted layer a cell measures, lowered with the weights as
        inputs rather than closed over. The serving layer receives its weights
        as arguments, so the serving-path census compares the two sides at
        that boundary: a program that closed its weights over as constants
        differs from serving's in its constants and its convert count without
        differing in anything the layer computes.

        No compiler options here, for the reason _sources.jit gives: the frame
        is the driver's. The frame is the same on both sides of the census, so
        applying it to neither cannot hide a difference between them.
        """
        st = topo()
        b, e = int(batch), st["experts"]
        return _sources.lower_for_device(
            st["run"],
            _sources.spec((b, H), _TOKENS, st["tok"]),
            _sources.spec((b, E), _LOGITS, st["tok"]),
            _sources.spec((E, H, 2 * IM), shape["weight_dtype"], e),
            _sources.spec((E, IM, H), shape["weight_dtype"], e),
            _sources.spec(s13_shape, _SCALES, e),
            _sources.spec(s2_shape, _SCALES, e))

    def build_check(batch):
        census_lowering(batch)

    axes = topology()
    layout = f"P('{_MODEL}', None)" if attn_dp else "P() replicated"
    return {
        "call": call, "prepare": prepare, "operands": weights,
        "build_check": build_check,
        "census_lowering": census_lowering,
        "contract": _sources.contract(
            # Both grouped matmuls carry this prefix and each is its own
            # operation, so the anchor fires once per call either way.
            ("gmm_v2", "gmm_fused"), True,
            # Under attention data-parallelism the window holds the routing
            # all-gather and the token gather at the shard_map boundary.
            # Without it neither exists: the routing gather is gated off and
            # the entry is already the layout the boundary wants.
            ("all-gather",) if attn_dp else (),
            "reduce_scatter" if attn_dp else "all_reduce",
            ("logits",)),
        "source_hash": _sources.source_hash(sources, tree),
        "tree_sha": _sources.tree_sha(tree), "tree": tree,
        # The top-level module this arm resolves, so the source gate can
        # show which checkout its imports came from and a per-configuration
        # child can assert one tree per namespace.
        "namespace": _TOP, "sources": sources,
        # The tree's own envs.py registry is quarantined wholesale by the
        # driver, so this arm declares no environment of its own. Naming a
        # few switches here would cover only the ones a reader thought of,
        # and the quarantine covers every variable the tree declares.
        "env_reads": [],
        "flags": dict(kw, attn_data_parallel=attn_dp,
                      axis_names={k: list(v) if isinstance(v, tuple) else v
                                  for k, v in axes.items()}),
        "provenance": {
            "expert_compute": expert_compute,
            "entry": "tpu_inference/layers/common/fused_moe_gmm.py "
                     "fused_moe_func",
            "keywords_sent": dict(kw), "keywords_refused": refused,
            "topology_follows": "scatter_results, because upstream's own "
                                "interface sets scatter_results = "
                                "is_attn_dp(mesh) (interface/moe.py:110-111)",
            "attn_data_parallel": attn_dp,
            "axis_names_resolved": {k: list(v) if isinstance(v, tuple) else v
                                    for k, v in axes.items()},
            "entry_layout": {
                "x": f"{layout} {_TOKENS} [tokens, hidden]",
                "logits": f"{layout} {_LOGITS} [tokens, experts]"},
            "exit": (f"P('{_MODEL}', None): every rank ends holding its own "
                     f"tokens/{shape['ep']} rows, which is the scattered "
                     f"combine's own output layout"
                     if attn_dp else
                     "P() replicated: the psum combine's own output layout, "
                     "which is what the next layer consumes when attention "
                     "is not data-parallel"),
            "entry_asymmetry": (
                "the two forms enter differently BY CONSTRUCTION and each "
                "renders under its own configuration: the replicated form "
                "pays no gather and carries every token on every rank, the "
                "sharded form carries a rank's own rows and pays the gather. "
                "Neither figure may be quoted as the other's."),
        },
    }


def _default_tree(fused):
    """The tree an arm of this family binds when the manifest names none."""
    if fused:
        return _sources.own_tree()
    return _sources.tree_worktree(_sources.config.UPSTREAM_PIN)


def bind(tree_path, shape, profile_flags=None):
    return bind_family(tree_path, shape, profile_flags, fused=False,
                       expert_compute="two grouped matmuls, unfused")
