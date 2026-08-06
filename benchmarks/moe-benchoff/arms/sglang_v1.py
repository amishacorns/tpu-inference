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
"""SGLang-JAX Fused MoE V1, the kernel their own stack routes this model to.

Their configuration code holds a short list of architectures allowed on
their second-generation kernel, this model's architecture is not among them,
and a one-member set forces it onto their first-generation fused
expert-parallel path. So this is the kernel their stack would run this model
on, and their newer kernel is a best-supported reading rather than a
production one.

NAME COLLISION, stated because two projects ship a first-generation kernel:
this is SGLang's. Its scope name writes three values in the token-block
group where the other writes two.

The frame both their arms share is in _sglang.py; this file is the entry
call, the selector, the scale layout and the contract.

  :2647-2667  the scale layout
  :2670-2688  the entry call
  :2704-2725  the fields a cell carries about their selection

THE SCALE LAYOUT IS A LAYOUT CHANGE, NOT A REQUANTIZATION. This kit holds
one scale per output channel; their kernel wants one per (reduction block,
output channel). Repeating the same value across the blocks leaves the
dequantized weights unchanged to the last bit, and it is what their own
layer does structurally.

THE REFUSAL GUARD STAYS. A count their table tunes for this shape family and
this mirror cannot resolve is refused by name rather than handed their
default, because a default that looked reasonable is how a row ends up
reporting a configuration nobody selected.
"""

import functools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import _sglang  # noqa: E402
import _sources  # noqa: E402

REQUIRES_EXTERNAL = True

_KERNEL = "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py"
_TABLE = "python/sgl_jax/srt/kernels/fused_moe/v1/tuned_block_configs.py"
_LAYER = "python/sgl_jax/srt/layers/fused_moe.py"
_MODULE_KEY = "sgl_v1_kernel"
_TABLE_NAME = "TUNED_BLOCK_CONFIGS"
_DEFAULT_NAME = "DEFAULT_FUSED_MOE_BLOCK_CONFIG"
_QUANT_NAME = "wsz"                       # their layers/fused_moe.py:219-223
# Their value order, from their own table comment (v1/tuned_block_configs.py
# :39-41).
_FIELDS = ("bt", "bf", "bd1", "bd2", "bts", "btc", "bfc", "bd1c", "bd2c",
           "bse")
# Their model renormalizes the selection weights.
_RENORM = True


def _key(shape, batch, weight_dtype):
    """Their ten-field table key without the device (their
    get_simplified_key, v1/tuned_block_configs.py:251-284). No shared expert
    and no grouped selection, which is what their model constructs here."""
    return (_sglang.TOKENS, weight_dtype, int(batch), shape["experts"],
            shape["top_k"], shape["hidden"], shape["intermediate"],
            shape["ep"], False, False)


def _family(key):
    return key[:2] + key[3:]


def _selector(mod, root, shape, batch, device_key, weight_dtype, qbk):
    """The configuration THEIR selector returns at one count, and its origin.

    Their lookup, their default, and their own effective_for clamps, which
    their kernel applies to whatever it is handed (v1/kernel.py:3378-3384),
    so handing it this answer builds the program the None path would build.
    """
    import jax.numpy as jnp
    if int(batch) % int(shape["ep"]):
        raise _sources.ShapeConstraintRefusal(
            f"their selector refuses num_tokens={batch}, which is not "
            f"aligned to ep_size={shape['ep']}")
    table = _sources.read_literal(os.path.join(root, _TABLE), _TABLE_NAME)
    key = _key(shape, batch, weight_dtype)
    per_device, wild = table.get(device_key) or {}, table.get("*") or {}
    hit = per_device.get(key) or wild.get(key)
    if hit is not None:
        raw, origin = dict(zip(_FIELDS, hit)), "owner-selector-tuned"
    else:
        tuned_family = [k for k in list(per_device) + list(wild)
                        if _family(k) == _family(key)]
        if tuned_family:
            # A NAMED CONSERVATIVE DEVIATION, not their behaviour: THEIR code
            # would run their default here without complaint. This refuses
            # instead, because a count sitting between tuned neighbours would
            # publish a default under a row every other count of which is
            # tuned, and that reads as one configuration when it is two. The
            # deviation is recorded on the refusal so the cell says which
            # counts their table does tune.
            raise _sources.ShapeConstraintRefusal(
                f"their table tunes this shape at "
                f"{sorted(k[2] for k in tuned_family)} "
                f"and not at {batch}. THEIR code would take their default "
                f"here; this kit refuses instead rather than render a "
                f"defaulted count inside a tuned row.")
        raw = _sources.read_call_kwargs(os.path.join(root, _TABLE),
                                        _DEFAULT_NAME)
        origin = "owner-selector-default"
    cfg = mod.FusedMoEBlockConfig(**raw).effective_for(
        num_tokens=int(batch), ep_size=int(shape["ep"]),
        dtype=jnp.dtype(_sglang.TOKENS), quant_block_k=qbk,
        intermediate_size=int(shape["intermediate"]))
    return cfg, origin, raw


def _token_clamp(raw, batch, ep):
    """Their token-tile clamp, declared once more and compared at every count.

    Only the token tiles: the width reduction and the quantization-group
    alignment are their code's alone and are not restated anywhere here.
    """
    import math
    local = int(batch) // int(ep)
    bt = math.gcd(min(raw["bt"], local), local)
    bts = bt if raw.get("bts") is None else min(raw["bts"], bt * int(ep))
    return {"bt": bt, "bts": bts, "btc": min(raw["btc"], bts)}


def _layer(x, tw, ti, w1, w2, w3, s1, s2, s3, block, *, mod, mesh, topk, qbk):
    return mod.fused_ep_moe(
        mesh, x, w1, w2, w3, tw, ti, topk, act_fn=_sglang.ACT,
        block_config=block, quant_block_k=qbk,
        w1_scale=s1, w2_scale=s2, w3_scale=s3,
        renormalize_topk_logits=_RENORM, dp_axis_name=_sglang.DATA,
        tp_axis_name=_sglang.TENSOR)


def bind(tree_path, shape, profile_flags=None):
    del tree_path  # this arm binds an external pin, never a tree of ours
    _sglang.assert_weight_form(shape)  # no four-bit form exists here
    fetched, root, mod = _sglang.load(_KERNEL, _MODULE_KEY)
    device_key, key_from = _sources.owner_device_key(root, _sglang.RULE)
    # THE QUANTIZATION BLOCK IS THEIRS, AND SO IS THE BRANCH IT COMES FROM.
    # Their layer picks it in one expression (layers/fused_moe.py:219-223):
    # a checkpoint that carries its own block-wise weight_block_size uses
    # THAT value, and only a per-channel checkpoint falls to the constant
    # this reads. Every shape in this kit's registry is per-channel fp8, so
    # the constant is the value their stack would use here -- but a
    # block-quantized checkpoint would make this read the wrong number, and
    # the mirror check below is what would catch the expression changing
    # shape rather than the branch being the wrong one.
    qbk = _sources.read_only_int(os.path.join(root, _LAYER), _QUANT_NAME)
    wdt = shape["weight_dtype"]
    H, IM = shape["hidden"], shape["intermediate"]
    if H % qbk or IM % qbk:
        raise _sources.ShapeConstraintRefusal(
            f"their kernel needs the reduction width to divide by {qbk}, "
            f"which their own layer sets for this kernel; hidden {H} and "
            f"per-expert width {IM} do not")
    blocks = (H // qbk, IM // qbk)
    sources = [os.path.join(root, _KERNEL), os.path.join(root, _TABLE)]
    resolved = {int(b): _selector(mod, root, shape, b, device_key, wdt, qbk)
                for b in _sources.config.BATCHES}

    def block_for(batch):
        if int(batch) not in resolved:
            raise _sources.ShapeConstraintRefusal(
                f"their selector was mirrored at {list(resolved)} and this "
                f"count is {batch}: a count outside config.BATCHES has no "
                f"resolved configuration and this arm will not invent one")
        return resolved[int(batch)][0]

    def verify_mirror():
        def extra(proot, detail):
            pqbk = _sources.read_only_int(os.path.join(proot, _LAYER),
                                          _QUANT_NAME)
            detail["quant_block_k"] = qbk
            detail["quant_block_k_pinned"] = pqbk
            if pqbk != qbk:
                detail["problems"].append(
                    f"their layer sets a quantization block of {pqbk} at the "
                    f"pin and this binding ran {qbk}")
        return _sglang.mirror(
            _KERNEL, _MODULE_KEY, _FIELDS,
            lambda pmod, proot, batch: _selector(
                pmod, proot, shape, batch, device_key, wdt,
                _sources.read_only_int(os.path.join(proot, _LAYER),
                                       _QUANT_NAME)),
            lambda raw, batch: _token_clamp(raw, batch, shape["ep"]),
            {b: resolved[b][0] for b in resolved}, device_key, key_from,
            extra=extra)

    bound = _sglang.binding(
        shape,
        lambda mesh: functools.partial(_layer, mod=mod, mesh=mesh,
                                       topk=shape["top_k"], qbk=qbk),
        block_for, scale_blocks=blocks)
    bound.update({
        "verify_mirror": verify_mirror,
        "contract": _sources.contract(
            # Their scope name (v1/kernel.py:3469-3475); three values in the
            # token-block group are what separate it from the other project's
            # first-generation kernel.
            ("fused-moe-k_",), False,
            # THEIR OWN metadata all-gather, in XLA rather than in the kernel:
            # their use_jax_allreduce_metadata defaults to true and takes that
            # path at any expert parallelism above one (v1/kernel.py:3242,
            # reached from :3422). It is their collective, not one this
            # harness inserts, and it is inside the window, so it is declared.
            # The dispatch and the combine themselves are inside their pallas
            # program.
            ("all-gather",), "none", ("idx", "weights")),
        "source_hash": _sources.source_hash(sources, root),
        "tree_sha": fetched["commit"], "tree": root, "pin": _sglang.PIN,
        "namespace": _MODULE_KEY, "sources": sources,
        # Their first-generation kernel reads NO environment at all, and
        # the pinned file holds no os.environ and no getenv. The only
        # variable this arm's own code reads is the one that can name an
        # already-present checkout.
        "env_reads": ["BENCHOFF_SRC_" + _sglang.PIN.upper()],
        "flags": _sglang.flags(_FIELDS, resolved, {
            "block_config_note": "their selector's answer before their "
                                 "kernel's own clamps, which the scope name "
                                 "shows",
            "quant_block_k": qbk, "renormalize_topk_logits": _RENORM,
            "device_key": device_key, "device_key_from": key_from}),
        "provenance": {
            "entry": "sgl_jax/srt/kernels/fused_moe/v1/kernel.py:3301 "
                     "fused_ep_moe",
            "router": "EXTERNAL: their model computes the selection and "
                      "passes it in, so it is outside the window",
            "scale_layout": f"one scale per (reduction block, output channel):"
                            f" {blocks[0]} blocks on hidden, {blocks[1]} on "
                            f"the per-expert width, the same value repeated, "
                            f"which leaves the dequantized weights unchanged",
            "quant_block_k_read_from": f"{_LAYER} {_QUANT_NAME}",
            "metadata_all_gather": "theirs, in XLA: their "
                                   "use_jax_allreduce_metadata default takes "
                                   "an all-gather of the per-block expert "
                                   "counts (v1/kernel.py:3242) at any expert "
                                   "parallelism above one",
            "entry_layout": {
                "x": f"P(('{_sglang.DATA}','{_sglang.TENSOR}'), None) "
                     f"{_sglang.TOKENS} [tokens, hidden]",
                "idx": f"P(('{_sglang.DATA}','{_sglang.TENSOR}'), None) "
                       f"{_sglang.IDX} [tokens, top_k]",
                "weights": f"P(('{_sglang.DATA}','{_sglang.TENSOR}'), None) "
                           f"{_sglang.WTS} [tokens, top_k]"},
        },
    })
    return bound
