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
"""SGLang-JAX Fused MoE V2, at the configuration their own selector returns.

Their second-generation fused expert-parallel kernel, imported as a
standalone copy of the pinned file (their package pins a different jax; the
kernel file imports nothing from their package), at their own block
configuration, on the one weight master every arm here multiplies. The frame
both their arms share is in _sglang.py; this file is the entry call, the
selector and the contract.

THEIR ROUTER IS OUTSIDE THE WINDOW, and that is load-bearing: their serving
layer receives the selection precomputed per rank from the upstream router,
so this arm consumes idx and weights and its contract says router_in_window
is false. The render layer prints that as table mechanics, so the
declaration has to be exact rather than convenient.

THE CONFIGURATION IS THEIRS, RESOLVED FROM THEIR OWN FILES. Their kernel
resolves the block configuration itself when handed None, but that path runs
a package-relative import a copy import cannot satisfy, so the answer is
resolved here from their pinned table and their pinned default and handed
over explicitly. Their kernel then applies its own effective_for clamps to
whatever it is given, so this builds the program the None path would have
built. Nothing in this file chooses a configuration: the table, the default,
the key rule and the clamp are all read or called from their source.
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

_KERNEL = "python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py"
_TABLE = "python/sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py"
_MODULE_KEY = "sgl_v2_kernel"
_TABLE_NAME, _DEFAULT_NAME = "TUNED_BLOCK_CONFIGS", "DEFAULT_V2_BLOCK_CONFIG"
_FIELDS = ("bt", "bf", "btc", "bse", "bts")
# Their two shipped switches, at their own defaults for this row: the
# scaled dot the per-channel scales require, and the eight-bit token mode
# their defaults leave off.
_DIRECT_SCALED_DOT, _ACT_QUANT = True, False


def _key(shape, batch, weight_dtype):
    """Their table key without the device, by their own construction
    (v2/tuned_block_configs.py:80-117). The last three fields are false for
    this row: no shared expert, no grouped selection, no token quantization."""
    return (_sglang.TOKENS, weight_dtype, int(batch), shape["experts"],
            shape["top_k"], shape["hidden"], shape["intermediate"],
            shape["ep"], False, False, _ACT_QUANT)


def _selector(mod, root, shape, batch, device_key, weight_dtype):
    """The configuration THEIR selector returns at one count, and its origin.

    Their lookup takes the eleven-field key, falls back to their legacy
    ten-field key, then to their own default; the clamp is their
    effective_for, called on their own class (v2/kernel.py:43-63).
    """
    table = _sources.read_literal(os.path.join(root, _TABLE), _TABLE_NAME)
    key = _key(shape, batch, weight_dtype)
    per_device = table.get(device_key) or {}
    hit = (per_device.get(key) or (table.get("*") or {}).get(key)
           or per_device.get(key[:-1]) or (table.get("*") or {}).get(key[:-1]))
    if hit is not None:
        raw = dict(zip(_FIELDS, hit))
        origin = "owner-selector-tuned"
    else:
        raw = _sources.read_call_kwargs(os.path.join(root, _TABLE),
                                        _DEFAULT_NAME)
        origin = "owner-selector-default"
    cfg = mod.FusedMoEBlockConfig(**raw).effective_for(
        num_tokens=int(batch), ep_size=shape["ep"])
    return cfg, origin, raw


def _clamp_declared(raw, batch, ep):
    """The same clamp, declared once more and compared, never used to run.

    A rule written twice and never compared is a rule that drifts, so the
    second declaration is checked against the first at every count.
    """
    import math
    local = int(batch) // int(ep)
    bt = math.gcd(min(raw["bt"], local), local)
    bts = bt if raw.get("bts") is None else min(raw["bts"], bt * int(ep))
    return {"bt": bt, "bf": raw["bf"], "btc": min(raw["btc"], bts),
            "bse": raw["bf"] if raw.get("bse") is None else raw["bse"],
            "bts": bts}


def _layer(x, tw, ti, w1, w2, w3, s1, s2, s3, block, *, mod, mesh, topk):
    return mod.fused_ep_moe_v2(
        mesh, x, w1, w2, w3, tw, ti, topk, act_fn=_sglang.ACT,
        block_config=block, quant_block_k=None,
        w1_scale=s1, w2_scale=s2, w3_scale=s3,
        direct_scaled_dot=_DIRECT_SCALED_DOT, enable_act_quant=_ACT_QUANT,
        dp_axis_name=_sglang.DATA, tp_axis_name=_sglang.TENSOR)


def bind(tree_path, shape, profile_flags=None):
    del tree_path  # this arm binds an external pin, never a tree of ours
    _sglang.assert_weight_form(shape)  # no four-bit form exists here
    fetched, root, mod = _sglang.load(_KERNEL, _MODULE_KEY)
    device_key, key_from = _sources.owner_device_key(root, _sglang.RULE)
    wdt = shape["weight_dtype"]
    sources = [os.path.join(root, _KERNEL), os.path.join(root, _TABLE)]
    resolved = {int(b): _selector(mod, root, shape, b, device_key, wdt)
                for b in _sources.config.BATCHES}
    # THEIR SILENT DEFAULTS, READ OFF THEIR OWN SIGNATURE AND STAMPED. These
    # three decide the program and this row never passes them, so a cell that
    # did not record them would rely on a default staying a default across
    # pins. Read from their entry rather than restated (v2/kernel.py:2296).
    silent = _sources.signature_defaults(
        mod.fused_ep_moe_v2,
        ("enable_bt_scatter_overlap", "interleave_bt",
         "cross_expert_prefetch_mode"))
    # Their two environment switches, RESOLVED by their own reader with their
    # own call-site defaults (v2/kernel.py:2453-2454), so a cell says what ran
    # rather than what the shell happened to hold. Their third read,
    # SGLJAX_SKIP_PREQUANT (:1947, :2166), sits behind enable_act_quant, which
    # this row leaves off, so it cannot reach this program.
    env_defaults = _sources.read_call_literals(os.path.join(root, _KERNEL),
                                               "_env_bool")
    env_resolved = {name: bool(mod._env_bool(name, default))
                    for name, default in env_defaults}

    def block_for(batch):
        if int(batch) not in resolved:
            raise _sources.ShapeConstraintRefusal(
                f"their selector was mirrored at {list(resolved)} and this "
                f"count is {batch}: a count outside config.BATCHES has no "
                f"resolved configuration and this arm will not invent one")
        return resolved[int(batch)][0]

    def verify_mirror():
        return _sglang.mirror(
            _KERNEL, _MODULE_KEY, _FIELDS,
            lambda pmod, proot, batch: _selector(pmod, proot, shape, batch,
                                                 device_key, wdt),
            lambda raw, batch: _clamp_declared(raw, batch, shape["ep"]),
            {b: resolved[b][0] for b in resolved}, device_key, key_from)

    bound = _sglang.binding(
        shape,
        lambda mesh: functools.partial(_layer, mod=mod, mesh=mesh,
                                       topk=shape["top_k"]),
        block_for)
    bound.update({
        "verify_mirror": verify_mirror,
        "contract": _sources.contract(
            ("fused-moe-v2",), False,
            # Their integration needs no collective outside the kernel: the
            # routing-metadata all-gather, the dispatch and the combine
            # are all inside their one pallas program, and their v2 kernel
            # file holds no collective at all.
            (), "none", ("idx", "weights")),
        "source_hash": _sources.source_hash(sources, root),
        "tree_sha": fetched["commit"], "tree": root, "pin": _sglang.PIN,
        "namespace": _MODULE_KEY, "sources": sources,
        # What their kernel reads from the environment, by name, so the engine
        # can quarantine it per cell.
        "env_reads": [name for name, _ in env_defaults]
                     + ["BENCHOFF_SRC_" + _sglang.PIN.upper()],
        "flags": _sglang.flags(_FIELDS, resolved, dict(
            silent, act_fn=_sglang.ACT,
            direct_scaled_dot=_DIRECT_SCALED_DOT,
            enable_act_quant=_ACT_QUANT, quant_block_k=None,
            env_resolved=env_resolved,
            device_key=device_key, device_key_from=key_from)),
        "provenance": {
            "entry": "sgl_jax/srt/kernels/fused_moe/v2/kernel.py:2296 "
                     "fused_ep_moe_v2",
            "router": "EXTERNAL: their serving layer takes the selection "
                      "precomputed per rank, so it is outside the window",
            "activation_contract": "sixteen-bit token by eight-bit weight, "
                                   "their default: their eight-bit token mode "
                                   "is a different row, not this one",
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
