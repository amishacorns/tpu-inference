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

"""The cell record: one measurement, with its complete meaning attached.

A cell is the only unit of data in this benchmark. Every table in the
document is derived from cells and nothing else. A cell that does not
carry its complete fingerprint cannot be written: the writer raises,
because a cell whose meaning needs archaeology is how a benchmark lies
a month later.

Row identity is not stored on the cell. It is derived at render time
from config_origin, so a cell measured under a configuration this
project chose can never render under an external project's name.
"""

import json
import os

SCHEMA_VERSION = 2

# Every field, with the question it answers. All are required. "None" is a
# legal value only where explicitly noted, and means the field was
# affirmatively established to be absent, never that it was skipped.
FINGERPRINT_FIELDS = {
    # What was measured
    "arm": "registry name of the arm, e.g. production_pair, fused_ep_v2",
    "shape_id": "which model shape, from config.SHAPES",
    "batch": "token count",
    "routing": "replay | random",
    # Where the code came from
    "tree": "path of the source tree the arm bound",
    "tree_sha": "commit of that tree",
    "source_hash": "sha256 of the kernel source file(s) the arm executed",
    # What configuration it ran
    "profile": "config.PROFILES key the flags came from",
    "config_origin": "owner-selector | owner-default | ours-served | "
                     "ours-tuned | upstream-default",
    "flags": "dict of every flag value in force, from config, verbatim",
    "frame": "config.FRAMES key of the compiler options",
    "frame_options": "the frame's RESOLVED option dict, verbatim. Stamped so "
                     "a renamed or env-extended frame cannot alias: two cells "
                     "compare only if these dicts are equal, never by name",
    # What the measurement was
    "draw_seed": "seed list for the routing draw; None for replay",
    "replay_steps": "list of [capture_id, call] pairs; None for random",
    "input_hashes": "dict of sha256 by path for the recorded capture a "
                    "replay cell replays; None for random. In the "
                    "fingerprint so two cells read from two recordings at "
                    "one path can never share an identity",
    "iters": "profiled iterations",
    "repeats": "independent capture repeats",
    "warmup": "warmup iterations",
    "tier": "screen | heavy",
    # What session it belongs to
    "role": "measure | control-open | control-close. Controls carry distinct "
            "roles so the opening and closing reading of one configuration "
            "never share a cell_id, and their exclusion from tables is "
            "structural rather than a rendering convention",
    "session_id": "run id of the lock window",
    "session_drift_us": "closing-minus-opening control delta at this batch; "
                        "None until the closing control lands, backfilled "
                        "by run.py before the session's cells are final",
    "env_hash": "sha256 of the pip freeze",
    "device": "device fingerprint string",
}

RESULT_FIELDS = {
    "status": "ok | refused | failed",
    "program_us": "whole-layer program device time, mean; None unless ok",
    "per_step_us": "list, one entry per replay step or repeat; None unless ok",
    "kernel_self_us": "kernel self time, mean; None unless ok",
    # The honesty quantities, on the cell rather than in a side file, so a
    # reader of the number sees the wall that cages it and how much of the
    # window it was read from.
    "wall_us": "host wall per call, untraced loop, mean over steps; None "
               "unless ok",
    "window_wall_us": "host wall per call inside the profiled window, mean; "
                      "None unless ok",
    "wall_over_program": "wall_us over program_us; None unless ok",
    "coverage": "captured over expected window executions, the smallest "
                "across steps; None unless ok",
    "derivation": "full-window | self-normalized per execution; None unless "
                  "ok",
    "executions": "captured window executions, the smallest across steps; "
                  "None unless ok",
    "per_step_basis": "list of per-step {executions, coverage, derivation}; "
                      "None unless ok",
    "kit_source_hash": "sha256 of the kit's own python sources that measured "
                       "this cell, so a stored cell names its instrument",
    "refusal": "the verbatim refusal text; None unless refused/failed",
    "refusal_attribution": "kernel | harness | shape-constraint | fetch | "
                           "declared-out-of-scope; None unless refused/failed. "
                           "A harness-attributed refusal may never render "
                           "where a reader would read it as the kernel's "
                           "capability",
}

REQUIRED = list(FINGERPRINT_FIELDS) + list(RESULT_FIELDS) + ["schema_version"]

VALID_ORIGINS = ("owner-selector", "owner-default", "ours-served",
                 "ours-tuned", "upstream-default")
VALID_STATUS = ("ok", "refused", "failed")
VALID_ROUTING = ("replay", "random")
VALID_TIER = ("screen", "heavy")
VALID_ATTRIBUTION = ("kernel", "harness", "shape-constraint", "fetch",
                     "declared-out-of-scope")


class CellError(ValueError):
    pass


def cell_id(fields):
    """Content address: sha256 over the fingerprint, never the file name.

    No consumer may resolve a cell by file name, filesystem order or
    modification time. A reading published under an identity it does not
    have is the failure this exists to kill."""
    import hashlib
    basis = {k: fields[k] for k in FINGERPRINT_FIELDS}
    return hashlib.sha256(
        json.dumps(basis, sort_keys=True).encode()).hexdigest()[:16]


def result_hash(fields):
    """Digest of the RESULT half of a cell, checked at read time.

    The cell_id covers the fingerprint alone, so without this a stored
    number could be edited and the cell would still validate, and two cells
    with one fingerprint and two different figures would share one
    address."""
    import hashlib
    basis = {k: fields.get(k) for k in RESULT_FIELDS}
    return hashlib.sha256(
        json.dumps(basis, sort_keys=True).encode()).hexdigest()[:16]


VALID_ROLE = ("measure", "control-open", "control-close")


def make_cell(**fields):
    """Build and validate one cell dict. Raises CellError on any gap."""
    fields.setdefault("schema_version", SCHEMA_VERSION)
    fields.setdefault("role", "measure")
    fields.pop("cell_id", None)
    fields.pop("result_hash", None)
    missing = [k for k in REQUIRED if k not in fields]
    if missing:
        raise CellError(f"cell missing required fields: {missing}")
    extra = [k for k in fields if k not in REQUIRED]
    if extra:
        raise CellError(f"cell has unknown fields: {extra}")
    if fields["config_origin"] not in VALID_ORIGINS:
        raise CellError(f"config_origin {fields['config_origin']!r} not in "
                        f"{VALID_ORIGINS}")
    if fields["status"] not in VALID_STATUS:
        raise CellError(f"status {fields['status']!r} not in {VALID_STATUS}")
    if fields["routing"] not in VALID_ROUTING:
        raise CellError(f"routing {fields['routing']!r}")
    if fields["tier"] not in VALID_TIER:
        raise CellError(f"tier {fields['tier']!r}")
    if fields["role"] not in VALID_ROLE:
        raise CellError(f"role {fields['role']!r} not in {VALID_ROLE}")
    if (fields["routing"] == "replay" and fields["status"] == "ok"
            and not fields["replay_steps"]):
        raise CellError("ok replay cell without replay_steps")
    if (fields["routing"] == "random" and fields["status"] == "ok"
            and fields["draw_seed"] is None):
        raise CellError("ok random cell without draw_seed")
    if fields["status"] == "ok":
        for k in ("program_us", "per_step_us", "wall_us", "coverage"):
            if fields[k] is None:
                raise CellError(f"ok cell with {k}=None")
    else:
        if not fields["refusal"]:
            raise CellError("refused/failed cell without refusal text")
        if fields["refusal_attribution"] not in VALID_ATTRIBUTION:
            raise CellError(
                f"refusal_attribution {fields['refusal_attribution']!r} "
                f"not in {VALID_ATTRIBUTION}")
    if not isinstance(fields["flags"], dict):
        raise CellError("flags must be the verbatim dict of values in force")
    if not isinstance(fields["frame_options"], dict):
        raise CellError("frame_options must be the resolved option dict")
    fields["cell_id"] = cell_id(fields)
    fields["result_hash"] = result_hash(fields)
    return fields


def write_cells(path, cells):
    """Append validated cells to a jsonl store. One line per cell."""
    with open(path, "a") as f:
        for c in cells:
            f.write(json.dumps(make_cell(**dict(c)), sort_keys=True) + "\n")


def read_cells(path):
    """Read and re-validate every cell. A store with an invalid line is
    refused whole: partial trust is how mixed-meaning rows happen."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                stored_id = raw.get("cell_id")
                stored_rh = raw.get("result_hash")
                cell = make_cell(**raw)
                if stored_id is not None and stored_id != cell["cell_id"]:
                    raise CellError(
                        f"cell_id mismatch: stored {stored_id}, derived "
                        f"{cell['cell_id']} (fingerprint edited after write)")
                if stored_rh is not None and stored_rh != cell["result_hash"]:
                    raise CellError(
                        f"result_hash mismatch: stored {stored_rh}, derived "
                        f"{cell['result_hash']} (a result field edited after "
                        f"write)")
                out.append(cell)
            except (json.JSONDecodeError, CellError) as e:
                raise CellError(f"{path}:{i}: {e}") from e
    return out
