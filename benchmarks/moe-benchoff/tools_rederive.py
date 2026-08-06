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

"""Re-derive every number in MOE_COMPARISON.md from the cell stores.

    python3 tools_rederive.py

Reads every store under cells/ through cells.read_cells, which re-validates
each record against its own content address, then rebuilds each table of the
document from those records and compares. Exits nonzero on any mismatch.

THE DERIVATION RULE. A table is one manifest. Its rows are the manifest's
arms at the profiles the manifest declares, and their role is measure, so a
control reading is never a row. A COLUMN REPLAYS ONE DRAW. Every row of one
count has to replay the same steps or the column compares two routings, so
the draw a column publishes is the one the most rows carry, and the newest
of those where two draws carry as many. Within that draw the last cell of an
(arm, count) is the reading, so a re-measurement supersedes the one before
it. A drawn-routing column carries one draw already and the rule reduces to
last wins. Recommended is the fastest of the manifest's switch members at
that count, falling back to the named arm where no member ran. A measured
value and a Recommended value have to match the document exactly at one
decimal. A comparison percentage is how much time Production spends over
Recommended, and it can be taken either from the underlying cells or from
the two values the table prints, so the second is accepted as a rounding
note. Anything else is a mismatch and this exits nonzero.
"""

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cells  # noqa: E402

DOC = os.path.join(HERE, "MOE_COMPARISON.md")


def stores():
    """Every shipped store, read and re-validated. Raises on a bad record."""
    return {os.path.basename(p).split(".")[0]: cells.read_cells(p)
            for p in sorted(glob.glob(os.path.join(HERE, "cells", "*.jsonl")))}


def readings(store, profiles):
    """{(arm, count): cell} under the one-draw rule and the last-wins rule."""
    rows = [(i, c) for i, c in enumerate(store)
            if c["role"] == "measure"
            and profiles.get(c["arm"]) == c["profile"]]
    draws = {}
    for i, cell in rows:
        seen = draws.setdefault(
            (int(cell["batch"]), json.dumps(cell["replay_steps"])),
            [set(), -1])
        seen[0].add(cell["arm"])
        seen[1] = max(seen[1], i)
    published = {}
    for (count, steps), (arms, last) in draws.items():
        if (len(arms), last) > published.get(count, ((0, -1), None))[0]:
            published[count] = ((len(arms), last), steps)
    return {(c["arm"], int(c["batch"])): c for _, c in rows
            if json.dumps(c["replay_steps"]) == published[int(c["batch"])][1]}


def sections(text):
    """{top-level heading: its markdown tables}, each table (header, rows)."""
    out, name, found, block = {}, None, [], []
    for line in text.split("\n") + ["## "]:
        if line.startswith("|"):
            block.append([c.strip() for c in line.strip("|").split("|")])
            continue
        if block:
            found.append((block[0], [(r[0], r[1:]) for r in block[2:]]))
            block = []
        if line.startswith("## "):
            out[name] = found
            name, found = line[3:].strip(), []
    return out


def check(manifest, store, doc_tables, report):
    """Every value of one shape's tables, against one store."""
    labels = {a["row"]: a["arm"] for a in manifest["arms"]}
    switch = manifest["recommended"]
    read = readings(store, {a["arm"]: a["profile"] for a in manifest["arms"]})
    shape = manifest["shape_label"]
    printed, best = {}, {}
    for head, rows in doc_tables:
        if not head[1:] or not all(c.isdigit() for c in head[1:]):
            continue          # a table whose columns are not token counts
        for label, values in rows:
            for count, shown in zip([int(c) for c in head[1:]], values):
                shown = shown.strip("*")
                where = "%s %s B%d" % (shape, label, count)
                printed.setdefault(label, {})[count] = shown
                numeric = bool(re.match(r"^[-+]?[0-9.]+%?$", shown))
                if head[0] == "Comparison":
                    other = label.split(" vs ")[-1]
                    base = read.get((labels.get(other), count))
                    if base is None or count not in best:
                        report("MISSING", where, "no pair to compare")
                        continue
                    exact = (float(base["program_us"]) / best[count] - 1) * 100
                    table = (float(printed[other][count]) /
                             float(printed["Recommended"][count]) - 1) * 100
                    report("OK" if "%.1f" % exact == shown.strip("+%") else
                           "ROUNDING" if "%.1f" % table == shown.strip("+%")
                           else "MISMATCH", where,
                           "derived %.4f percent, document %s"
                           % (exact, shown))
                elif label == "Recommended":
                    if count not in best:
                        fallback = read.get((switch["fallback"], count))
                        if fallback is None:
                            report("MISSING", where, "no switch member ran")
                            continue
                        best[count] = float(fallback["program_us"])
                    report("OK" if "%.1f" % best[count] == shown else
                           "MISMATCH", where, "derived %.4f, document %s"
                           % (best[count], shown))
                elif label in labels:
                    cell = read.get((labels[label], count))
                    if cell is None:
                        report("MISSING", where, "no cell in the store")
                    elif cell["status"] != "ok":
                        report("MISMATCH" if numeric or
                               cell["refusal_attribution"] == "harness"
                               else "OK", where, "%s refusal, document %r"
                               % (cell["refusal_attribution"], shown))
                    elif not numeric:
                        report("MISMATCH", where, "the store holds a "
                               "measurement, the document holds %r" % shown)
                    else:
                        value = float(cell["program_us"])
                        if labels[label] in switch["switch_members"]:
                            best[count] = min(best.get(count, value), value)
                        report("OK" if "%.1f" % value == shown else "MISMATCH",
                               where, "derived %.4f, document %s"
                               % (value, shown))


def main():
    store, found = stores(), sections(open(DOC).read())
    counts = {"OK": 0, "ROUNDING": 0, "MISMATCH": 0, "MISSING": 0}
    notes = []

    def report(kind, where, detail):
        counts[kind] += 1
        if kind != "OK":
            notes.append("%-9s %-44s %s" % (kind, where, detail))

    print("cell stores, read and revalidated through cells.read_cells")
    for name, records in sorted(store.items()):
        print("  %-24s %3d records" % (name, len(records)))
    print("")
    for path in sorted(glob.glob(os.path.join(HERE, "manifests", "*.json"))):
        manifest = json.load(open(path))
        if manifest["shape_label"] not in found:
            print("%-26s no section in the document, nothing to re-derive"
                  % os.path.basename(path))
            continue
        print("%-26s -> %s" % (os.path.basename(path), manifest["cells"]))
        name = os.path.basename(manifest["cells"]).split(".")[0]
        check(manifest, store[name], found[manifest["shape_label"]], report)
    print("\n" + "\n".join(notes) if notes else "")
    print("re-derived %d values from %d cells, %d exact, %d rounding notes, "
          "%d mismatches, %d missing"
          % (sum(counts.values()), sum(len(s) for s in store.values()),
             counts["OK"], counts["ROUNDING"], counts["MISMATCH"],
             counts["MISSING"]))
    return 1 if counts["MISMATCH"] or counts["MISSING"] else 0


if __name__ == "__main__":
    sys.exit(main())
