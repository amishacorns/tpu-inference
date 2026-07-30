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

"""The measurement engine: routing in, device microseconds out.

This module owns four things and nothing else.

Routing Draws. One routing per (seed, mode, token count), a function of
those three and never of the arm's name, so two arms measured at the same
cell see byte-identical router logits.

The Replay Loader. A recorded capture holds per-expert row totals for one
real layer call. This rebuilds router inputs whose own softmax and top-k
land exactly those totals, and verifies that on the host for every step
before the step is measured. A step whose reconstruction does not match
the record is not measured at all.

The Measure Loop. Blocked warm-ups, an untraced loop for the host wall
figure, then a profiler window holding exactly the requested number of
calls. A compile event anywhere after the warm-up raises: the cell
produces no number rather than a polluted one.

The Profiler Readout. Device time comes from the xprof conversion of the
xplane capture. There is no host-timer fallback. The whole-layer figure
is the sum of per-operation self time over every operation in the
anchor's program; the kernel's own self time is reported beside it and is
never the ranking number.

IMPORT SAFE BEFORE JAX. Nothing here imports jax or numpy at module
level, because the compiler frame and the deviceless flags have to be set
in the environment before the first `import jax`, and run.py imports this
module while doing that. Every heavy import lives inside a function.
"""

import glob
import gzip
import heapq
import json
import os
import shutil

import config

# Widths a case is built at, with the dtype each array carries.
#
# The dtypes:
#   x       bfloat16    the activation dtype the layer is measured at:
#                       x is (B,H) bf16 and gating is (B,E) bf16, and the
#                       matmul left-hand side is cast to bf16.
#   logits  float32     what every arm here places and what the served
#                       adapter itself feeds its router (the pinned pull
#                       request 3288 adapter,
#                       tpu_inference/layers/common/moe_fused_ep.py, casts
#                       gating to float32). The declared dtype has to
#                       match, because host_top_k and an in-program router
#                       must select the same experts, and a bf16 default
#                       here would route differently for any arm that ever
#                       omitted prepare.
#   idx     int32       host top-k expert indices are int32.
#   weights float32     host top-k gate weights are float32.
#   sizes   int32       group sizes are int32.
CASE_DTYPES = {
    "x": "bfloat16",
    "logits": "float32",
    "idx": "int32",
    "weights": "float32",
    "group_sizes": "int32",
}

# One random-number stream per routing mode, so the replay stream and the
# drawn stream never share draws at the same (seed, batch). The mapping is
# fixed, so that rerunning a recorded drawn cell reproduces its logits.
MODE_STREAMS = {"random": 1, "replay": 3}


# ---------------------------------------------------------------------------
# Refusals, One Class Per Reason
# ---------------------------------------------------------------------------
# A refused cell carries an attribution from the cells.py enum, and the
# attribution is decided by the TYPE of what was raised, never by reading the
# message. The failure this closes: booking assertion failures as benign
# refusals publishes a harness artefact as a kernel's refusal.
#
# An arm raises one of these to refuse. Anything else it raises is attributed
# to the kernel, because it came out of the kernel's own code path. An
# AssertionError is not a refusal at all: see Defect.
class Refusal(Exception):
    """A point that cannot run, with the side it belongs to attached."""

    ATTRIBUTION = "kernel"


class KernelRefusal(Refusal):
    """The kernel cannot do this: it says so and the cell records it."""

    ATTRIBUTION = "kernel"


class HarnessRefusal(Refusal):
    """This kit cannot present the point. Never the kernel's capability."""

    ATTRIBUTION = "harness"


class ShapeConstraintRefusal(Refusal):
    """The shape or the count is outside what the arm accepts by design."""

    ATTRIBUTION = "shape-constraint"


class FetchRefusal(Refusal):
    """The source was not fetched, so there is nothing to measure."""

    ATTRIBUTION = "fetch"


class OutOfScopeRefusal(Refusal):
    """The arm declared this point out of scope before the run."""

    ATTRIBUTION = "declared-out-of-scope"


class Defect(Exception):
    """Not a refusal. A bug in this kit or an arm, and it fails the run.

    An AssertionError is converted to this. An assertion is a statement
    that something cannot happen; when it happens, the instrument is
    broken, and recording that as a cell the kernel refused is how a
    harness artefact gets published as a kernel property.
    """


class CaptureError(HarnessRefusal):
    """A recorded capture that cannot serve the cell being measured."""


class WindowError(HarnessRefusal):
    """The timed window did not hold what a measurement requires."""


def attribution_for(exc):
    """Which side a raised exception belongs to, or None when it is a defect.

    None means the caller must fail the run rather than write a cell.

    AN IMPORT FAILURE IS ABOUT SOURCE, NOT ABOUT A KERNEL. When a module is
    not where the arm looked, nothing has been asked of the kernel yet: the
    code is absent from the tree that was bound, which is the fetch
    attribution's own question. Booking
    "No module named tpu_inference.kernels.fused_moe.v2" as the kernel's
    refusal reads in a table as a kernel that cannot do the work.
    """
    if isinstance(exc, Defect) or isinstance(exc, AssertionError):
        return None
    if isinstance(exc, Refusal):
        return exc.ATTRIBUTION
    if isinstance(exc, ImportError):
        return "fetch"
    # A LEAKED TRACER IS NEVER A KERNEL'S CAPABILITY. jax raises it when a
    # value built inside one trace is used in another, which is a statement
    # about how this kit or an arm cached something across shapes, not about
    # what the kernel can compute. The type is matched by name so this stays
    # importable before jax; the message is never read.
    if type(exc).__name__ == "UnexpectedTracerError":
        return "harness"
    return "kernel"


# ---------------------------------------------------------------------------
# Routing Draws
# ---------------------------------------------------------------------------
def draw_seed(seed, mode, batch):
    """The seed list the draw's generator is built from, stamped on the cell.

    The list is the whole state: (seed, mode stream, token count). A cell
    carrying it can be redrawn without this file.
    """
    if mode not in MODE_STREAMS:
        raise ValueError(f"unknown routing mode {mode!r}")
    return [int(seed), MODE_STREAMS[mode], int(batch)]


def _rng(seed, mode, batch):
    import numpy as np
    return np.random.default_rng(draw_seed(seed, mode, batch))


def random_logits(batch, experts, rng):
    """Pure-noise router logits, scaled so no tie survives the top-k.

    The draw is standard normal times 0.01, then times 100. Crafted-balanced
    logits activate every local expert and overstate the decode weight
    stream, which is why the drawn mode here is noise.
    """
    logits = rng.standard_normal((batch, experts)).astype("float32") * 0.01
    return logits * 100.0


def host_top_k(logits, top_k):
    """Softmax, top-k, renormalize, on the host.

    This is the selection an arm whose serving contract takes the routing
    precomputed is handed, and it is the same arithmetic the arms that
    select inside their own program perform, so the two kinds of arm route
    identically.
    """
    import numpy as np
    g = np.asarray(logits, dtype=np.float32)
    probs = np.exp(g - g.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    idx = np.argpartition(-probs, top_k, axis=1)[:, :top_k].astype(np.int32)
    w = np.take_along_axis(probs, idx, axis=1)
    w = (w / w.sum(1, keepdims=True)).astype(np.float32)
    return idx, w


def routing_accounting(group_sizes, batch, shape):
    """How much of the row buffer the routing actually fills.

    The buffer every arm reasons about is tokens times the selection width.
    What a chip really computes is the fill of its own contiguous slice of
    experts. The gap is padding, and it is a property of the routing rather
    than of any arm.
    """
    import numpy as np
    gs = np.asarray(group_sizes, dtype=np.int64)
    ep = int(shape["ep"])
    local = int(shape["experts"]) // ep
    buffer_rows = int(batch) * int(shape["top_k"])
    fills = gs.reshape(ep, local).sum(axis=1)
    busiest = int(fills.max())
    return {
        "buffer_rows_per_chip": buffer_rows,
        "real_rows_total": int(gs.sum()),
        "real_rows_busiest_chip": busiest,
        "pad_fraction_busiest_chip": float(1.0 - busiest / buffer_rows),
        "empty_experts": int((gs == 0).sum()),
    }


class Case:
    """One measured point's inputs, on the host, with where they came from."""

    def __init__(self, batch, routing, x, logits, idx, weights, group_sizes,
                 origin, accounting):
        self.batch = batch
        self.routing = routing
        self.x = x
        self.logits = logits
        self.idx = idx
        self.weights = weights
        self.group_sizes = group_sizes
        self.origin = origin
        self.accounting = accounting

    def gating(self):
        """The selection, in every form an arm may consume.

        An arm reads the entries its contract names in gating_consumed and
        ignores the rest. Handing all four keeps one case type for arms that
        route inside their own program and arms that take the routing ready
        made.
        """
        return {"logits": self.logits, "idx": self.idx,
                "weights": self.weights, "group_sizes": self.group_sizes}


def _tokens(batch, shape, rng):
    """The token activations. Standard normal at the shape's own width.

    Rows are standard normal at unit scale, matching the matmul left-hand
    side the layer is measured against.
    """
    return rng.standard_normal((int(batch), int(shape["hidden"]))
                               ).astype("float32")


def draw_case(shape, batch, seed):
    """One drawn-routing case, a function of (seed, mode, token count) only."""
    import numpy as np
    rng = _rng(seed, "random", batch)
    x = _tokens(batch, shape, rng)
    logits = random_logits(int(batch), int(shape["experts"]), rng)
    idx, w = host_top_k(logits, int(shape["top_k"]))
    gs = np.bincount(idx.ravel(), minlength=int(shape["experts"]))
    return Case(batch=int(batch), routing="random", x=x, logits=logits,
                idx=idx, weights=w, group_sizes=gs,
                origin={"collection": "drawn", "mode": "random",
                        "draw_seed": draw_seed(seed, "random", batch)},
                accounting=routing_accounting(gs, batch, shape))


def case_specs(shape, batch):
    """Shapes and dtypes of one case, for the deviceless build map.

    Returned as plain tuples so this module stays importable before jax.
    run.py turns them into jax.ShapeDtypeStruct values.
    """
    b, h = int(batch), int(shape["hidden"])
    e, k = int(shape["experts"]), int(shape["top_k"])
    return (((b, h), CASE_DTYPES["x"]),
            {"logits": ((b, e), CASE_DTYPES["logits"]),
             "idx": ((b, k), CASE_DTYPES["idx"]),
             "weights": ((b, k), CASE_DTYPES["weights"]),
             "group_sizes": ((e,), CASE_DTYPES["group_sizes"])})


def place(case):
    """The case on the default device, in the dtypes CASE_DTYPES names.

    Placement and every cast happen here, before the timed window, so a
    timed program never contains a conversion that belongs to the benchmark
    rather than to the arm. An arm that needs its own sharding or layout
    declares a prepare hook and this is not used.
    """
    import jax
    import jax.numpy as jnp

    def put(a, dt):
        return jax.device_put(jnp.asarray(a, jnp.dtype(dt)))

    x = put(case.x, CASE_DTYPES["x"])
    gating = {k: put(v, CASE_DTYPES[k]) for k, v in case.gating().items()}
    return x, gating


# ---------------------------------------------------------------------------
# The Replay Loader
# ---------------------------------------------------------------------------
# A capture record says how many token-expert rows each expert received on
# one real layer call, and nothing else. It cannot say which token went
# where, so this does not reconstruct the original pairing. It reproduces
# the per-expert totals, which is what every arm here is sensitive to.
def _capture_id(record):
    """Which recorded layer a record belongs to, as a string.

    Captures write this two ways and both ship: the serving recorder tags
    each layer by module path in a "tag" field, and the kit's own trimmed
    capture carries the layer index in a "layer" field (see
    routing_capture/qwen3_5_397b/MANIFEST.json record_fields). Either is
    accepted; neither is invented.
    """
    for key in ("tag", "layer"):
        if key in record:
            return str(record[key])
    raise CaptureError(f"capture record has neither a tag nor a layer field: "
                       f"{sorted(record)}")


def load_capture(path):
    """Every record of a capture, grouped by token count.

    The serving recorder's tag field is accepted alongside a layer field.
    """
    if not os.path.exists(path):
        raise CaptureError(
            f"the recorded-routing capture is not at {path}. Replay cannot "
            f"run without it, and a drawn cell is not a substitute.")
    by_count = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise CaptureError(f"{path} line {number} is not "
                                   f"readable: {exc}") from exc
            for key in ("num_tokens", "call", "group_sizes"):
                if key not in record:
                    raise CaptureError(f"{path} line {number} has no "
                                       f"{key!r} field")
            _capture_id(record)
            by_count.setdefault(int(record["num_tokens"]), []).append(record)
    if not by_count:
        raise CaptureError(f"{path} holds no records")
    return by_count


def check_capture(by_count, shape, batches):
    """Prove a capture can serve a run, without touching a device.

    Every token count must be covered, every record must be the right
    width, every record's totals must equal tokens times the selection
    width, and no expert may hold more rows than there are tokens. A
    capture from another geometry is refused here rather than producing a
    quietly wrong selection later.
    """
    experts, top_k = int(shape["experts"]), int(shape["top_k"])
    problems = []
    for batch in batches:
        records = by_count.get(int(batch)) or []
        if not records:
            problems.append(f"no recorded call at {batch} tokens")
            continue
        for record in records:
            sizes = record["group_sizes"]
            if len(sizes) != experts:
                problems.append(
                    f"a record at {batch} tokens carries {len(sizes)} group "
                    f"sizes, but this shape has {experts} experts")
                break
            if sum(sizes) != int(batch) * top_k:
                problems.append(
                    f"a record at {batch} tokens has group sizes summing to "
                    f"{sum(sizes)}, but {batch} tokens selecting {top_k} "
                    f"experts each is {int(batch) * top_k} rows")
                break
            if max(sizes) > int(batch):
                problems.append(
                    f"a record at {batch} tokens gives one expert "
                    f"{max(sizes)} rows, which is more than the {batch} "
                    f"tokens there are: no token picks one expert twice")
                break
    if problems:
        raise CaptureError("the capture does not fit this run: "
                           + "; ".join(problems))
    return True


def active_experts(record):
    """How many experts a recorded call gave at least one row."""
    return sum(1 for n in record["group_sizes"] if int(n) > 0)


def is_degenerate(record, top_k):
    """Whether a record is a start-up artefact rather than a serving call.

    THE RULE: a call whose active-expert count is at or below
    config.REPLAY_EXCLUDE_ACTIVE_FACTOR times the selection width
    concentrated its routing the way the capture's opening warm-up sweep
    does, which leaves whole chips idle. That is a start-up artefact, not
    a serving condition.

    WHY A MULTIPLE AND NOT THE WIDTH ITSELF. The equality case
    (active == top_k) appears at every token count in the capture's first
    records, but the same warm-up sweep's other-layer copies light up to
    about 2.4 times the width and survive an equality-only rule; at a token
    count where a capture is thin they can dominate a cell. The bound is
    therefore a small multiple of the width. Every recorded serving call in
    this capture lights at least twenty times the width's worth short of
    the bound (the thinnest serving population's quartiles start near two
    hundred active experts of 512), so the widened rule cannot exclude an
    honest step here; a future capture whose serving calls concentrate
    harder must revisit the factor, and check_capture stamps the exclusion
    count per batch so a surprising cut is visible.

    The rule is a property of the record, not an index into the file. An
    index cut would exclude honest early serving steps and would keep a
    degenerate call that appears late.
    """
    if not config.REPLAY_EXCLUDE_ACTIVE_EQ_TOPK:
        return False
    factor = getattr(config, "REPLAY_EXCLUDE_ACTIVE_FACTOR", 1)
    return active_experts(record) <= int(top_k) * int(factor)


def select_replay_steps(by_count, batch, seed, shape,
                        count=config.REPLAY_STEPS_PER_CELL):
    """The records one replay cell measures, and their stamps.

    Degenerate calls are excluded by the rule above, before anything is
    selected, and the count of what the rule removed is reported in the
    refusal when too few serving calls remain.

    The chosen steps are named: the return value carries the
    (capture id, call) pair of every chosen record, so run.py can stamp
    replay_steps on the cell and a reader can go back to the exact calls.

    Selection is deterministic. Positions are drawn from the cell's own
    generator, which is a function of (seed, mode, token count) only, so a
    rerun at the same seed over the same capture replays the same calls.

    THE POSITIONS INDEX THE CAPTURE'S OWN ORDER, which is the order the
    recorder wrote and the order load_capture reads. A capture whose records
    were reordered or trimmed therefore draws different calls at the same
    seed, and the (capture id, call) pairs stamped on every cell are what
    names the calls a published reading used.
    """
    records = by_count.get(int(batch)) or []
    top_k = int(shape["top_k"])
    eligible = [r for r in records if not is_degenerate(r, top_k)]
    removed = len(records) - len(eligible)
    if len(eligible) < int(count):
        raise CaptureError(
            f"the capture holds {len(eligible)} serving calls at {batch} "
            f"tokens ({len(records)} records, {removed} excluded as "
            f"degenerate: every token on {top_k} experts), but a replay cell "
            f"measures {count}.")
    rng = _rng(seed, "replay", batch)
    positions = sorted(rng.choice(len(eligible), size=int(count),
                                  replace=False).tolist())
    chosen = [eligible[p] for p in positions]
    steps = [[_capture_id(r), int(r["call"])] for r in chosen]
    return chosen, steps


def selection_from_group_sizes(group_sizes, batch, top_k, order):
    """A per-token selection whose per-expert totals are the recorded ones.

    Tokens are filled one at a time, each taking the top_k experts with the
    most rows still to place, which places every recorded row and gives
    every token a full set of distinct experts.

    The fill order is a permutation rather than token order. Filling in
    token order hands the busiest experts to the lowest-numbered tokens,
    and token rows are sharded in contiguous blocks, so one chip would end
    up holding every token that wants a busy expert. That is an artefact of
    the layout rule, not something the capture says.
    """
    import numpy as np
    remaining = [(-int(n), int(e)) for e, n in enumerate(group_sizes) if n > 0]
    heapq.heapify(remaining)
    idx = np.zeros((int(batch), int(top_k)), dtype=np.int32)
    for position, token in enumerate(order):
        if len(remaining) < int(top_k):
            raise CaptureError(
                f"only {len(remaining)} experts still have rows to place at "
                f"token {position} of {batch}, but each token selects "
                f"{top_k}. The recorded totals cannot be laid out as a "
                f"selection.")
        taken = [heapq.heappop(remaining) for _ in range(int(top_k))]
        for rank, (negative, expert) in enumerate(taken):
            idx[token, rank] = expert
            if negative + 1 < 0:
                heapq.heappush(remaining, (negative + 1, expert))
    if remaining:
        raise CaptureError(
            "the recorded totals were not fully placed, which means they do "
            "not sum to the token count times the selection width")
    return idx


def logits_for_selection(idx, batch, shape, rng):
    """Router logits whose own top-k is the selection handed in.

    An arm that selects inside its own program has to arrive at the same
    experts as one that takes the selection ready made, or the two are not
    measuring the same routing. The chosen experts are lifted far enough
    above the noise that the ordering cannot be disturbed by it: the lift is
    8.0 minus 0.5 per rank.
    """
    import numpy as np
    logits = rng.standard_normal(
        (int(batch), int(shape["experts"]))).astype("float32") * 0.01
    rows = np.arange(int(batch))[:, None]
    ranks = np.arange(int(shape["top_k"]))[None, :]
    logits[rows, idx] += 8.0 - 0.5 * ranks
    return logits


def replay_case(record, shape, batch, seed):
    """One replay step's case, verified on the host before it is measured.

    The verification is the point of the loader: the logits are pushed back
    through the same softmax and top-k an arm runs, and the resulting
    per-expert totals must equal the recorded ones exactly. If they do not,
    the cell has no recorded routing rather than an approximation of it.
    """
    import numpy as np

    # The step's generator carries the call index as well, so the four steps
    # of one cell do not share token rows while a rerun still reproduces each
    # step exactly.
    rng = np.random.default_rng(draw_seed(seed, "replay", batch)
                                + [int(record["call"])])
    x = _tokens(batch, shape, rng)
    idx = selection_from_group_sizes(record["group_sizes"], batch,
                                     shape["top_k"],
                                     rng.permutation(int(batch)))
    logits = logits_for_selection(idx, batch, shape, rng)
    idx, w = host_top_k(logits, int(shape["top_k"]))
    gs = np.bincount(idx.ravel(), minlength=int(shape["experts"]))
    recorded = np.asarray(record["group_sizes"], dtype=np.int64)
    if not np.array_equal(gs.astype(np.int64), recorded):
        raise CaptureError(
            f"replaying capture {_capture_id(record)} call {record['call']} "
            f"at {batch} tokens produced expert loads that differ from the "
            f"recorded ones in {int((gs != recorded).sum())} experts. The "
            f"step is not measured.")
    return Case(batch=int(batch), routing="replay", x=x, logits=logits,
                idx=idx, weights=w, group_sizes=gs,
                origin={"collection": "recorded",
                        "capture_id": _capture_id(record),
                        "call": int(record["call"])},
                accounting=routing_accounting(gs, batch, shape))


# ---------------------------------------------------------------------------
# The Measure Loop
# ---------------------------------------------------------------------------
# Two things this loop deliberately does not do. It never samples periodic
# counters around the timed window, because that perturbs the timing it
# samples and so can never be the source of a published microsecond. And it
# writes no wall-clock sidecar file: the wall figure is returned to the
# caller instead.
COMPILE_EVENT_PREFIXES = ("/jax/core/compile/", "/jax/compilation_cache/")


def measure(call, args, iters, warmup, logdir, fresh=True):
    """Warm up, time untraced, then trace exactly `iters` calls.

    The warm-up is blocked, so every compile the cell needs happens before
    anything is timed, and a warm-up of at least one call is mandatory: a
    timed loop must never own a compile.

    The untraced loop produces the host wall figure. It is reported beside
    the device figure and is never the ranking number, but it has to exist:
    a device number far below the wall clock is a host-tax finding, and
    without the wall figure it is invisible.

    The traced window holds exactly `iters` calls and nothing else. The last
    result is blocked on inside the window: dispatch still pipelines, but
    the profiler session must not close before the queued device work
    completes. A session that closes early captures host dispatch events
    and zero device operations, which is a capture defect that reads as a
    plausible number.

    Any compile event observed after the warm-up raises. The cell produces
    no number rather than one that includes a compile.
    """
    import time

    import jax
    import jax.monitoring as mon
    if int(warmup) < 1:
        raise WindowError("warmup of at least one call is mandatory: a timed "
                          "loop must never own a compile")
    if int(iters) < 1:
        raise WindowError("a profiler window of zero calls measures nothing")
    if fresh:
        shutil.rmtree(logdir, ignore_errors=True)
    for _ in range(int(warmup)):
        jax.block_until_ready(call(*args))

    events = []

    def on_duration(event, duration, **kw):
        if event.startswith(COMPILE_EVENT_PREFIXES):
            events.append(event)

    def on_event(event, **kw):
        if event.startswith(COMPILE_EVENT_PREFIXES):
            events.append(event)

    mon.register_event_duration_secs_listener(on_duration)
    mon.register_event_listener(on_event)
    try:
        result = None
        started = time.perf_counter()
        for _ in range(int(iters)):
            result = call(*args)
        jax.block_until_ready(result)
        wall_us = (time.perf_counter() - started) / int(iters) * 1e6
        with jax.profiler.trace(logdir):
            result = None
            window_started = time.perf_counter()
            for _ in range(int(iters)):
                result = call(*args)
            jax.block_until_ready(result)
            window_wall_us = ((time.perf_counter() - window_started)
                              / int(iters) * 1e6)
            jax.effects_barrier()
    finally:
        try:
            mon.unregister_event_duration_listener(on_duration)
            mon.unregister_event_listener(on_event)
        except Exception:
            pass
    if events:
        raise WindowError(
            f"{len(events)} compile event(s) fired after the warm-up "
            f"({sorted(set(events))[:4]}): the window is polluted, so this "
            f"cell has no number. Fix the warm-up or the shape coverage and "
            f"measure again.")
    # THE WALL FIGURE IS RETURNED, NOT WRITTEN BESIDE THE CAPTURE. Writing it
    # to a sidecar file next to the trace and reading it back later separates
    # a figure from its bound; here it travels with the measurement and is
    # stamped, so a figure can never be read without the physical bound that
    # cages it.
    #
    # Two walls, because they answer different questions. wall_us comes from
    # the untraced loop and is the honest host-plus-device cost of one call,
    # with no profiler overhead in it: that is the one to compare against
    # program time. window_wall_us is the same loop inside the profiler and
    # carries the tracing overhead, so it says what the capture itself cost.
    return {"logdir": logdir, "wall_us": wall_us,
            "window_wall_us": window_wall_us, "iters": int(iters),
            "warmup": int(warmup)}


# ---------------------------------------------------------------------------
# The Profiler Readout
# ---------------------------------------------------------------------------
def hlo_stats_rows(logdir):
    """Every per-operation row of a capture, as a list of dicts.

    An empty conversion is never returned quietly. The first retry bypasses
    the converter's own result cache, because an empty parse gets cached in
    the capture directory and then poisons every later read of it. If the
    fresh parse is still empty the capture holds no device operations, which
    is a capture defect upstream of this reader, and it raises.
    """
    from xprof.convert import raw_to_tool_data as rtd

    # EVERY capture file under the directory is handed to the converter, in
    # sorted order. Nothing here picks A file: no newest-first listing, no
    # modification time, no reliance on the order the filesystem answered in.
    # Resolving a reading by file order is how one cell's number ends up
    # published under another cell's name.
    paths = sorted(glob.glob(os.path.join(logdir, "**/*.xplane.pb"),
                             recursive=True))
    if not paths:
        raise WindowError(f"no xplane capture under {logdir}")

    def convert(params):
        data, _ = rtd.xspace_to_tool_data(paths, "hlo_stats", params)
        if data is None:
            # The converter reports failure as (None, content_type). Parsing
            # that would raise TypeError, which the attribution machinery
            # books against the KERNEL; a truncated or unreadable capture is
            # the harness's own failure, so it raises here.
            raise WindowError(
                f"the hlo_stats conversion of {logdir} returned no data: the "
                f"capture is unreadable (truncated xplane, or the converter "
                f"failed), so the number does not exist. Capture again.")
        parsed = json.loads(data.decode() if isinstance(data, bytes) else data)
        cols = [c["id"] for c in parsed["cols"]]
        return [dict(zip(cols, [c["v"] for c in row["c"]]))
                for row in parsed["rows"]]

    rows = convert({})
    if not rows:
        rows = convert({"use_saved_result": False})
    if not rows:
        raise WindowError(
            f"the hlo_stats conversion of {logdir} returned zero rows even "
            f"with the result cache bypassed: the capture has no device "
            f"operation events, so the number does not exist. Capture again.")
    return rows


# How far an occurrence ratio may sit from a whole number before the scaling
# it implies is called a guess rather than a fact. Past this distance the
# reader refuses rather than warns, because a figure divided by a guessed
# count is not a measurement.
RATIO_TOLERANCE = 0.15

# Device programs that may execute inside a timed window besides the arm's
# own. THE LIST IS EMPTY, and that is a finding rather than an oversight:
# every device window enumerated so far has held exactly one program, the one
# carrying the arm's anchor (62 to 213 operations, 1080 to 4800 occurrences).
# No profiler start or stop artifact program appeared in any of them.
#
# So a second program in a window is refused rather than ignored. That closes
# a class it is otherwise only possible to note: work in an anchor-less
# program escapes a sum taken over the anchor's program alone, and it escapes
# silently. If a real artifact program is ever observed, add it here WITH the
# capture it was seen in and what it did, and never as a bare identifier: an
# allowlist entry is a claim that the work does not belong to the layer.
WINDOW_PROGRAM_ALLOWLIST = ()

# The fewest program executions a capture may hold and still be read. The
# floor exists because xprof has been seen dropping executions at the heavy
# token counts (1072 of an expected 2400 at 8192 tokens). Above this the
# per-operation self-normalization carries the figure; below it the sample is
# too thin to average and the cell has no number.
MIN_CAPTURED_EXECUTIONS = 50

# The smallest fraction of the window's executions a capture may hold. The
# absolute floor above guards tiny windows; this guards big ones, where 50
# captured executions of 2400 performed would pass the absolute floor while
# describing two percent of the window. The observed worst real capture is
# 44.7 percent at 8192 tokens, which clears this with margin.
MIN_COVERAGE = 0.25


def lowered_census(lowered):
    """The STRUCTURE of a lowered program, walked in memory.

    Returns counts only: how many functions the module holds, every custom
    call target with its count, every collective with its count, and an
    operation census by dialect and name. That is what one program being the
    same program as another means here, and it is what a serving-path
    comparison can check without a device.

    NEVER THROUGH as_text(). An arm's weights are materialized before its
    program is traced, so they are captured into the lowering as constants,
    and rendering such a module as text can produce a string large enough to
    fill the disk. The module is walked as operations instead, and no
    constant is ever read, so the census costs nothing and the text is never
    built.
    """
    module = lowered.compiler_ir(dialect="stablehlo")
    functions = 0
    calls, collectives, ops = {}, {}, {}
    collective_names = ("all_gather", "all_reduce", "reduce_scatter",
                        "collective_permute", "all_to_all",
                        "collective_broadcast")

    def visit(operation):
        nonlocal functions
        name = operation.name
        ops[name] = ops.get(name, 0) + 1
        if name == "func.func":
            functions += 1
        if name.endswith(".custom_call"):
            # The one attribute this census reads, by name. Nothing else is
            # touched: a constant's attribute holds the bulk the census exists
            # to avoid rendering.
            try:
                target = str(operation.attributes["call_target_name"])
            except (KeyError, IndexError, TypeError, ValueError):
                target = None
            key = (target or "(unnamed)").strip('"')
            calls[key] = calls.get(key, 0) + 1
        for candidate in collective_names:
            if name.endswith("." + candidate):
                collectives[candidate] = collectives.get(candidate, 0) + 1
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child.operation)

    visit(module.operation)
    return {"functions": functions, "custom_calls": calls,
            "collectives": collectives, "operations": ops,
            "operation_kinds": len(ops)}


def programs_in_window(rows):
    """Every device program the capture holds, with what it did.

    One entry per program identifier: how many operations it ran, the largest
    occurrence count among them, its total device self time, and the names of
    its heaviest few. This is the whole-window view a figure taken over one
    program has to be justified against.
    """
    census = {}
    for row in rows:
        pid = row.get("program_id")
        entry = census.setdefault(str(pid), {
            "program_id": str(pid), "ops": 0, "max_occurrences": 0.0,
            "self_time_us": 0.0, "heaviest": []})
        entry["ops"] += 1
        entry["max_occurrences"] = max(
            entry["max_occurrences"], float(row.get("occurrences", 0) or 0))
        entry["self_time_us"] += float(row.get("total_self_time", 0) or 0)
        entry["heaviest"].append((float(row.get("total_self_time", 0) or 0),
                                  str(row.get("hlo_op_name", ""))))
    for entry in census.values():
        entry["heaviest"] = [name for _, name in
                             sorted(entry["heaviest"], reverse=True)[:4]]
    return census


def program_ops(logdir, anchor, iters, executions_per_call=None):
    """Every operation of the anchor's program, and how often it ran.

    WHAT THE ANCHOR IS FOR. It names the program, and nothing else. The
    invariant here is about the PROGRAM, not the operation. Demanding that
    the anchor OPERATION run once per call refuses correct arithmetic: an arm
    may run the anchor operation several times inside one program by design,
    as the production pair path does with its two grouped matmuls.

    WHAT ONE CALL EXECUTES. One execution per device on a sharded mesh, so
    the expected count for a window is the call count times the device count.
    That expectation is config.DEVICE_REQUIRED's chip count unless a caller
    names another.

    THE WITNESS IS LOSSY, AND THAT IS NOT AN ERROR. At the heavy counts xprof
    drops executions from the trace: at 8192 tokens a 300-call window that
    performed 2400 executions was captured with 1072 of them. A shortfall is
    a property of the profiler, not a change in what the program did, so the
    figure is derived by per-operation self-normalization, each operation's
    device self total divided by ITS OWN occurrence count and scaled by its
    multiplicity. That is drop-robust by construction, and the arithmetic is
    identical when nothing was dropped, so a full-coverage capture reads
    exactly as a plain per-call division would.

    WHAT STILL RAISES. Several programs carrying the anchor, because a
    whole-layer figure is one program's. More executions than the window
    could have performed, because that is a real multiplicity surprise rather
    than a lossy witness. And coverage below MIN_CAPTURED_EXECUTIONS, because
    below that the sample is too thin to average.
    """
    expected_per_call = int(executions_per_call
                            or config.DEVICE_REQUIRED["chips"])
    rows = hlo_stats_rows(logdir)
    anchor_rows = [r for r in rows
                   if str(r.get("hlo_op_name", "")).startswith(anchor)]
    if not anchor_rows:
        raise WindowError(
            f"no operation named {anchor!r} in the capture under {logdir}: "
            f"the arm's declared anchor is not in its own timed program")
    programs = []
    for r in anchor_rows:
        if r.get("program_id") not in programs:
            programs.append(r.get("program_id"))
    if len(programs) != 1:
        raise WindowError(
            f"anchor {anchor!r} appears in {len(programs)} separately jitted "
            f"programs under {logdir}. A whole-layer figure is the total of "
            f"one program; summing several is a figure with no owner. Give "
            f"the arm one program or declare a different anchor.")
    pid = programs[0]
    # THE WHOLE WINDOW, NOT JUST THE ANCHOR'S PROGRAM. A second device
    # program executing inside the window is work the layer's figure does not
    # contain, and taking the sum over the anchor's program alone would drop
    # it without a word. Every program in the capture is enumerated and
    # anything that is neither the anchor's nor allowlisted refuses the point.
    census = programs_in_window(rows)
    strangers = [entry for key, entry in sorted(census.items())
                 if key != str(pid)
                 and key not in WINDOW_PROGRAM_ALLOWLIST]
    if strangers:
        described = "; ".join(
            f"program {e['program_id']} ran {e['ops']} operations, up to "
            f"{e['max_occurrences']:.0f} occurrences, "
            f"{e['self_time_us']:.1f} us of device self time, heaviest "
            f"{e['heaviest']}" for e in strangers)
        raise WindowError(
            f"the window under {logdir} holds {len(census)} device programs, "
            f"and the layer's figure is one program's total. Besides the "
            f"anchor's program {pid}: {described}. Work in a second program "
            f"is not in the figure, so this point has no number rather than "
            f"a figure that quietly excludes it.")
    in_program = [r for r in rows if r.get("program_id") == pid]
    counts = [float(r.get("occurrences", 0) or 0) for r in in_program]
    positive = [c for c in counts if c > 0]
    if not positive:
        raise WindowError(
            f"program {pid} under {logdir} has no operation with a positive "
            f"occurrence count: the capture holds the program's name and "
            f"none of its work, so the number does not exist.")
    # THE WINDOW BASIS IS THE MODAL OCCURRENCE COUNT, not the minimum. Most
    # operations run once per execution, so the mode IS the execution count
    # the capture holds; the minimum is one rare or plane-local operation
    # away from poisoning every other operation's multiplicity (a 300-of-2400
    # op would multiply the whole figure by eight, silently). Ties resolve to
    # the largest candidate. Normalizing against one named operation's own
    # count is the same idea; the mode is that idea without trusting a single
    # operation.
    tally = {}
    for c in positive:
        tally[c] = tally.get(c, 0) + 1
    top = max(tally.values())
    captured = max(c for c, n in tally.items() if n == top)
    expected = float(iters) * expected_per_call
    if captured > expected:
        # A program where EVERY operation runs k times per execution puts the
        # mode at k times the execution count. When the excess is a clean
        # integer, that is a uniform multiplicity, and the basis is rebased;
        # anything else is a multiplicity surprise and refuses.
        k = captured / expected if expected else 0.0
        if k and abs(k - round(k)) <= RATIO_TOLERANCE and round(k) >= 1:
            captured = captured / round(k)
        else:
            raise WindowError(
                f"program {pid} shows {captured:.0f} executions where a "
                f"window of {iters} calls at {expected_per_call} executions "
                f"per call can hold at most {expected:.0f}. More executions "
                f"than the window performed is a multiplicity this "
                f"measurement does not understand, not a lossy capture.")
    coverage = captured / expected if expected else 0.0
    if captured < MIN_CAPTURED_EXECUTIONS:
        raise WindowError(
            f"program {pid} was captured {captured:.0f} times, under the "
            f"{MIN_CAPTURED_EXECUTIONS} executions a figure needs to be worth "
            f"averaging ({coverage:.1%} of the {expected:.0f} the window "
            f"performed). The window produced no number.")
    if coverage < MIN_COVERAGE:
        raise WindowError(
            f"program {pid} was captured {captured:.0f} of {expected:.0f} "
            f"times ({coverage:.1%}), under the {MIN_COVERAGE:.0%} of the "
            f"window a figure needs. The window produced no number.")
    out, ambiguous = [], []
    for r in in_program:
        occ = float(r.get("occurrences", 0)) or 1.0
        ratio = occ / captured
        row = dict(r)
        row["ops_per_execution"] = max(1.0, round(ratio))
        row["ambiguous_multiplicity"] = abs(ratio - round(ratio)) > \
            RATIO_TOLERANCE
        if row["ambiguous_multiplicity"]:
            ambiguous.append(f"{row.get('hlo_op_name')} occurrences "
                             f"{occ:.0f} against basis {captured:.0f} "
                             f"(ratio {ratio:.2f})")
        # Self-normalizing: the operation's own occurrences, not the window's.
        row["us_per_execution"] = (float(r.get("total_self_time", 0)) / occ
                                   * row["ops_per_execution"])
        out.append(row)
    if ambiguous:
        # A warning here, or a flag that is computed and then dropped, is not
        # enough. A figure scaled by a guessed multiplicity is not a
        # measurement, so the point refuses and names the operations.
        raise WindowError(
            f"{len(ambiguous)} operation(s) in program {pid} sit too far "
            f"from an integer multiple of the window basis to scale "
            f"({'; '.join(ambiguous[:4])}): dropped events and genuine "
            f"multiplicity cannot be told apart, so the figure would be a "
            f"guess. The window produced no number.")
    basis = {
        "iters": int(iters),
        "expected_executions_per_call": expected_per_call,
        "expected_executions": int(expected),
        "captured_executions": int(captured),
        "coverage": coverage,
        "derivation": ("full-window" if captured >= expected
                       else "self-normalized per execution"),
        "programs_in_window": len(census),
        "program_census": census,
    }
    return sorted(out, key=lambda r: -r["us_per_execution"]), pid, basis


def program_totals(logdir, anchor, iters, executions_per_call=None):
    """The whole-layer figure, the kernel's own figure, and the operations.

    WHAT THE FIGURE IS. Device self time of ONE EXECUTION of the anchor's
    program: the kernel, the router where the arm runs it inside the
    program, every collective inside the program, and every cast, copy and
    reshape inside it. On a sharded mesh one call executes the program once
    per device, and the profiler counts each device, so this is the per-call
    figure of one device's share of the layer. Every arm in a manifest is
    read the same way, and the basis travels with the figure:

    whole_layer_us       sum over every operation in the program
    kernel_self_us       sum over the anchor-prefixed operations only,
                         reported beside the whole-layer figure and never
                         ranked on
    expected_executions  what the window performed: its calls times the
                         executions one call performs, one per device
    executions           what the capture actually holds, which at the heavy
                         counts is fewer, because the profiler drops events
    coverage             captured over expected, as a fraction
    derivation           full-window when the capture holds every execution,
                         self-normalized per execution when it holds fewer.
                         The arithmetic is the same either way; the label
                         says how much of the window the figure was read from
    anchor_ops_per_execution  how many anchor-matching operations run in one
                         execution, which is 2 for the production pair and
                         is a property of the arm, not an error
    ops                  every row, so the window census can read the
                         operation names the trace actually holds

    The window's call count is required, not optional: dividing by an
    occurrence count without knowing what the window held is how a figure
    gets divided by the wrong number.

    executions_per_call overrides the device count for an arm whose program
    runs more than once per call per device. An arm that needs it declares
    it in its contract; without a declaration, more executions than one per
    device raises rather than being averaged into a figure nobody can read.
    """
    rows, pid, basis = program_ops(logdir, anchor, iters,
                                   executions_per_call)
    ambiguous = [r.get("hlo_op_name") for r in rows
                 if r.get("ambiguous_multiplicity")]
    anchor_rows = [r for r in rows
                   if str(r.get("hlo_op_name", "")).startswith(anchor)]
    return {
        "whole_layer_us": sum(r["us_per_execution"] for r in rows),
        "kernel_self_us": sum(r["us_per_execution"] for r in anchor_rows),
        "n_programs": 1,
        "program_id": pid,
        "iters": basis["iters"],
        "executions": basis["captured_executions"],
        "expected_executions": basis["expected_executions"],
        "executions_per_call": basis["expected_executions_per_call"],
        "coverage": basis["coverage"],
        "derivation": basis["derivation"],
        "programs_in_window": basis["programs_in_window"],
        "program_census": basis["program_census"],
        "anchor_ops_per_execution": int(sum(r["ops_per_execution"]
                                            for r in anchor_rows)),
        "ambiguous_ops": ambiguous,
        "basis": (f"device self time of one execution of program {pid}, "
                  f"{basis['derivation']}; the window held {basis['iters']} "
                  f"calls at {basis['expected_executions_per_call']} "
                  f"executions per call, "
                  f"{basis['expected_executions']} expected and "
                  f"{basis['captured_executions']} captured "
                  f"({basis['coverage']:.1%})"),
        "ops": rows,
        "top_ops": [{"op": r.get("hlo_op_name"),
                     "us": r["us_per_execution"],
                     "ops_per_execution": r["ops_per_execution"],
                     "category": r.get("category")} for r in rows[:12]],
    }
