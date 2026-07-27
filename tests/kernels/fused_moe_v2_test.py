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
"""Unit tests for the fused expert-parallel MoE kernel's host-side pieces.

Covers pallas_select, plan_ragged_dispatch and rowquant_fp8."""

import contextlib
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental import pallas as pl

from tpu_inference.kernels.fused_moe.v2.kernel import (FP8_MAX, ROWBLK,
                                                       align_up,
                                                       plan_ragged_dispatch,
                                                       rowquant_fp8)
from tpu_inference.kernels.fused_moe.v2.router_ops import NEG, pallas_select

jax.config.parse_flags_with_absl()

# Served geometry, for the device-marked cases.
SERVED_EXPERTS = 512
SERVED_TOPK = 10
SERVED_EP = 8


@contextlib.contextmanager
def interpret_pallas():
    """Run pallas_call under interpret for the duration of the block."""
    real = pl.pallas_call

    def interpreted(*args, **kwargs):
        kwargs.setdefault("interpret", True)
        return real(*args, **kwargs)

    with mock.patch.object(pl, "pallas_call", interpreted):
        yield


def ref_top_k(scores, topk):
    """Reference selection: XLA's sort-based top-k."""
    weights, indices = jax.lax.top_k(scores, topk)
    return np.asarray(weights), np.asarray(indices).astype(np.int32)


def ref_rowquant_scale(x_bf16):
    """Reference row scale from an f32-widened reduce."""
    x32 = x_bf16.astype(jnp.float32)
    amax = jnp.max(jnp.abs(x32), axis=-1, keepdims=True)
    return amax / FP8_MAX


def softmax_scores(logits):
    return jax.nn.softmax(jnp.asarray(logits, jnp.float32), axis=-1)


def scatter_target_counts(didx, rows_alloc):
    """How many routed pairs land on each slab row."""
    counts = jnp.zeros((rows_alloc + 1, ), jnp.int32).at[didx].add(1)
    return np.asarray(counts)


def make_plan(idx, weights, *, e_total, ep, block, tile_m):
    """The plan as the serving layer builds it: per-shard strided slabs."""
    tokens, topk = idx.shape
    stride = align_up(tokens * topk + (ROWBLK - 1) * e_total + tile_m, tile_m)
    return plan_ragged_dispatch(jnp.asarray(idx, jnp.int32),
                                jnp.asarray(weights, jnp.float32),
                                e_total=e_total,
                                ep=ep,
                                t_local=tokens // ep,
                                block=block,
                                tile_m=tile_m,
                                shard_stride=stride)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RouterSelectionTest(jtu.JaxTestCase):
    """pallas_select: the docstring contract and the NaN-row guarantee."""

    @parameterized.named_parameters(
        dict(testcase_name="_one_block",
             rows=32,
             experts=64,
             topk=4,
             block_rows=32),
        dict(testcase_name="_multi_block",
             rows=64,
             experts=32,
             topk=6,
             block_rows=16),
        dict(testcase_name="_topk_one",
             rows=16,
             experts=128,
             topk=1,
             block_rows=16),
    )
    def test_matches_lax_top_k_on_ordinary_scores(self, rows, experts, topk,
                                                  block_rows):
        """The stated contract: same weights, same indices, same bits."""
        rng = np.random.default_rng(0)
        scores = softmax_scores(rng.normal(size=(rows, experts)))
        with interpret_pallas():
            weights, indices = pallas_select(scores,
                                             topk=topk,
                                             block_rows=block_rows)
        want_w, want_i = ref_top_k(scores, topk)
        np.testing.assert_array_equal(np.asarray(indices), want_i)
        np.testing.assert_array_equal(np.asarray(weights), want_w)

    def test_ties_break_on_the_lowest_index(self):
        """Repeated scores select the lowest columns, as lax.top_k does."""
        rng = np.random.default_rng(1)
        rows, experts, topk = 32, 24, 8
        logits = rng.choice(np.array([-1.0, 0.0, 1.0], np.float32),
                            size=(rows, experts))
        scores = softmax_scores(logits)
        # Guard the premise: without ties this test proves nothing.
        self.assertLess(len(np.unique(np.asarray(scores)[0])), experts)
        with interpret_pallas():
            weights, indices = pallas_select(scores,
                                             topk=topk,
                                             block_rows=rows)
        want_w, want_i = ref_top_k(scores, topk)
        np.testing.assert_array_equal(np.asarray(indices), want_i)
        np.testing.assert_array_equal(np.asarray(weights), want_w)

    def test_nan_row_indices_stay_inside_the_expert_range(self):
        """A row of NaN scores must still name real experts."""
        rng = np.random.default_rng(2)
        rows, experts, topk = 32, 64, 6
        logits = rng.normal(size=(rows, experts)).astype(np.float32)
        nan_row = 7
        logits[nan_row, 3] = np.inf
        scores = softmax_scores(logits)
        self.assertTrue(np.isnan(np.asarray(scores)[nan_row]).all())
        with interpret_pallas():
            _, indices = pallas_select(scores, topk=topk, block_rows=rows)
        indices = np.asarray(indices)
        self.assertTrue((indices >= 0).all())
        self.assertTrue((indices < experts).all(),
                        f"out-of-range expert ids: {indices.max()}")

    def test_nan_row_selects_the_lowest_expert_with_defined_weights(self):
        """A NaN row selects expert 0 with sentinel weights, and
        renormalizing those weights gives exactly zero."""
        rng = np.random.default_rng(3)
        rows, experts, topk = 16, 32, 5
        logits = rng.normal(size=(rows, experts)).astype(np.float32)
        nan_row = 4
        logits[nan_row, 0] = np.inf
        scores = softmax_scores(logits)
        with interpret_pallas():
            weights, indices = pallas_select(scores,
                                             topk=topk,
                                             block_rows=rows)
        np.testing.assert_array_equal(
            np.asarray(indices)[nan_row], np.zeros((topk, ), np.int32))
        row_w = np.asarray(weights)[nan_row]
        np.testing.assert_array_equal(row_w, np.full((topk, ), NEG,
                                                     np.float32))
        self.assertTrue(np.isfinite(row_w).all())
        renormalized = weights / weights.sum(axis=-1, keepdims=True)
        renormalized = np.asarray(renormalized)
        self.assertTrue(np.isfinite(renormalized).all())
        np.testing.assert_array_equal(renormalized[nan_row],
                                      np.zeros((topk, ), np.float32))

    def test_nan_row_leaves_every_other_row_bitwise_unchanged(self):
        """Token locality: one bad row may not touch its neighbours."""
        rng = np.random.default_rng(4)
        rows, experts, topk = 32, 64, 6
        clean = rng.normal(size=(rows, experts)).astype(np.float32)
        poisoned = clean.copy()
        nan_row = 11
        poisoned[nan_row, 5] = np.inf
        with interpret_pallas():
            w_clean, i_clean = pallas_select(softmax_scores(clean),
                                             topk=topk,
                                             block_rows=rows)
            w_bad, i_bad = pallas_select(softmax_scores(poisoned),
                                         topk=topk,
                                         block_rows=rows)
        keep = [r for r in range(rows) if r != nan_row]
        np.testing.assert_array_equal(
            np.asarray(i_bad)[keep],
            np.asarray(i_clean)[keep])
        np.testing.assert_array_equal(
            np.asarray(w_bad)[keep],
            np.asarray(w_clean)[keep])

    def test_served_geometry_matches_lax_top_k(self):
        """The served router shape, on the device it is served on."""
        if not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+")
        rng = np.random.default_rng(5)
        rows = 8192 // SERVED_EP
        scores = softmax_scores(rng.normal(size=(rows, SERVED_EXPERTS)))
        weights, indices = pallas_select(scores,
                                         topk=SERVED_TOPK,
                                         block_rows=256)
        want_w, want_i = ref_top_k(scores, SERVED_TOPK)
        np.testing.assert_array_equal(np.asarray(indices), want_i)
        np.testing.assert_array_equal(np.asarray(weights), want_w)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class DispatchPlanTest(jtu.JaxTestCase):
    """plan_ragged_dispatch: one slab row per routed pair, and no more."""

    @parameterized.named_parameters(
        dict(testcase_name="_e8_ep4",
             e_total=8,
             ep=4,
             tokens=32,
             topk=2,
             block=8,
             tile_m=32),
        dict(testcase_name="_e16_ep8",
             e_total=16,
             ep=8,
             tokens=64,
             topk=4,
             block=32,
             tile_m=32),
        dict(testcase_name="_e4_ep2",
             e_total=4,
             ep=2,
             tokens=16,
             topk=3,
             block=8,
             tile_m=16),
        dict(testcase_name="_e64_ep8",
             e_total=64,
             ep=8,
             tokens=64,
             topk=6,
             block=16,
             tile_m=64),
    )
    def test_scatter_targets_are_unique_for_random_routings(
            self, e_total, ep, tokens, topk, block, tile_m):
        for seed in range(4):
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, e_total, size=(tokens, topk))
            weights = rng.random((tokens, topk)).astype(np.float32)
            plan = make_plan(idx,
                             weights,
                             e_total=e_total,
                             ep=ep,
                             block=block,
                             tile_m=tile_m)
            counts = scatter_target_counts(plan["didx"], plan["rows_alloc"])
            self.assertEqual(
                int(counts.max()), 1,
                f"two routed pairs share a slab row (seed {seed})")
            self.assertEqual(int(counts.sum()), tokens * topk)
            self.assertEqual(int(plan["stride_over"]), 0)

    @parameterized.named_parameters(
        dict(testcase_name="_first_expert", expert=0),
        dict(testcase_name="_last_expert", expert=7),
        dict(testcase_name="_mid_shard_expert", expert=3),
    )
    def test_scatter_targets_are_unique_when_every_token_picks_one_expert(
            self, expert):
        """The skewed extreme: one expert owns every routed row."""
        e_total, ep, tokens, topk = 8, 4, 32, 2
        rng = np.random.default_rng(6)
        idx = np.full((tokens, topk), expert, np.int32)
        weights = rng.random((tokens, topk)).astype(np.float32)
        plan = make_plan(idx,
                         weights,
                         e_total=e_total,
                         ep=ep,
                         block=8,
                         tile_m=32)
        counts = scatter_target_counts(plan["didx"], plan["rows_alloc"])
        self.assertEqual(int(counts.max()), 1)
        self.assertEqual(int(counts.sum()), tokens * topk)
        self.assertEqual(int(plan["stride_over"]), 0)

    def test_tables_recover_every_pair_exactly(self):
        """What injectivity buys: each row holds one token and one weight."""
        e_total, ep, tokens, topk = 16, 8, 64, 4
        rng = np.random.default_rng(7)
        idx = rng.integers(0, e_total, size=(tokens, topk))
        weights = rng.random((tokens, topk)).astype(np.float32)
        plan = make_plan(idx,
                         weights,
                         e_total=e_total,
                         ep=ep,
                         block=32,
                         tile_m=32)
        didx = np.asarray(plan["didx"])
        token_of_pair = np.arange(tokens * topk) // topk
        np.testing.assert_array_equal(
            np.asarray(plan["token_gather"])[didx], token_of_pair)
        np.testing.assert_array_equal(
            np.asarray(plan["w_row"])[didx], weights.reshape(-1))

    def test_nan_row_routing_keeps_the_plan_injective(self):
        """Router output for a batch carrying a NaN row still plans
        injectively, with every token's table entries recovered."""
        e_total, ep, tokens, topk = 16, 8, 64, 4
        rng = np.random.default_rng(8)
        logits = rng.normal(size=(tokens, e_total)).astype(np.float32)
        nan_row = 21
        logits[nan_row, 2] = np.inf
        scores = softmax_scores(logits)
        with interpret_pallas():
            weights, idx = pallas_select(scores, topk=topk, block_rows=32)
        plan = make_plan(np.asarray(idx),
                         np.asarray(weights),
                         e_total=e_total,
                         ep=ep,
                         block=32,
                         tile_m=32)
        counts = scatter_target_counts(plan["didx"], plan["rows_alloc"])
        self.assertEqual(int(counts.max()), 1)
        didx = np.asarray(plan["didx"])
        token_of_pair = np.arange(tokens * topk) // topk
        np.testing.assert_array_equal(
            np.asarray(plan["token_gather"])[didx], token_of_pair)
        np.testing.assert_array_equal(
            np.asarray(plan["w_row"])[didx],
            np.asarray(weights).reshape(-1))

    @pytest.mark.xfail(
        strict=True,
        reason="the plan has no sink for an out-of-range expert id yet; "
        "such a pair lands on the first slab row, which expert 0 already "
        "owns. The router upstream of this plan guarantees in-range ids, "
        "so nothing served reaches it -- this case turns green the day "
        "the plan sends them to the dropped sentinel row instead.")
    def test_an_out_of_range_expert_id_goes_to_the_sink(self):
        """An expert id outside [0, E) should land on the sentinel row."""
        e_total, ep, tokens, topk = 8, 4, 32, 2
        rng = np.random.default_rng(9)
        idx = rng.integers(0, e_total, size=(tokens, topk))
        idx[5, 0] = e_total
        weights = rng.random((tokens, topk)).astype(np.float32)
        plan = make_plan(idx,
                         weights,
                         e_total=e_total,
                         ep=ep,
                         block=8,
                         tile_m=32)
        counts = scatter_target_counts(plan["didx"], plan["rows_alloc"])
        self.assertEqual(int(counts.max()), 1)
        didx = np.asarray(plan["didx"])
        token_of_pair = np.arange(tokens * topk) // topk
        real = didx != int(plan["_sink"])
        # The sink index is one past the last row token_gather holds, so
        # the indices are masked before the gather rather than after it.
        landed = np.asarray(plan["token_gather"])[np.where(real, didx, 0)]
        np.testing.assert_array_equal(landed[real], token_of_pair[real])

    def test_the_scatter_reserves_one_dropped_sentinel_row(self):
        """The tables allocate rows_alloc + 1, drop the last row, and
        report it as _sink."""
        e_total, ep, tokens, topk = 8, 4, 32, 2
        rng = np.random.default_rng(10)
        idx = rng.integers(0, e_total, size=(tokens, topk))
        weights = rng.random((tokens, topk)).astype(np.float32)
        plan = make_plan(idx,
                         weights,
                         e_total=e_total,
                         ep=ep,
                         block=8,
                         tile_m=32)
        rows_alloc = int(plan["rows_alloc"])
        self.assertEqual(int(plan["_sink"]), rows_alloc)
        self.assertEqual(
            np.asarray(plan["token_gather"]).shape, (rows_alloc, ))
        # A write at the sentinel index changes nothing the tables hand out.
        base = jnp.zeros((rows_alloc + 1, ), jnp.int32)
        without = np.asarray(base.at[jnp.asarray([3])].add(5)[:-1])
        with_sink = np.asarray(base.at[jnp.asarray([3, rows_alloc])].add(
            jnp.asarray([5, 999]))[:-1])
        np.testing.assert_array_equal(with_sink, without)

    def test_served_geometry_plan_is_injective(self):
        """The served routing shape (8192 tokens, 512 experts, top-10)."""
        tokens = 8192
        rng = np.random.default_rng(11)
        idx = rng.integers(0, SERVED_EXPERTS, size=(tokens, SERVED_TOPK))
        weights = rng.random((tokens, SERVED_TOPK)).astype(np.float32)
        plan = make_plan(idx,
                         weights,
                         e_total=SERVED_EXPERTS,
                         ep=SERVED_EP,
                         block=256,
                         tile_m=128)
        counts = scatter_target_counts(plan["didx"], plan["rows_alloc"])
        self.assertEqual(int(counts.max()), 1)
        self.assertEqual(int(plan["stride_over"]), 0)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RowQuantTest(jtu.JaxTestCase):
    """rowquant_fp8: exact scales, no overflow, and the stated flushes."""

    @parameterized.named_parameters(
        dict(testcase_name="_narrow", rows=8, cols=128),
        dict(testcase_name="_served_row", rows=16, cols=4096),
        dict(testcase_name="_wide_range", rows=32, cols=512),
    )
    def test_scale_is_exact_against_an_f32_widened_reduce(self, rows, cols):
        """abs and max are exact in bf16, so the f32 scale is identical."""
        rng = np.random.default_rng(12)
        mags = np.exp(rng.uniform(-12, 12, size=(rows, 1)))
        x = (rng.normal(size=(rows, cols)) * mags).astype(np.float32)
        x_bf16 = jnp.asarray(x).astype(jnp.bfloat16)
        _, scale = rowquant_fp8(x_bf16)
        np.testing.assert_array_equal(
            np.asarray(scale.view(jnp.int32)),
            np.asarray(ref_rowquant_scale(x_bf16).view(jnp.int32)))

    def test_zero_row_scale_is_zero_and_the_row_stays_zero(self):
        x = jnp.zeros((4, 128), jnp.bfloat16)
        quantized, scale = rowquant_fp8(x)
        np.testing.assert_array_equal(np.asarray(scale),
                                      np.zeros((4, 1), np.float32))
        np.testing.assert_array_equal(
            np.asarray(quantized.astype(jnp.float32)),
            np.zeros((4, 128), np.float32))

    def test_no_fp8_overflow_through_the_bf16_rounded_inverse_scale(self):
        """Rounding the inverse scale to bf16 never pushes a row past 448."""
        rng = np.random.default_rng(13)
        for seed_row in range(64):
            mags = np.exp(rng.uniform(-20, 20, size=(1, 1)))
            x = (rng.normal(size=(1, 256)) * mags).astype(np.float32)
            x_bf16 = jnp.asarray(x).astype(jnp.bfloat16)
            quantized, _ = rowquant_fp8(x_bf16)
            widened = np.asarray(quantized.astype(jnp.float32))
            self.assertTrue(
                np.isfinite(widened).all(), f"row {seed_row} overflowed e4m3")
            self.assertLessEqual(float(np.abs(widened).max()), FP8_MAX)

    def test_no_fp8_overflow_on_rows_whose_maximum_sits_at_the_boundary(self):
        """The adversarial case: every bf16 row maximum in a window around
        448, where the inverse scale is as close to 1 as bf16 allows."""
        candidates = np.unique(
            np.asarray(
                jnp.asarray(np.linspace(440.0, 456.0, 2048,
                                        dtype=np.float32)).astype(
                                            jnp.bfloat16).astype(jnp.float32)))
        rng = np.random.default_rng(14)
        for peak in candidates:
            row = (rng.random((1, 128)) * peak).astype(np.float32)
            row[0, 0] = peak
            x_bf16 = jnp.asarray(row).astype(jnp.bfloat16)
            quantized, scale = rowquant_fp8(x_bf16)
            widened = np.asarray(quantized.astype(jnp.float32))
            self.assertTrue(
                np.isfinite(widened).all(), f"peak {peak} overflowed e4m3")
            self.assertLessEqual(float(np.abs(widened).max()), FP8_MAX)
            self.assertGreater(float(scale[0, 0]), 0.0)

    def test_an_infinity_erases_the_rest_of_its_row(self):
        """One infinity zeroes every other value in its row, NaNs its own
        lane, and leaves the rows beside it untouched."""
        rng = np.random.default_rng(15)
        rows, cols = 4, 128
        x = rng.normal(size=(rows, cols)).astype(np.float32)
        clean_q, clean_s = rowquant_fp8(jnp.asarray(x).astype(jnp.bfloat16))
        poisoned = x.copy()
        inf_row, inf_col = 1, 9
        poisoned[inf_row, inf_col] = np.inf
        quantized, scale = rowquant_fp8(
            jnp.asarray(poisoned).astype(jnp.bfloat16))
        widened = np.asarray(quantized.astype(jnp.float32))
        self.assertTrue(np.isinf(float(scale[inf_row, 0])))
        others = np.delete(widened[inf_row], inf_col)
        np.testing.assert_array_equal(others, np.zeros_like(others))
        self.assertTrue(np.isnan(widened[inf_row, inf_col]))
        keep = [r for r in range(rows) if r != inf_row]
        np.testing.assert_array_equal(
            widened[keep],
            np.asarray(clean_q.astype(jnp.float32))[keep])
        np.testing.assert_array_equal(
            np.asarray(scale)[keep],
            np.asarray(clean_s)[keep])

    def test_a_nan_row_stays_nan_and_stays_local(self):
        rng = np.random.default_rng(16)
        rows, cols = 4, 128
        x = rng.normal(size=(rows, cols)).astype(np.float32)
        clean_q, clean_s = rowquant_fp8(jnp.asarray(x).astype(jnp.bfloat16))
        poisoned = x.copy()
        nan_row = 2
        poisoned[nan_row, 5] = np.nan
        quantized, scale = rowquant_fp8(
            jnp.asarray(poisoned).astype(jnp.bfloat16))
        widened = np.asarray(quantized.astype(jnp.float32))
        self.assertTrue(np.isnan(float(scale[nan_row, 0])))
        self.assertTrue(np.isnan(widened[nan_row]).all())
        keep = [r for r in range(rows) if r != nan_row]
        np.testing.assert_array_equal(
            widened[keep],
            np.asarray(clean_q.astype(jnp.float32))[keep])
        np.testing.assert_array_equal(
            np.asarray(scale)[keep],
            np.asarray(clean_s)[keep])


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
