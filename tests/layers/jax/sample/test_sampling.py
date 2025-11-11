import jax
import jax.numpy as jnp
import numpy as np
from vllm.v1.outputs import LogprobsTensors

from tpu_inference.layers.jax.sample.sampling import (compute_logprobs,
                                                      gather_logprobs, sample)
from tpu_inference.layers.jax.sample.sampling_metadata import TPUSupportedSamplingMetadata
from tpu_inference.layers.jax.binary_search import topp_mask, topk_mask
from jax.sharding import Mesh


def _single_device_mesh() -> Mesh:
    devices = sorted(jax.devices(), key=lambda d: d.id)[:1]
    arr = np.asarray(devices).reshape((1, 1))
    return Mesh(arr, axis_names=("data", "model"))


def _run_sample(logits: jnp.ndarray, metadata: TPUSupportedSamplingMetadata, key: jax.Array) -> jnp.ndarray:
    mesh = _single_device_mesh()
    return sample(rng=key, mesh=mesh, logits=logits, tpu_sampling_metadata=metadata)


class TestSampling:

    def test_compute_logprobs(self):
        logits = jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
                           dtype=jnp.float32)
        logprobs = compute_logprobs(logits)

        # Expected values computed with scipy.special.log_softmax
        expected_logprobs = np.array(
            [
                [-2.40760596, -1.40760596, -0.40760596],
                [-0.40760596, -1.40760596, -2.40760596],
            ],
            dtype=np.float32,
        )
        assert np.allclose(logprobs, expected_logprobs, atol=1e-6)

    def test_gather_logprobs(self):
        logprobs = jnp.array(
            [
                [-2.40760596, -1.40760596, -0.40760596, -3.40760596],
                [-0.40760596, -1.40760596, -2.40760596, -3.40760596],
            ],
            dtype=jnp.float32,
        )
        token_ids = jnp.array([2, 0], dtype=jnp.int32)
        num_logprobs = 2

        result: LogprobsTensors = gather_logprobs(logprobs, token_ids,
                                                  num_logprobs)

        # check indices
        expected_indices = np.array(
            [
                [2, 2, 1],  # token id 2, top-k are 2, 1
                [0, 0, 1],  # token id 0, top-k are 0, 1
            ],
            dtype=np.int32,
        )
        assert np.array_equal(result.logprob_token_ids, expected_indices)

        # check logprobs
        expected_logprobs_values = np.array(
            [
                [-0.40760596, -0.40760596, -1.40760596],
                [-0.40760596, -0.40760596, -1.40760596],
            ],
            dtype=np.float32,
        )
        assert np.allclose(result.logprobs,
                           expected_logprobs_values,
                           atol=1e-6)

        # check ranks
        expected_ranks = np.array([1, 1], dtype=np.int32)
        assert np.array_equal(result.selected_token_ranks, expected_ranks)

    def test_gather_logprobs_with_ties(self):
        logprobs = jnp.array(
            [
                [-1.0, -1.0, -2.0, -2.0],
            ],
            dtype=jnp.float32,
        )
        token_ids = jnp.array([1], dtype=jnp.int32)
        num_logprobs = 3

        result: LogprobsTensors = gather_logprobs(logprobs, token_ids,
                                                  num_logprobs)

        # check logprobs
        expected_logprobs_values = np.array(
            [
                [-1.0, -1.0, -1.0, -2.0],
            ],
            dtype=np.float32,
        )
        assert np.allclose(result.logprobs,
                           expected_logprobs_values,
                           atol=1e-6)

        # check ranks
        # rank of token 1 is 2 because there are 2 values >= -1.0
        expected_ranks = np.array([2], dtype=np.int32)
        assert np.array_equal(result.selected_token_ranks, expected_ranks)

        # check indices
        # The order of tied elements is not guaranteed.
        # token id is 1. top-k indices are a permutation of {0, 1, 2} or {0, 1, 3}.
        assert result.logprob_token_ids[0, 0] == 1
        top_k_indices = sorted(result.logprob_token_ids[0, 1:].tolist())
        assert top_k_indices == [0, 1, 2] or top_k_indices == [0, 1, 3]

    def test_greedy_sampling_disabled(self):
        logits = jnp.array([[1.0, 3.0, 2.0]], dtype=jnp.float32)
        metadata = TPUSupportedSamplingMetadata(
            do_sampling=False,
            logprobs=False,
            top_k=None,
            top_p=None,
            temperature=jnp.array([1.0], dtype=jnp.float32),
        )
        key = jax.random.PRNGKey(0)
        out = _run_sample(logits, metadata, key)
        assert int(out[0]) == 1  # argmax index (value 3.0)

    def test_temperature_below_eps_is_greedy(self):
        logits = jnp.array([[1.0, 3.0, 2.0]], dtype=jnp.float32)
        # do_sampling True but temperature very small (< _SAMPLING_EPS)
        metadata = TPUSupportedSamplingMetadata(
            do_sampling=True,
            logprobs=False,
            top_k=None,
            top_p=None,
            temperature=jnp.array([1e-7], dtype=jnp.float32),
        )
        key = jax.random.PRNGKey(1)
        out = _run_sample(logits, metadata, key)
        assert int(out[0]) == 1  # still argmax

    def test_top_k_membership(self):
        # vocab 16, choose top_k=3
        logits = jax.random.normal(jax.random.PRNGKey(7), (1, 16))
        top_k = jnp.array([3], dtype=jnp.int32)
        meta = TPUSupportedSamplingMetadata(
            do_sampling=True,
            logprobs=False,
            top_k=top_k,
            top_p=None,
            temperature=jnp.array([1.0], dtype=jnp.float32),
        )
        key = jax.random.PRNGKey(99)
        # Identify top-3 indices manually
        top_indices = set(np.asarray(jnp.argsort(-logits[0])[:3]))
        token = int(_run_sample(logits, meta, key)[0])
        assert token in top_indices

    def test_top_p_membership(self):
        logits = jax.random.normal(jax.random.PRNGKey(8), (1, 16))
        top_p_val = 0.85
        meta = TPUSupportedSamplingMetadata(
            do_sampling=True,
            logprobs=False,
            top_k=None,
            top_p=jnp.array([top_p_val], dtype=jnp.float32),
            temperature=jnp.array([1.0], dtype=jnp.float32),
        )
        key = jax.random.PRNGKey(123)
        masked = topp_mask(logits, top_p_val, replace_val=-1e12)
        allowed = set(np.where(np.asarray(masked[0]) != -1e12)[0])
        token = int(_run_sample(logits, meta, key)[0])
        assert token in allowed
        assert len(allowed) > 1

    def test_combined_top_k_top_p_intersection(self):
        logits = jax.random.normal(jax.random.PRNGKey(9), (1, 16))
        top_k_val = 5
        top_p_val = 0.7
        meta = TPUSupportedSamplingMetadata(
            do_sampling=True,
            logprobs=False,
            top_k=jnp.array([top_k_val], dtype=jnp.int32),
            top_p=jnp.array([top_p_val], dtype=jnp.float32),
            temperature=jnp.array([1.0], dtype=jnp.float32),
        )
        key = jax.random.PRNGKey(321)
        # Build intersection
        topk_indices = set(np.asarray(jnp.argsort(-logits[0])[:top_k_val]))
        masked = topp_mask(logits, top_p_val, replace_val=-1e12)
        top_p_indices = set(np.where(np.asarray(masked[0]) != -1e12)[0])
        inter = topk_indices.intersection(top_p_indices)
        assert inter, "Intersection should not be empty"
        token = int(_run_sample(logits, meta, key)[0])
        assert token in inter

    def test_top_k_sentinel_behavior(self):
        # Compare behavior when top_k disabled via None vs sentinel vocab_size
        logits = jax.random.normal(jax.random.PRNGKey(10), (1, 15))
        key = jax.random.PRNGKey(555)
        # Disabled top_k -> None
        meta_none = TPUSupportedSamplingMetadata(
            do_sampling=True,
            logprobs=False,
            top_k=None,
            top_p=None,
            temperature=jnp.array([1.0], dtype=jnp.float32),
        )
        # Sentinel top_k == vocab_size
        vocab_size = logits.shape[1]
        meta_sentinel = TPUSupportedSamplingMetadata(
            do_sampling=True,
            logprobs=False,
            top_k=jnp.array([vocab_size], dtype=jnp.int32),
            top_p=None,
            temperature=jnp.array([1.0], dtype=jnp.float32),
        )
        token_none = int(_run_sample(logits, meta_none, key)[0])
        token_sentinel = int(_run_sample(logits, meta_sentinel, key)[0])
        # With top_k disabled and sentinel == vocab_size
        assert token_none == token_sentinel

    def test_mixed_batch_top_k_rows(self):
        logits = jax.random.normal(jax.random.PRNGKey(11), (4, 30))
        top_k = jnp.array([5, 30, 5, 30], dtype=jnp.int32)  # 30 acts as sentinel (disabled)
        meta = TPUSupportedSamplingMetadata(
            do_sampling=True,
            logprobs=False,
            top_k=top_k,
            top_p=None,
            temperature=jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32),
        )
        key = jax.random.PRNGKey(77)
        out = _run_sample(logits, meta, key)
        # Rows with active k=5 must be within top-5 of their row
        for row in [0, 2]:
            top5 = set(np.asarray(jnp.argsort(-logits[row])[:5]))
            assert int(out[row]) in top5
        # Rows with sentinel should be unconstrained; just check not crashing
        assert out.shape[0] == 4

    def test_mixed_batch_top_p_rows(self):
        logits = jax.random.normal(jax.random.PRNGKey(12), (3, 25))
        top_p = jnp.array([0.8, 1.0, 0.6], dtype=jnp.float32)  # middle row disabled
        meta = TPUSupportedSamplingMetadata(
            do_sampling=True,
            logprobs=False,
            top_k=None,
            top_p=top_p,
            temperature=jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
        )
        key = jax.random.PRNGKey(88)
        out = _run_sample(logits, meta, key)
        # Membership checks for active rows
        for row in [0, 2]:
            masked = topp_mask(logits[row:row+1], float(top_p[row]), replace_val=-1e12)
            allowed = set(np.where(np.asarray(masked[0]) != -1e12)[0])
            assert int(out[row]) in allowed
        # Disabled row (p=1.0) should allow any token; just assert valid index
        assert 0 <= int(out[1]) < logits.shape[1]


    def test_temperature_near_zero_with_masks(self):
        logits = jax.random.normal(jax.random.PRNGKey(15), (1, 20))
        meta = TPUSupportedSamplingMetadata(
            do_sampling=True,
            logprobs=False,
            top_k=jnp.array([10], dtype=jnp.int32),
            top_p=jnp.array([0.9], dtype=jnp.float32),
            # Use very small temperature to trigger greedy behavior deterministically
            temperature=jnp.array([1e-7], dtype=jnp.float32),
        )
        key = jax.random.PRNGKey(700)
        token = int(_run_sample(logits, meta, key)[0])
        argmax_idx = int(jnp.argmax(logits[0]))
        # With near-zero temperature, sampling should be effectively greedy.
        assert token == argmax_idx

    def test_seeded_vs_unseeded_rows_determinism(self):
        # Create logits with identical rows for unseeded correlation check
        key_logits = jax.random.PRNGKey(101)
        logits = jax.random.normal(key_logits, (5, 64))
        logits = logits.at[4].set(logits[3])  # rows 3 and 4 identical

        temperature = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        rng_seeds = jnp.array([123, 123, 456, -1, -1], dtype=jnp.int32)
        rng_steps = jnp.zeros((5,), dtype=jnp.int32)
        meta = TPUSupportedSamplingMetadata(
            do_sampling=True,
            logprobs=False,
            top_k=None,
            top_p=None,
            temperature=temperature,
            rng_seeds=rng_seeds,
            rng_steps=rng_steps,
            has_seeds=True,
        )

        key1 = jax.random.PRNGKey(999)
        out1 = _run_sample(logits, meta, key1)
        out2 = _run_sample(logits, meta, key1)
        # Same global key + same per-request seeds must be identical
        assert np.array_equal(np.asarray(out1), np.asarray(out2))

    def test_per_step_rng_advancement_for_seeded_rows(self):
        # Use uniform logits to minimize collision chance across different steps
        vocab = 1024
        logits = jnp.zeros((1, vocab), dtype=jnp.float32)
        temperature = jnp.array([1.0], dtype=jnp.float32)
        rng_seeds = jnp.array([2024], dtype=jnp.int32)
        meta_step0 = TPUSupportedSamplingMetadata(
            do_sampling=True,
            logprobs=False,
            top_k=None,
            top_p=None,
            temperature=temperature,
            rng_seeds=rng_seeds,
            rng_steps=jnp.array([0], dtype=jnp.int32),
            has_seeds=True,
        )
        meta_step1 = TPUSupportedSamplingMetadata(
            do_sampling=True,
            logprobs=False,
            top_k=None,
            top_p=None,
            temperature=temperature,
            rng_seeds=rng_seeds,
            rng_steps=jnp.array([1], dtype=jnp.int32),
            has_seeds=True,
        )
        key = jax.random.PRNGKey(0)
        t0 = int(_run_sample(logits, meta_step0, key)[0])
        t1 = int(_run_sample(logits, meta_step1, key)[0])
        # With different per-step advancement, sampled token should differ with overwhelming probability
        assert t0 != t1

    def test_temperature_greedy_overrides_seeded_and_unseeded(self):
        # Near-zero temperature forces greedy regardless of seeds
        logits = jax.random.normal(jax.random.PRNGKey(222), (2, 50))
        temperature = jnp.array([1e-8, 1e-8], dtype=jnp.float32)
        rng_seeds = jnp.array([42, -1], dtype=jnp.int32)
        rng_steps = jnp.array([0, 0], dtype=jnp.int32)
        meta = TPUSupportedSamplingMetadata(
            do_sampling=True,
            logprobs=False,
            top_k=None,
            top_p=None,
            temperature=temperature,
            rng_seeds=rng_seeds,
            rng_steps=rng_steps,
            has_seeds=True,
        )
        key = jax.random.PRNGKey(1234)
        out = _run_sample(logits, meta, key)
        assert int(out[0]) == int(jnp.argmax(logits[0]))
        assert int(out[1]) == int(jnp.argmax(logits[1]))

    def test_mixed_masks_with_seeds_deterministic_across_runs(self):
        # Combine top-k and top-p with per-row seeds; outputs should be identical across runs
        logits = jax.random.normal(jax.random.PRNGKey(333), (4, 40))
        top_k = jnp.array([5, 0, 7, 0], dtype=jnp.int32)  # 0 disables top-k
        top_p = jnp.array([0.9, 1.0, 0.8, 1.0], dtype=jnp.float32)  # 1.0 disables top-p
        temperature = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        rng_seeds = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        rng_steps = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        meta = TPUSupportedSamplingMetadata(
            do_sampling=True,
            logprobs=False,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            rng_seeds=rng_seeds,
            rng_steps=rng_steps,
            has_seeds=True,
        )
        out_a = _run_sample(logits, meta, jax.random.PRNGKey(1))
        out_b = _run_sample(logits, meta, jax.random.PRNGKey(2))
        # Same per-request seeds -> identical outputs even if global key differs
        assert np.array_equal(np.asarray(out_a), np.asarray(out_b))