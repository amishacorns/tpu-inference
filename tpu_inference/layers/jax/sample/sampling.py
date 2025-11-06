import functools

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.v1.outputs import LogprobsTensors

from tpu_inference.layers.jax.binary_search import topk_mask, topp_mask
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata
from tpu_inference.layers.jax.sharding import ShardingAxisName

_SAMPLING_EPS = 1e-5


@functools.partial(
    jax.jit,
    static_argnames=["mesh"],
)
def sample(
    rng: jax.Array,
    mesh: Mesh,
    logits: jax.Array,
    tpu_sampling_metadata: TPUSupportedSamplingMetadata,
) -> jax.Array:
    # (B, vocab_size)
    if tpu_sampling_metadata.do_sampling:
        # Unshard the logits explicity to avoid latency increase.
        logits = jax.lax.with_sharding_constraint(
            logits, NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None)))
    greedy_sampled = jnp.argmax(logits, axis=-1)
    if not tpu_sampling_metadata.do_sampling:
        return greedy_sampled

    logits = logits.astype(jnp.float32)
    logits = topk_mask(logits, tpu_sampling_metadata.top_k, replace_val=-1e12)
    logits = topp_mask(logits, tpu_sampling_metadata.top_p, replace_val=-1e12)

    temperatures = tpu_sampling_metadata.temperature.astype(logits.dtype)
    temperatures = jnp.expand_dims(temperatures, axis=-1)
    logits /= temperatures

    # (batch_size,)
    # Derive per-request keys by splitting and folding in per-request seeds.
    # `seeds` is always provided when do_sampling=True; entries < 0 mean "no seed".
    seeds = tpu_sampling_metadata.seeds
    batch_size = logits.shape[0]
    keys = jax.random.split(rng, batch_size)
    # Ensure integer dtype for fold_in
    seeds = seeds.astype(jnp.int32)
    # Use -1 as the only sentinel for "no per-request seed" to match vLLM semantics
    def fold_if_needed(key, seed):
        return jax.lax.cond(
            (seed != jnp.int32(-1)),
            lambda k: jax.random.fold_in(k, seed),
            lambda k: k,
            key,
        )
    keys = jax.vmap(fold_if_needed)(keys, seeds)
    # Categorical expects a single key; vmap over batch to use per-request keys
    def cat_one(k, l):
        return jax.random.categorical(k, l)
    next_tokens = jax.vmap(cat_one)(keys, logits)
    # Note: avoid using the sample result when temperature < _SAMPLING_EPS
    # If temperature < 0, logits /= temperatures will flip the result, causing error.
    return jnp.where(tpu_sampling_metadata.temperature < _SAMPLING_EPS,
                     greedy_sampled, next_tokens)


def compute_logprobs(logits: jax.Array) -> jax.Array:
    return jax.nn.log_softmax(logits, axis=-1)


def gather_logprobs(
    logprobs: jax.Array,
    token_ids: jax.Array,
    num_logprobs: int,
) -> LogprobsTensors:
    """
    Gather logprobs for topk and sampled/prompt token.

    Args:
        logprobs: (num tokens) x (vocab) tensor
        token_ids: prompt tokens (if prompt logprobs)
                    or sampled tokens (if sampled
                    logprobs); 1D token ID tensor
                    with (num tokens) elements
        num_logprobs: minimum number of logprobs to
                    retain per token


    Returns:
        Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
        Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
        Sampled token rank tensor, (num tokens)
    """
    # Find the topK values.
    topk_logprobs, topk_indices = jax.lax.top_k(logprobs, k=num_logprobs)

    # Get with the logprob of the prompt or sampled token.
    token_ids = jnp.expand_dims(token_ids, axis=-1)
    token_logprobs = jnp.take_along_axis(logprobs, token_ids, axis=-1)

    # Compute the ranks of the actual token.
    token_ranks = jnp.sum(logprobs >= token_logprobs, axis=-1)

    # Concatenate together with the topk.
    indices = jnp.concatenate((token_ids, topk_indices), axis=1)
    logprobs = jnp.concatenate((token_logprobs, topk_logprobs), axis=1)

    # Use int32 to reduce the tensor size.
    indices = jnp.int32(indices)

    return LogprobsTensors(indices, logprobs, token_ranks)
