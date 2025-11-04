from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
from tpu_inference.runner.tpu_jax_runner import TPUModelRunner
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from tpu_inference.runner.input_batch_jax import CachedRequestState
from vllm.sampling_params import SamplingParams


class TestTPUJaxRunner:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock(coords=i) for i in range(4)]
        self.mock_mesh = MagicMock()
        self.mock_rng_key = MagicMock()

        with patch('jax.devices', return_value=self.mock_devices), \
             patch('jax.make_mesh', return_value=self.mock_mesh), \
             patch('jax.random.key', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_jax_runner.get_model', return_value=MagicMock()):

            class DummyModelConfig:
                def __init__(self):
                    self.seed = 0
                    self.dtype = 'bfloat16'
                    self.max_model_len = 128
                    self.is_multimodal_model = False
                    self.uses_mrope = False

                def get_sliding_window(self):
                    return 0

                def get_vocab_size(self):
                    return 8

            class DummyCacheConfig:
                def __init__(self):
                    self.block_size = 16
                    self.cache_dtype = 'auto'

            class DummySchedulerConfig:
                def __init__(self):
                    self.max_num_seqs = 16
                    self.max_num_batched_tokens = 16
                    self.async_scheduling = False

            class DummyParallelConfig:
                def __init__(self):
                    self.decode_context_parallel_size = 1

            class DummyLoraConfig:
                def __init__(self):
                    # minimal attribute used by runner to extend vocab size
                    self.lora_extra_vocab_size = 0

            class DummyLoadConfig:
                def __init__(self):
                    # placeholder for fields the runner might access in future
                    self.tensor_parallel_size = 1

            class DummyVllmConfig:
                def __init__(self):
                    self.model_config = DummyModelConfig()
                    self.cache_config = DummyCacheConfig()
                    self.scheduler_config = DummySchedulerConfig()
                    self.parallel_config = DummyParallelConfig()
                    self.lora_config = DummyLoraConfig()
                    self.load_config = DummyLoadConfig()
                    self.speculative_config = None
                    self.observability_config = {}
                    self.additional_config = {}
                    self.device_config = {}

            vllm_config = DummyVllmConfig()

            self.runner = TPUModelRunner(vllm_config,
                                         devices=self.mock_devices)
            # Avoid attribute errors in forward when we stub model_fn
            self.runner.layer_name_to_kvcache_index = {}
            # Provide default for optional M-RoPE helper when not loading model
            self.runner.get_mrope_input_positions_fn = None
            # Prevent persistent batch manager from mutating state during tests
            self.runner.persistent_batch_manager.update_states = lambda *_, **__: None

    def test_get_supported_tasks_runner(self):
        """Test get_supported_tasks for generate runner type."""
        supported_tasks = self.runner.get_supported_tasks()
        assert supported_tasks == ("generate", )

    def test_get_input_ids_embeds(self):
        """Tests _get_input_ids_embeds for both multimodal and text-only models."""
        # 1. ===== Setup =====
        dummy_input_ids = jnp.array([1, 2, 3])
        dummy_mm_embeds = jnp.ones((10, 128))
        dummy_final_embeds = jnp.ones((3, 128))

        # Mock the embedding function
        self.mock_get_input_embed_fn = MagicMock()
        self.runner.get_input_embeddings_fn = self.mock_get_input_embed_fn
        self.mock_get_input_embed_fn.return_value = dummy_final_embeds
        self.runner.state = MagicMock()

        # 2. ===== Act & Assert (Multimodal) =====
        self.runner.is_multimodal_model = True

        input_ids_res, inputs_embeds_res = self.runner._get_input_ids_embeds(
            dummy_input_ids, dummy_mm_embeds)

        assert input_ids_res is None
        np.testing.assert_array_equal(np.asarray(inputs_embeds_res),
                                      np.asarray(dummy_final_embeds))
        self.mock_get_input_embed_fn.assert_called_once_with(
            self.runner.state, dummy_input_ids, dummy_mm_embeds)

        # 3. ===== Act & Assert (Text-only) =====
        self.mock_get_input_embed_fn.reset_mock()
        self.runner.is_multimodal_model = False

        input_ids_res, inputs_embeds_res = self.runner._get_input_ids_embeds(
            dummy_input_ids, dummy_mm_embeds)

        assert inputs_embeds_res is None
        np.testing.assert_array_equal(np.asarray(input_ids_res),
                                      np.asarray(dummy_input_ids))
        self.mock_get_input_embed_fn.assert_not_called()

    def _mk_scheduler_output(self, req_id: str, num_scheduled_tokens: int
                              ) -> SchedulerOutput:
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={req_id: num_scheduled_tokens},
            total_num_scheduled_tokens=num_scheduled_tokens,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            structured_output_request_ids=[],
            grammar_bitmask=None,
        )

    def _install_minimal_model_stubs(self, total_positions: int, vocab: int = 8):
        # model_fn returns dummy hidden states for all scheduled positions
        hidden_states = jnp.zeros((total_positions, 4), dtype=jnp.float32)
        # Ensure kv_caches exists for the stubbed return value
        if not hasattr(self.runner, 'kv_caches'):
            self.runner.kv_caches = []
        self.runner.model_fn = MagicMock(return_value=(self.runner.kv_caches,
                                                       hidden_states, None))
        # compute_logits_fn maps hidden states -> logits per position
        def _compute_logits_fn(state, hs, lora):
            n = hs.shape[0]
            return jnp.zeros((n, vocab), dtype=jnp.float32)

        self.runner.compute_logits_fn = MagicMock(side_effect=_compute_logits_fn)

    def _add_single_request(self, req_id: str, prompt_tokens: list[int],
                            prompt_logprobs: int) -> CachedRequestState:
        sampling_params = SamplingParams(temperature=0.0,
                                         top_p=1.0,
                                         top_k=0,
                                         prompt_logprobs=prompt_logprobs)
        req = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=prompt_tokens,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=([0], ),
            num_computed_tokens=0,
            lora_request=None,
            prompt_embeds=None,
            output_token_ids=[],
        )
        # Mirror scheduler-side cache of request
        self.runner.requests[req_id] = req
        # Add to input batch for execution
        self.runner.input_batch.add_request(req)
        return req

    @patch('tpu_inference.runner.tpu_jax_runner.sample', return_value=jnp.array([42], dtype=jnp.int32))
    def test_prompt_logprobs_partial_chunk_no_emit(self, _mock_sample):
        """Prefill chunk that doesn't finish the prompt should buffer only."""
        req_id = "r1"
        # Prompt of length 5 -> 4 positions will have prompt logprobs
        prompt = [10, 11, 12, 13, 14]
        self._add_single_request(req_id, prompt, prompt_logprobs=2)

        # Prepare minimal inputs: 1 req, schedule 3 tokens this step
        num_sched = 3
        scheduler_output = self._mk_scheduler_output(req_id, num_sched)
        # Stub _prepare_inputs to avoid heavy path; emulate indices and metadata
        logits_indices = jnp.array([num_sched - 1], dtype=jnp.int32)
        class _M: pass
        attn_meta = _M()
        attn_meta.query_start_loc_cpu = np.array([0], dtype=np.int32)
        # sampling_metadata.logprobs=False so no sampled logprobs collected
        class _SM: pass
        sampling_metadata = _M()
        sampling_metadata.logprobs = False

        # Install minimal model stubs returning per-position hidden states/logits
        self._install_minimal_model_stubs(total_positions=num_sched, vocab=8)

        with patch.object(self.runner, '_prepare_inputs', return_value=(
            jnp.array([0] * num_sched, dtype=jnp.int32),  # input_ids
            attn_meta,
            sampling_metadata,
            logits_indices,
            None,  # spec_decode_metadata
        )):
            _attn, out = self.runner._execute_model(scheduler_output)

        # Not final chunk yet -> do not emit prompt logprobs
        assert out.prompt_logprobs_dict[req_id] is None
        # Internal buffer should have accumulated 3 rows so far
        buf = self.runner._pending_prompt_logprobs.get(req_id)
        assert buf is not None
        assert buf.logprob_token_ids.shape[0] == 3
        # columns = prompt_logprobs + 1
        assert buf.logprob_token_ids.shape[1] == 3

    @patch('tpu_inference.runner.tpu_jax_runner.sample', return_value=jnp.array([7], dtype=jnp.int32))
    def test_prompt_logprobs_final_chunk_emit_and_concat(self, _mock_sample):
        """Final prefill chunk should emit concatenated prompt logprobs."""
        req_id = "r1"
        prompt = [10, 11, 12, 13, 14]
        self._add_single_request(req_id, prompt, prompt_logprobs=2)

        # Step 1: partial prefill of 3 tokens -> buffer 3 rows
        self._install_minimal_model_stubs(total_positions=3, vocab=8)
        logits_indices = jnp.array([2], dtype=jnp.int32)
        class _M: pass
        attn_meta = _M(); attn_meta.query_start_loc_cpu = np.array([0], dtype=np.int32)
        sampling_metadata = _M(); sampling_metadata.logprobs = False
        with patch.object(self.runner, '_prepare_inputs', return_value=(
            jnp.array([0, 0, 0], dtype=jnp.int32), attn_meta, sampling_metadata, logits_indices, None)):
            _attn, _out1 = self.runner._execute_model(self._mk_scheduler_output(req_id, 3))

        # Simulate scheduler advancing computed tokens for the request
        self.runner.input_batch.num_computed_tokens_cpu[0] = 3

        # Step 2: final chunk schedules 2 tokens, but only 1 remaining prompt position -> emit full (3+1) rows
        self._install_minimal_model_stubs(total_positions=2, vocab=8)
        logits_indices = jnp.array([1], dtype=jnp.int32)
        attn_meta2 = _M(); attn_meta2.query_start_loc_cpu = np.array([0], dtype=np.int32)
        with patch.object(self.runner, '_prepare_inputs', return_value=(
            jnp.array([0, 0], dtype=jnp.int32), attn_meta2, sampling_metadata, logits_indices, None)):
            _attn, out2 = self.runner._execute_model(self._mk_scheduler_output(req_id, 2))

        # Should emit concatenated prompt logprobs on this step
        lp = out2.prompt_logprobs_dict[req_id]
        assert lp is not None
        # Total rows equal to prompt_len - 1 = 4
        assert lp.logprob_token_ids.shape[0] == 4
        # columns = prompt_logprobs + 1
        assert lp.logprob_token_ids.shape[1] == 3
        # Buffer cleared
        assert req_id not in self.runner._pending_prompt_logprobs


class TestTPUJaxRunnerMultimodalModelLoadedForTextOnly:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock(coords=i) for i in range(4)]
        self.mock_mesh = MagicMock()
        self.mock_rng_key = MagicMock()

        # Setup the runner with the model_config.is_multimodal_model set to True but get_model returning None for get_multimodal_embeddings_fn and get_input_embeddings_fn.
        with patch('jax.devices', return_value=self.mock_devices), \
             patch('jax.make_mesh', return_value=self.mock_mesh), \
             patch('jax.random.key', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_jax_runner.nnx.Rngs', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_jax_runner.get_model', return_value=self._model_get_model()):

            class DummyModelConfig:
                def __init__(self):
                    self.seed = 0
                    self.dtype = 'bfloat16'
                    self.max_model_len = 128
                    self.is_multimodal_model = True
                    self.uses_mrope = False

                def get_sliding_window(self):
                    return 0

                def get_vocab_size(self):
                    return 8

            class DummyCacheConfig:
                def __init__(self):
                    self.block_size = 16
                    self.cache_dtype = 'auto'

            class DummySchedulerConfig:
                def __init__(self):
                    self.max_num_seqs = 16
                    self.max_num_batched_tokens = 16
                    self.async_scheduling = False

            class DummyParallelConfig:
                def __init__(self):
                    self.decode_context_parallel_size = 1

            class DummyLoraConfig:
                def __init__(self):
                    self.lora_extra_vocab_size = 0

            class DummyLoadConfig:
                def __init__(self):
                    self.tensor_parallel_size = 1

            class DummyVllmConfig:
                def __init__(self):
                    self.model_config = DummyModelConfig()
                    self.cache_config = DummyCacheConfig()
                    self.scheduler_config = DummySchedulerConfig()
                    self.parallel_config = DummyParallelConfig()
                    self.lora_config = DummyLoraConfig()
                    self.load_config = DummyLoadConfig()
                    self.speculative_config = None
                    self.observability_config = {}
                    self.additional_config = {}
                    self.device_config = {}

            vllm_config = DummyVllmConfig()

            self.runner = TPUModelRunner(vllm_config,
                                         devices=self.mock_devices)
            self.runner.load_model()

    def _model_get_model(self):
        mock_multimodal_fns = {
            "precompile_vision_encoder_fn": None,
            "get_multimodal_embeddings_fn": None,
            "get_input_embeddings_fn": None,
            "get_mrope_input_positions_fn": None
        }
        return (
            MagicMock(),  # TPUModelRunner.model_fn
            MagicMock(),  # TPUModelRunner.compute_logits_fn
            MagicMock(),  # TPUModelRunner.combine_hidden_states_fn
            mock_multimodal_fns,  # TPUModelRunner.multimodal_fns
            MagicMock(),  # TPUModelRunner.state (model params)
            None,  # TPUModelRunner.lora_manager
            None,  # TPUModelRunner.model
        )

    def test_is_multimodal_model(self):
        # Precondition: make sure the model_config claims the model supports MM.
        assert self.runner.model_config.is_multimodal_model

        # Precondition: load the model and returns get_multimodal_embeddings_fn as None.
        assert self.runner.get_multimodal_embeddings_fn is None

        assert not self.runner.is_multimodal_model

        self.runner.get_input_embeddings_fn = MagicMock()
        dummy_input_ids = jnp.array([1, 2, 3])
        dummy_mm_embeds = jnp.ones((10, 128))
        _ = self.runner._get_input_ids_embeds(dummy_input_ids, dummy_mm_embeds)
        self.runner.get_input_embeddings_fn.assert_not_called()
