#!/usr/bin/env python3
"""Benchmark DeepSeek MoE+MLA layer via TorchAX/vLLM path.

Tests the full TorchAX path for DeepSeek-V3:
1. Creates VllmConfig with small DeepSeek-V3 config
2. Loads model on CPU with dummy weights
3. Shards weights to TPU via shard_model_to_tpu
4. Patches MLAAttention.forward() for TPU MLA Pallas kernel
5. Creates MLA KV caches
6. Runs forward pass under torchax env
"""

import enum
import os
import sys
import time

sys.path.insert(0, '/mnt/pd/tpu-inference')

# ---- PauseState stub (version mismatch workaround) ----
import vllm.v1.core.sched.interface as _iface
if not hasattr(_iface, 'PauseState'):
    class _PauseState(enum.Enum):
        UNPAUSED = 0
        PAUSED = 1
    _iface.PauseState = _PauseState

# ---- Environment ----
os.environ.setdefault('MODEL_IMPL_TYPE', 'vllm')
os.environ.setdefault('TPU_BACKEND_TYPE', 'NEW_MODEL_DESIGN')
os.environ.setdefault('VLLM_USE_V1', '1')
os.environ.setdefault('VLLM_TPU_DISABLE_TOPK_MASK', '1')

import functools
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from torchax.interop import jax_view, torch_view

from vllm.config import (
    VllmConfig, ModelConfig, CacheConfig, ParallelConfig,
    DeviceConfig, LoadConfig, SchedulerConfig, set_current_vllm_config,
)
from vllm.forward_context import set_forward_context

# tpu-inference imports
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context)
from tpu_inference.layers.vllm.mla_attention import patch_mla_for_tpu
from tpu_inference.layers.vllm.process_weights.cleanup_sharding import (
    shard_model_to_tpu)
from tpu_inference.kernels.mla.v1.kernel import (
    get_kv_cache_shape as mla_get_kv_cache_shape)
from tpu_inference.layers.common.sharding import ShardingAxisName


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_LAYERS = 4                # 3 dense + 1 MoE (first_k_dense_replace=3)
NUM_EXPERTS = 256             # Real DeepSeek-V3 has 256

# ATTENTION_MODE: 'decode' or 'prefill'
#   decode:  MAX_NUM_TOKENS sequences, 1 new token each (batch decode)
#   prefill: 1 sequence, MAX_NUM_TOKENS new tokens (single prefill)
ATTENTION_MODE = os.environ.get('ATTENTION_MODE', 'decode')  # 'decode' or 'prefill'
MAX_NUM_TOKENS = int(os.environ.get('MAX_NUM_TOKENS', '64'))
MAX_NUM_SEQS = MAX_NUM_TOKENS if ATTENTION_MODE == 'decode' else MAX_NUM_TOKENS
MAX_MODEL_LEN = 2048          # Context window
PAGE_SIZE = 16                # Tokens per page
TP_SIZE = 8                   # Tensor parallel across 8 chips
DECODE_CONTEXT_LEN = int(os.environ.get('DECODE_CONTEXT_LEN', '64'))

# pages_per_seq: max pages any single sequence can use in block tables
# This determines the block_table layout: [MAX_NUM_SEQS * PAGES_PER_SEQ]
# The kernel derives pages_per_seq = page_indices.shape[0] // max_num_seqs
PAGES_PER_SEQ = (MAX_MODEL_LEN + PAGE_SIZE - 1) // PAGE_SIZE  # 128

# NUM_KV_PAGES: total physical cache pages (the actual KV cache buffer size)
# The cache is sharded across TP_SIZE devices on the page dimension.
# Each device gets NUM_KV_PAGES // TP_SIZE pages.
# Page indices must be < NUM_KV_PAGES // TP_SIZE.
# So we need: NUM_KV_PAGES >= total_pages_needed * TP_SIZE
_pages_needed = MAX_NUM_SEQS * ((max(DECODE_CONTEXT_LEN, MAX_MODEL_LEN) + PAGE_SIZE - 1) // PAGE_SIZE)
NUM_KV_PAGES = max(256, _pages_needed * TP_SIZE)  # multiply by TP to account for sharding

# DECODE_PATH: 'native' uses [N,N,N] (real decode kernel path),
#              'mixed' uses [0,0,N] (mixed path with dynamic q_len=1, better tested)
DECODE_PATH = os.environ.get('DECODE_PATH', 'mixed')
DTYPE = 'bfloat16'

# FP4 quantization for MoE weights
# Set USE_FP4=True to convert MoE weights from bf16 → float4_e2m1fn + subchannel scales.
# Follows the same approach as benchmark_deepseek_moe_layer.py and the GPT-OSS flow:
#   1. Load model with unquantized dummy weights (bf16)
#   2. After sharding, requantize MoE w13/w2 to fp4 + block scales
#   3. Patch apply_monolithic to pass scales through to GMM kernel
USE_FP4 = os.environ.get('USE_FP4', '0') == '1'
QUANT_BLOCK_SIZE = 256        # Subchannel quantization block size

# MoE-only benchmark settings
MOE_TOKEN_COUNTS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
MOE_WARMUP = 10
MOE_ITERS = 30
BENCHMARK_MODE = os.environ.get('BENCHMARK_MODE', 'moe')  # 'full' or 'moe'


def create_vllm_config():
    """Create a VllmConfig for a small DeepSeek-V3 model."""
    mc = ModelConfig(
        model='deepseek-ai/DeepSeek-V3',
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
    )
    # Set model size
    mc.hf_config.num_hidden_layers = NUM_LAYERS
    mc.hf_config.n_routed_experts = NUM_EXPERTS
    # Force unquantized (DeepSeek-V3 HF config has fp8 quant, but dummy
    # loader can't create random fp8 tensors)
    mc.hf_config.quantization_config = None
    mc.quantization = None
    # DeepSeek-V3 MLA params from config:
    # kv_lora_rank=512, q_lora_rank=1536, qk_nope_head_dim=128,
    # qk_rope_head_dim=64, v_head_dim=128, num_attention_heads=128

    lc = LoadConfig(load_format='dummy')
    cc = CacheConfig(
        block_size=PAGE_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype='auto',
    )
    pc = ParallelConfig(
        tensor_parallel_size=TP_SIZE,
        enable_expert_parallel=True,
    )
    dc = DeviceConfig(device='tpu')
    sc = SchedulerConfig(
        max_num_batched_tokens=MAX_NUM_TOKENS,
        max_num_seqs=MAX_NUM_SEQS,
        max_model_len=MAX_MODEL_LEN,
        is_encoder_decoder=False,
    )

    vc = VllmConfig(
        model_config=mc,
        load_config=lc,
        cache_config=cc,
        parallel_config=pc,
        device_config=dc,
        scheduler_config=sc,
    )
    print(f"VllmConfig created: use_mla={mc.use_mla}, "
          f"is_deepseek_mla={mc.is_deepseek_mla}, "
          f"load_format={lc.load_format}")
    return vc


def create_mesh():
    """Create a JAX mesh for 8 TPU chips."""
    devices = jax.devices()
    num_devices = len(devices)
    print(f"Found {num_devices} JAX devices: {devices[0].device_kind}")
    # 2D mesh: (data=1, model=8) matching ShardingAxisName2D
    mesh = Mesh(
        np.array(devices).reshape((1, num_devices)),
        axis_names=('data', 'model'),
    )
    return mesh


def create_model_on_cpu(vllm_config, mesh):
    """Create DeepSeek model on CPU with dummy weights."""
    import copy
    from vllm.model_executor.model_loader import get_model as vllm_get_model

    print("Creating model on CPU with dummy weights...")
    t0 = time.time()

    # Patch torch._sync for dummy weight loading (same as VllmModelWrapper)
    torch._sync = lambda x: None

    # Copy config and override device to CPU for loading
    vc = copy.deepcopy(vllm_config)
    vc.device_config = DeviceConfig(device='cpu')
    # Disable EP for CPU loading (single process)
    vc.parallel_config.enable_expert_parallel = False

    # Use torchax.default_env() so that weight processing ops (.T etc.)
    # work on torchax tensors, and jax.default_device(cpu) to keep
    # JAX operations on CPU during loading.
    jax_cpu_ctx = jax.default_device(jax.devices("cpu")[0])
    with set_current_vllm_config(vc), torchax.default_env(), jax_cpu_ctx:
        model = vllm_get_model(vllm_config=vc)

    # Copy static_forward_context and static_all_moe_layers back
    # (populated during model init for MoE layer registry)
    vllm_config.compilation_config.static_forward_context = (
        vc.compilation_config.static_forward_context)
    vllm_config.compilation_config.static_all_moe_layers = (
        vc.compilation_config.static_all_moe_layers)

    t1 = time.time()
    print(f"Model created on CPU in {t1-t0:.1f}s")
    print(f"Model type: {type(model).__name__}")
    print(f"MoE layers registered: {vllm_config.compilation_config.static_all_moe_layers}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.1f}M")

    return model


def find_mla_layer_names(model):
    """Find all MLAAttention layer names in the model."""
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, MLAAttention):
            layer_names.append(module.layer_name)
    return layer_names


def _convert_plain_tensor_attrs(model, mesh):
    """Convert plain CPU torch.Tensor attributes on modules and their
    non-Module sub-objects to torchax tensors.

    Some attributes like e_score_correction_bias are stored as plain
    torch.Tensor on non-nn.Module objects (e.g., FusedTopKBiasRouter),
    so they aren't handled by shard_model_to_tpu or functional_call.
    We replicate them to TPU.
    """
    from torchax.interop import jax_view, torch_view
    count = 0
    visited = set()

    def _convert_obj(obj, path=""):
        """Recursively convert plain tensors on any object."""
        nonlocal count
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if not hasattr(obj, '__dict__'):
            return

        for attr_name in list(vars(obj).keys()):
            if attr_name.startswith('_'):
                continue
            val = getattr(obj, attr_name, None)
            if val is None:
                continue
            if isinstance(val, torch.Tensor) and not isinstance(
                    val, torchax.tensor.Tensor):
                # Convert CPU tensor to replicated torchax tensor
                np_val = val.detach().float().numpy()
                jax_dtype = jnp.bfloat16 if val.dtype == torch.bfloat16 else None
                jax_arr = jax.device_put(
                    np_val, NamedSharding(mesh, P()))
                if jax_dtype is not None:
                    jax_arr = jax_arr.astype(jax_dtype)
                setattr(obj, attr_name, torch_view(jax_arr))
                full_path = f"{path}.{attr_name}" if path else attr_name
                print(f"  Converted {full_path} ({val.shape}, {val.dtype})")
                count += 1
            elif (not isinstance(val, torch.nn.Module)
                  and not isinstance(val, (str, int, float, bool, type,
                                          list, tuple, dict, set))
                  and hasattr(val, '__dict__')):
                # Recurse into non-Module objects that may hold tensors
                # (e.g., FusedTopKBiasRouter, DefaultMoERunner)
                full_path = f"{path}.{attr_name}" if path else attr_name
                _convert_obj(val, full_path)

    for name, module in model.named_modules():
        _convert_obj(module, name)

    print(f"Converted {count} plain CPU tensor attributes to torchax")


def convert_moe_weights_to_fp4(model, params_and_buffers, mesh):
    """Convert MoE weights from bf16 → float4_e2m1fn with subchannel scales.

    Follows the same pattern as benchmark_deepseek_moe_layer.py:
    - Replace w13_weight and w2_weight with fp4 versions
    - Add w13_weight_scale and w2_weight_scale entries
    - Monkey-patch apply_monolithic to pass scales to the GMM kernel

    The GMM v2 kernel auto-detects subchannel mode when
    rhs_scale.shape[1] > 1, so no kernel changes needed.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from tpu_inference.layers.common.process_weights.moe_weights import (
        FusedMoEWeights, quantize_moe_weights, process_moe_weights,
        shard_moe_weights)
    from tpu_inference.layers.common.moe import MoEBackend
    from tpu_inference.layers.vllm.moe import vllm_moe_apply

    qbs = QUANT_BLOCK_SIZE
    count = 0

    for name, module in model.named_modules():
        if not isinstance(module, FusedMoE):
            continue

        prefix = name
        w13_key = f"{prefix}.w13_weight"
        w2_key = f"{prefix}.w2_weight"

        if w13_key not in params_and_buffers:
            print(f"  Skipping {prefix}: w13_weight not in params")
            continue

        # Get current bf16 weights from params dict (already on TPU)
        w13_jax = jax_view(params_and_buffers[w13_key])  # [E, K, 2*I]
        w2_jax = jax_view(params_and_buffers[w2_key])    # [E, I, K]
        print(f"  {prefix}: w13={w13_jax.shape} {w13_jax.dtype}, "
              f"w2={w2_jax.shape} {w2_jax.dtype}")

        # Quantize bf16 → fp4 + subchannel scales
        # quantize_moe_weights expects: w13 [E, 2*I, K] but our weights
        # are already transposed to [E, K, 2*I] by process_moe_weights.
        # We need to provide pre-transpose layout [E, 2*I, K].
        w13_pre = jnp.swapaxes(w13_jax, 1, 2)  # [E, 2*I, K]
        w2_pre = jnp.swapaxes(w2_jax, 1, 2)    # [E, K, I]

        weights = FusedMoEWeights(
            w13_weight=w13_pre,
            w13_weight_scale=None,
            w13_bias=None,
            w2_weight=w2_pre,
            w2_weight_scale=None,
            w2_bias=None,
        )

        @jax.jit
        def quantize_and_process(w13, w2):
            ws = FusedMoEWeights(
                w13_weight=w13, w13_weight_scale=None, w13_bias=None,
                w2_weight=w2, w2_weight_scale=None, w2_bias=None,
            )
            ws = quantize_moe_weights(ws, jnp.float4_e2m1fn, qbs)
            ws = process_moe_weights(
                ws, moe_backend=MoEBackend.GMM_EP,
                w13_reorder_size=None, w13_interleave=False,
            )
            return ws

        ws = quantize_and_process(w13_pre, w2_pre)
        ws = shard_moe_weights(ws, MoEBackend.GMM_EP, mesh)

        # Store back into params dict
        params_and_buffers[w13_key] = torch_view(ws.w13_weight)
        params_and_buffers[w2_key] = torch_view(ws.w2_weight)
        params_and_buffers[f"{prefix}.w13_weight_scale"] = torch_view(
            ws.w13_weight_scale)
        params_and_buffers[f"{prefix}.w2_weight_scale"] = torch_view(
            ws.w2_weight_scale)

        print(f"    → w13: {ws.w13_weight.shape} {ws.w13_weight.dtype}, "
              f"scale: {ws.w13_weight_scale.shape}")
        print(f"    → w2:  {ws.w2_weight.shape} {ws.w2_weight.dtype}, "
              f"scale: {ws.w2_weight_scale.shape}")

        # Monkey-patch apply_monolithic to pass scales
        qm = module.quant_method
        original_apply = qm.apply_monolithic

        def make_fp4_apply(mod, pref):
            def fp4_apply_monolithic(layer, x, router_logits):
                w13_s_key = f"{pref}.w13_weight_scale"
                w2_s_key = f"{pref}.w2_weight_scale"
                # During functional_call, params are set on the module
                weights = FusedMoEWeights(
                    w13_weight=jax_view(layer.w13_weight),
                    w13_weight_scale=jax_view(layer.w13_weight_scale),
                    w13_bias=None,
                    w2_weight=jax_view(layer.w2_weight),
                    w2_weight_scale=jax_view(layer.w2_weight_scale),
                    w2_bias=None,
                )
                return vllm_moe_apply(
                    layer=layer, weights=weights,
                    quant_method_instance=qm, x=x,
                    router_logits=router_logits)
            return fp4_apply_monolithic

        qm.apply_monolithic = make_fp4_apply(module, prefix)

        # Register scale params on the module so functional_call picks them up
        from torch.nn.parameter import Parameter
        module.w13_weight_scale = Parameter(
            params_and_buffers[f"{prefix}.w13_weight_scale"],
            requires_grad=False)
        module.w2_weight_scale = Parameter(
            params_and_buffers[f"{prefix}.w2_weight_scale"],
            requires_grad=False)

        count += 1

    print(f"Converted {count} MoE layer(s) to fp4 (qbs={qbs})")


def create_mla_kv_caches(num_layers, mesh, hf_config):
    """Create MLA-shaped KV caches for all layers."""
    kv_lora_rank = hf_config.kv_lora_rank      # 512
    qk_rope_head_dim = hf_config.qk_rope_head_dim  # 64
    kv_dim = kv_lora_rank + qk_rope_head_dim    # 576

    cache_shape = mla_get_kv_cache_shape(
        NUM_KV_PAGES, PAGE_SIZE, kv_dim, jnp.bfloat16)
    print(f"MLA KV cache shape per layer: {cache_shape}")

    sharding = NamedSharding(mesh, P(ShardingAxisName.MLP_TENSOR))
    kv_caches = []
    for i in range(num_layers):
        cache = jax.device_put(
            jnp.zeros(cache_shape, dtype=jnp.bfloat16), sharding)
        kv_caches.append(cache)
    return kv_caches


def create_attention_metadata_decode(mesh, num_seqs, context_len=None):
    """Create attention metadata for DECODE: num_seqs sequences, 1 new token each.

    request_distribution semantics (from kernel.py):
      distribution = (i, j, k) where:
        sequences[0:i]   → decode      (static_q_len=1)
        sequences[i:j]   → chunked-prefill
        sequences[j:k]   → mixed (dynamic q_len)
        k = total active sequences
    For pure decode: i = j = k = num_seqs.
    For mixed path:  i = j = 0, k = num_seqs (processes q_len=1 dynamically).
    """
    if context_len is None:
        context_len = DECODE_CONTEXT_LEN
    assert num_seqs == MAX_NUM_TOKENS, \
        f"Decode mode: num_seqs ({num_seqs}) must equal MAX_NUM_TOKENS ({MAX_NUM_TOKENS})"

    # seq_lens = KV context length for each sequence (including the new token)
    seq_lens = np.full(MAX_NUM_SEQS, context_len, dtype=np.int32)

    # Block tables: each seq needs ceil(context_len/PAGE_SIZE) pages
    needed_pages = (context_len + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages_needed = num_seqs * needed_pages
    assert total_pages_needed <= NUM_KV_PAGES, \
        f"Need {total_pages_needed} pages but only have {NUM_KV_PAGES}. " \
        f"Increase NUM_KV_PAGES or reduce num_seqs/context_len."
    block_tables = np.zeros(MAX_NUM_SEQS * PAGES_PER_SEQ, dtype=np.int32)
    for s in range(num_seqs):
        for p in range(needed_pages):
            block_tables[s * PAGES_PER_SEQ + p] = s * needed_pages + p

    # cu_q_lens: each decode seq contributes 1 query token
    # [0, 1, 2, ..., num_seqs]
    query_start_loc = np.arange(MAX_NUM_SEQS + 1, dtype=np.int32)

    # Choose kernel dispatch path
    if DECODE_PATH == 'native':
        # True decode path: static_q_len=1 optimization
        request_distribution = np.array(
            [num_seqs, num_seqs, num_seqs], dtype=np.int32)
    else:
        # Mixed path: dynamic q_len (reads cu_q_lens which gives q_len=1)
        # This path is better tested (all kernel tests use it)
        request_distribution = np.array(
            [0, 0, num_seqs], dtype=np.int32)

    # Input positions: position of the new token for each seq
    padded_positions = np.full(MAX_NUM_TOKENS, context_len - 1, dtype=np.int32)

    sharding = NamedSharding(mesh, P())  # replicated
    attn_metadata = AttentionMetadata(
        input_positions=jax.device_put(
            jnp.array(padded_positions), sharding),
        block_tables=jax.device_put(
            jnp.array(block_tables), sharding),
        seq_lens=jax.device_put(
            jnp.array(seq_lens), sharding),
        query_start_loc=jax.device_put(
            jnp.array(query_start_loc), sharding),
        request_distribution=jax.device_put(
            jnp.array(request_distribution), sharding),
    )
    return attn_metadata, padded_positions


def create_attention_metadata_prefill(mesh, num_tokens, num_seqs=1):
    """Create attention metadata for PREFILL: num_seqs sequences totalling num_tokens.

    Distributes tokens evenly across sequences.
    Uses the "mixed" path (distribution = [0, 0, num_seqs]) following the
    real tpu_runner pattern.
    """
    assert num_tokens == MAX_NUM_TOKENS, \
        f"Prefill mode: num_tokens ({num_tokens}) must equal MAX_NUM_TOKENS ({MAX_NUM_TOKENS})"
    tokens_per_seq = num_tokens // num_seqs

    # seq_lens = KV length per sequence (= tokens being prefilled)
    seq_lens = np.zeros(MAX_NUM_SEQS, dtype=np.int32)
    for s in range(num_seqs):
        seq_lens[s] = tokens_per_seq

    # Block tables
    needed_pages = (tokens_per_seq + PAGE_SIZE - 1) // PAGE_SIZE
    block_tables = np.zeros(MAX_NUM_SEQS * PAGES_PER_SEQ, dtype=np.int32)
    for s in range(num_seqs):
        for p in range(needed_pages):
            block_tables[s * PAGES_PER_SEQ + p] = s * needed_pages + p

    # cu_q_lens: cumsum of query tokens per sequence
    query_start_loc = np.zeros(MAX_NUM_SEQS + 1, dtype=np.int32)
    for s in range(num_seqs):
        query_start_loc[s + 1] = query_start_loc[s] + tokens_per_seq
    # Pad remaining entries to total
    query_start_loc[num_seqs + 1:] = num_tokens

    # Mixed path (matches real tpu_runner): [0, 0, num_seqs]
    request_distribution = np.array(
        [0, 0, num_seqs], dtype=np.int32)

    # Input positions: [0, 1, ..., tokens_per_seq-1] repeated per seq
    padded_positions = np.zeros(MAX_NUM_TOKENS, dtype=np.int32)
    for s in range(num_seqs):
        start = s * tokens_per_seq
        padded_positions[start:start + tokens_per_seq] = np.arange(
            tokens_per_seq, dtype=np.int32)

    sharding = NamedSharding(mesh, P())  # replicated
    attn_metadata = AttentionMetadata(
        input_positions=jax.device_put(
            jnp.array(padded_positions), sharding),
        block_tables=jax.device_put(
            jnp.array(block_tables), sharding),
        seq_lens=jax.device_put(
            jnp.array(seq_lens), sharding),
        query_start_loc=jax.device_put(
            jnp.array(query_start_loc), sharding),
        request_distribution=jax.device_put(
            jnp.array(request_distribution), sharding),
    )
    return attn_metadata, padded_positions


def benchmark_moe_layer(model, params_and_buffers, vllm_config, mesh):
    """Benchmark ONLY the MoE FusedMoE layer over multiple token counts.

    Call chain:
      FusedMoE.forward_native() [torch]
        -> DefaultMoERunner.forward()
          -> VllmUnquantizedFusedMoEMethod.apply_monolithic() [is_monolithic=True]
            -> vllm_moe_apply() [torch->jax via jax_view()]
              -> moe_apply() [pure JAX]
                -> fused_moe_func(use_ep=True) [GMM EP path]
                  1. sigmoid scoring + topk routing
                  2. token permutation (argsort by expert)
                  3. GMM1: [M,H] @ w1[E,H,2I] via gmm_v2 Pallas kernel
                  4. SiLU activation + gate
                  5. GMM2: [M,I] @ w2[E,I,H] via gmm_v2 Pallas kernel
                  6. unpermute + weighted sum
                  7. psum across EP axis
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    # Find the MoE layer (layer 3, since first_k_dense_replace=3)
    moe_layer = None
    moe_prefix = None
    for name, m in model.named_modules():
        if isinstance(m, FusedMoE):
            moe_layer = m
            moe_prefix = name
            break
    assert moe_layer is not None, "No FusedMoE layer found in model!"

    # Extract just the MoE params
    moe_params = {}
    for k, v in params_and_buffers.items():
        if k.startswith(moe_prefix + "."):
            # Strip prefix so functional_call works
            short_key = k[len(moe_prefix) + 1:]
            moe_params[short_key] = v
    print(f"\nMoE layer: {moe_prefix}")
    print(f"MoE params: {len(moe_params)} tensors")

    # Print dtype info
    print("\n--- MoE Weight Dtypes ---")
    for k, v in sorted(moe_params.items()):
        jv = jax_view(v) if isinstance(v, torchax.tensor.Tensor) else v
        sh = jv.sharding if hasattr(jv, 'sharding') else 'n/a'
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} sharding={sh}")

    # Print quant method info
    qm = moe_layer.quant_method
    print(f"\n--- MoE Kernel Info ---")
    print(f"  quant_method: {type(qm).__name__}")
    print(f"  is_monolithic: {qm.is_monolithic}")
    print(f"  use_ep: {moe_layer.use_ep}")
    if hasattr(qm, 'moe_backend'):
        print(f"  moe_backend: {qm.moe_backend}")
    print(f"  scoring_func: {moe_layer.scoring_func}")
    print(f"  routed_scaling_factor: {moe_layer.routed_scaling_factor}")
    print(f"  top_k: {moe_layer.top_k}")
    print(f"  num_experts: {moe_layer.global_num_experts}")
    print(f"  local_num_experts: {moe_layer.local_num_experts}")

    # Dummy attention metadata (MoE doesn't use it, but context managers want it)
    dummy_attn = AttentionMetadata(
        input_positions=jax.device_put(jnp.zeros(64, dtype=jnp.int32),
                                       NamedSharding(mesh, P())),
        block_tables=jax.device_put(jnp.zeros(64, dtype=jnp.int32),
                                     NamedSharding(mesh, P())),
        seq_lens=jax.device_put(jnp.zeros(4, dtype=jnp.int32),
                                 NamedSharding(mesh, P())),
        query_start_loc=jax.device_put(jnp.zeros(5, dtype=jnp.int32),
                                        NamedSharding(mesh, P())),
        request_distribution=jax.device_put(jnp.array([4, 0, 0], dtype=jnp.int32),
                                             NamedSharding(mesh, P())),
    )

    moe_params_jax = jax_view(moe_params)
    tok_s = NamedSharding(mesh, P(None, None))
    HIDDEN = vllm_config.model_config.hf_config.hidden_size  # 7168
    N_EXPERTS = moe_layer.global_num_experts

    def make_moe_step(ntok):
        @jax.jit
        def moe_step(params, tokens, gating):
            with torchax.default_env(), set_vllm_model_wrapper_context(
                    kv_caches=[], mesh=mesh, layer_name_to_kvcache_index={}
            ), set_forward_context(attn_metadata=dummy_attn,
                                   vllm_config=vllm_config):
                out = torch.func.functional_call(
                    moe_layer,
                    torch_view(params),
                    args=(torch_view(tokens),),
                    kwargs={"router_logits": torch_view(gating)},
                )
                return jax_view(out)
        return moe_step

    # Benchmark header
    print(f"\n{'='*70}")
    print(f"  TorchAX FusedMoE Benchmark (EP={TP_SIZE}, {N_EXPERTS} experts, topk={moe_layer.top_k})")
    print(f"  Warmup={MOE_WARMUP}, Iters={MOE_ITERS}")
    print(f"{'='*70}")
    print(f"{'N':>6} {'M':>7}  {'median_ms':>10}  {'min_ms':>8}  {'max_ms':>8}")
    print("-" * 50)

    key = jax.random.PRNGKey(0)
    for ntok in MOE_TOKEN_COUNTS:
        k1, k2, key = jax.random.split(key, 3)
        tokens = jax.device_put(
            jax.random.normal(k1, (ntok, HIDDEN), dtype=jnp.bfloat16) / 10, tok_s)
        gating = jax.device_put(
            jax.random.normal(k2, (ntok, N_EXPERTS), dtype=jnp.bfloat16), tok_s)

        step = make_moe_step(ntok)

        # Warmup (includes JIT)
        for _ in range(MOE_WARMUP):
            out = step(moe_params_jax, tokens, gating)
            out.block_until_ready()

        # Timed
        times = []
        for _ in range(MOE_ITERS):
            t0 = time.perf_counter()
            out = step(moe_params_jax, tokens, gating)
            out.block_until_ready()
            times.append((time.perf_counter() - t0) * 1000)

        med = np.median(times)
        mn = np.min(times)
        mx = np.max(times)
        print(f"{ntok:>6} {ntok*moe_layer.top_k:>7}  {med:>10.2f}  {mn:>8.2f}  {mx:>8.2f}")

    print("\nDONE")


def main():
    print("=" * 60)
    print("DeepSeek-V3 TorchAX Layer Benchmark")
    print(f"  mode={BENCHMARK_MODE}, attn={ATTENTION_MODE}, "
          f"tokens={MAX_NUM_TOKENS}, layers={NUM_LAYERS}, "
          f"fp4={USE_FP4}")
    if ATTENTION_MODE == 'decode':
        print(f"  decode: {MAX_NUM_TOKENS} seqs, context_len={DECODE_CONTEXT_LEN}, "
              f"path={DECODE_PATH}, pages_per_seq={PAGES_PER_SEQ}, num_kv_pages={NUM_KV_PAGES}")
    else:
        print(f"  prefill: 1 seq, {MAX_NUM_TOKENS} tokens, "
              f"pages_per_seq={PAGES_PER_SEQ}, num_kv_pages={NUM_KV_PAGES}")
    print("=" * 60)

    # Step 1: Create config
    vllm_config = create_vllm_config()

    # Step 2: Create mesh (BEFORE Gloo init which can interfere)
    mesh = create_mesh()

    # Step 2b: Set up TPU quantization config — this ensures FusedMoE layers
    # get TPU-specific quant methods (monolithic MoE with GMM kernels)
    from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
    vllm_config.quant_config = get_tpu_quantization_config(vllm_config, mesh)
    print(f"TPU quant config: {vllm_config.quant_config}")

    # Step 2c: Initialize distributed env (single process, for vLLM parallel state)
    import tempfile
    from vllm.distributed.parallel_state import (
        init_distributed_environment, ensure_model_parallel_initialized)
    temp_file = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method=f"file://{temp_file}",
        backend="gloo",
    )
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    # Step 3: Create model on CPU with dummy weights
    with set_current_vllm_config(vllm_config):
        model = create_model_on_cpu(vllm_config, mesh)

    # Step 4: Find MLA layers and build cache index
    mla_layer_names = find_mla_layer_names(model)
    print(f"Found {len(mla_layer_names)} MLA layers: {mla_layer_names}")

    layer_name_to_kvcache_index = {}
    for i, name in enumerate(mla_layer_names):
        layer_name_to_kvcache_index[name] = i

    # Step 5: Patch MLA attention (only needed for full model mode)
    if BENCHMARK_MODE == 'full':
        print("Patching MLA attention for TPU...")
        with torchax.default_env():
            patch_mla_for_tpu(model)

    # Step 6: Shard model to TPU
    print("Sharding model to TPU...")
    t0 = time.time()
    with torchax.default_env():
        params_and_buffers = shard_model_to_tpu(model, mesh)
    t1 = time.time()
    print(f"Model sharded to TPU in {t1-t0:.1f}s")

    # Step 6b: Convert remaining plain CPU tensor attributes to torchax
    # (e.g., e_score_correction_bias on FusedTopKBiasRouter)
    _convert_plain_tensor_attrs(model, mesh)

    # Step 6c: Optionally convert MoE weights to fp4
    if USE_FP4:
        print(f"\nConverting MoE weights to fp4 (qbs={QUANT_BLOCK_SIZE})...")
        with torchax.default_env():
            convert_moe_weights_to_fp4(model, params_and_buffers, mesh)

    if BENCHMARK_MODE == 'full':
        # Step 7: Create MLA KV caches
        hf_config = vllm_config.model_config.hf_config
        kv_caches = create_mla_kv_caches(len(mla_layer_names), mesh, hf_config)
        print(f"Created {len(kv_caches)} MLA KV caches")

        # ---------- FULL MODEL BENCHMARK ----------
        # Step 8: Create attention metadata
        if ATTENTION_MODE == 'decode':
            attn_metadata, padded_positions = create_attention_metadata_decode(
                mesh, num_seqs=MAX_NUM_TOKENS)
            print(f"Attention metadata created (decode: {MAX_NUM_TOKENS} seqs, "
                  f"context={DECODE_CONTEXT_LEN}, path={DECODE_PATH})")
        else:
            attn_metadata, padded_positions = create_attention_metadata_prefill(
                mesh, num_tokens=MAX_NUM_TOKENS, num_seqs=1)
            print(f"Attention metadata created (prefill: 1 seq, {MAX_NUM_TOKENS} tokens)")

        # Step 9: Create dummy input
        input_ids = jnp.zeros(MAX_NUM_TOKENS, dtype=jnp.int32)
        input_positions = jnp.array(padded_positions, dtype=jnp.int32)

        # Step 10: Run forward pass
        print("\nRunning forward pass...")
        layer_name_to_kvcache_index_tuple = tuple(
            layer_name_to_kvcache_index.items())

        @functools.partial(
            jax.jit,
            donate_argnames=("kv_caches",),
            static_argnames=("layer_name_to_kvcache_index",),
        )
        def step_fn(params_and_buffers, kv_caches, input_ids,
                    attn_metadata, input_positions,
                    layer_name_to_kvcache_index):
            layer_name_to_kvcache_index = dict(layer_name_to_kvcache_index)
            with torchax.default_env(), set_vllm_model_wrapper_context(
                    kv_caches=kv_caches,
                    mesh=mesh,
                    layer_name_to_kvcache_index=layer_name_to_kvcache_index
            ), set_forward_context(attn_metadata=attn_metadata,
                                   vllm_config=vllm_config):
                output = torch.func.functional_call(
                    model,
                    torch_view(params_and_buffers),
                    kwargs={
                        "input_ids": torch_view(input_ids),
                        "positions": torch_view(input_positions),
                        "intermediate_tensors": None,
                        "inputs_embeds": None,
                    },
                    tie_weights=False,
                )
                from tpu_inference.models.vllm.vllm_model_wrapper_context import \
                    get_vllm_model_wrapper_context
                ctx = get_vllm_model_wrapper_context()
                new_kv_caches = ctx.kv_caches
            return new_kv_caches, jax_view(output)

        # JIT compile + first run
        print("JIT compiling...")
        t0 = time.time()
        params_jax = jax_view(params_and_buffers)
        try:
            new_kv_caches, output = step_fn(
                params_jax, kv_caches, input_ids,
                attn_metadata, input_positions,
                layer_name_to_kvcache_index_tuple)
            t1 = time.time()
            print(f"JIT compile + first run: {t1-t0:.2f}s")
            jax.block_until_ready((new_kv_caches, output))
            print(f"Output shape: {output.shape}")
            print(f"Output dtype: {output.dtype}")
            import sys
            sys.stdout.flush(); sys.stderr.flush()

            # Warmup + benchmark
            kv_caches = new_kv_caches
            import sys
            for i in range(3):
                print(f"Warmup {i}...", end=" ", flush=True)
                sys.stdout.flush(); sys.stderr.flush()
                kv_caches, output = step_fn(
                    params_jax, kv_caches, input_ids,
                    attn_metadata, input_positions,
                    layer_name_to_kvcache_index_tuple)
                jax.block_until_ready(output)
                print("done", flush=True)

            N_ITERS = 10
            print(f"Benchmarking {N_ITERS} iterations...", flush=True)
            sys.stdout.flush(); sys.stderr.flush()
            t0 = time.time()
            for i in range(N_ITERS):
                kv_caches, output = step_fn(
                    params_jax, kv_caches, input_ids,
                    attn_metadata, input_positions,
                    layer_name_to_kvcache_index_tuple)
            jax.block_until_ready(output)
            t1 = time.time()
            avg_ms = (t1 - t0) / N_ITERS * 1000
            print(f"\nBenchmark: {avg_ms:.2f} ms/step ({N_ITERS} iterations)")
            print("SUCCESS!")

        except Exception as e:
            t1 = time.time()
            print(f"\nFailed after {t1-t0:.2f}s")
            import traceback
            traceback.print_exc()
            raise

    else:
        # ---------- MOE-ONLY BENCHMARK ----------
        benchmark_moe_layer(model, params_and_buffers, vllm_config, mesh)


if __name__ == '__main__':
    main()
