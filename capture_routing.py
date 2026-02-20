#!/usr/bin/env python3
"""
Capture routing decisions from all MoE layers during real DeepSeek-R1 inference.

Usage:
    cd /mnt/pd/tpu-inference
    CAPTURE_ROUTING=1 /mnt/pd/vllm-venv/bin/python3 capture_routing.py

Produces: /mnt/pd/routing_captures.npz
"""

import os
import sys
import time
import enum

# ---- Environment setup (MUST be before any JAX/vLLM imports) ----
os.environ['CAPTURE_ROUTING'] = '1'
os.environ['CAPTURE_ROUTING_SCORES'] = '1'  # also capture full 256-expert scores
os.environ['NUM_MOE_LAYERS'] = '59'         # DeepSeek-R1: layers 3-61 are MoE
os.environ['SKIP_JAX_PRECOMPILE'] = '1'     # faster startup

sys.path.insert(0, '/mnt/pd/tpu-inference')

# ---- PauseState stub (version mismatch workaround) ----
import vllm.v1.core.sched.interface as _iface
if not hasattr(_iface, 'PauseState'):
    class _PauseState(enum.Enum):
        UNPAUSED = 0
        PAUSED = 1
    _iface.PauseState = _PauseState

import numpy as np
from dataclasses import asdict

from vllm import LLM, EngineArgs, SamplingParams


# ---- Diverse prompts for realistic routing patterns ----
PROMPTS = [
    # Reasoning / math
    "Solve step by step: If a train travels at 60 mph for 2.5 hours, how far does it go?",
    "What is the derivative of f(x) = x^3 * sin(x)?",
    # Code
    "Write a Python function to find the longest common subsequence of two strings.",
    "Explain the difference between a stack and a queue with examples.",
    # Science
    "Explain how CRISPR-Cas9 gene editing works in simple terms.",
    "What causes the Northern Lights (Aurora Borealis)?",
    # History / culture
    "Describe the major causes of World War I.",
    "What were the key achievements of the Tang Dynasty in China?",
    # Creative writing
    "Write a short poem about the ocean at midnight.",
    "Tell me a story about a robot learning to paint.",
    # General knowledge
    "What is the capital of Australia?",
    "How does a nuclear reactor generate electricity?",
    "What is the difference between machine learning and deep learning?",
    "Explain the concept of supply and demand in economics.",
    # Long-form
    "Describe the process of photosynthesis in detail, including the light and dark reactions.",
    "What are the main differences between TCP and UDP protocols?",
]


def main():
    model_path = "/mnt/pd/checkpoints/deepseek-r1-fp4-mlp-256"

    print(f"Loading model from {model_path}...")
    print(f"Routing capture: ENABLED (full scores + top-k decisions)")
    t0 = time.time()

    engine_args = EngineArgs(
        model=model_path,
        max_model_len=512,
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        max_num_batched_tokens=512,
        max_num_seqs=16,
        enable_prefix_caching=False,
    )

    llm = LLM(**asdict(engine_args))
    print(f"Model loaded in {time.time() - t0:.1f}s")

    sampling_params = SamplingParams(
        temperature=0.0,    # greedy for reproducibility
        max_tokens=64,      # short output to keep captures manageable
        ignore_eos=True,    # ensure we always generate max_tokens
    )

    print(f"\nRunning inference on {len(PROMPTS)} prompts...")
    t1 = time.time()
    outputs = llm.generate(PROMPTS, sampling_params)
    t_gen = time.time() - t1
    print(f"Inference completed in {t_gen:.1f}s")

    # Print a few outputs as sanity check
    print("\n" + "=" * 60)
    for i, output in enumerate(outputs[:3]):
        text = output.outputs[0].text[:200]
        print(f"Prompt {i}: {output.prompt[:60]}...")
        print(f"  Output: {text}...")
        print()

    # ---- Retrieve and save routing captures ----
    from tpu_inference.layers.common.fused_moe_gmm import (
        get_routing_captures, _routing_call_counter, _NUM_MOE_LAYERS)

    captures = get_routing_captures()
    total_calls = _routing_call_counter[0]
    num_steps = total_calls // _NUM_MOE_LAYERS if _NUM_MOE_LAYERS > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"Routing capture results:")
    print(f"  Total callback invocations: {total_calls}")
    print(f"  MoE layers per step: {_NUM_MOE_LAYERS}")
    print(f"  Inference steps: {num_steps}")
    print(f"  Captured entries: {len(captures)}")

    if not captures:
        print("WARNING: No routing data captured!")
        return

    # Show shape info from first capture
    c0 = captures[0]
    print(f"  First entry shapes:")
    print(f"    topk_indices: {c0['topk_indices'].shape}")
    print(f"    topk_weights: {c0['topk_weights'].shape}")
    if 'full_scores' in c0:
        print(f"    full_scores:  {c0['full_scores'].shape}")

    # ---- Save to compressed numpy file ----
    # Organize by step for easier analysis
    save_data = {}

    # Collect all unique steps
    steps = sorted(set(c['step'] for c in captures))
    layers = sorted(set(c['layer'] for c in captures))
    print(f"\n  Steps captured: {len(steps)} (step ids: {steps[0]}..{steps[-1]})")
    print(f"  Layers captured: {len(layers)} (layer ids: {layers[0]}..{layers[-1]})")

    # Save as flat arrays with metadata
    all_indices = []
    all_weights = []
    all_scores = []
    all_steps = []
    all_layers = []
    all_ntokens = []

    for c in captures:
        n_tokens = c['topk_indices'].shape[0]
        all_indices.append(c['topk_indices'])
        all_weights.append(c['topk_weights'])
        all_steps.append(np.full(n_tokens, c['step'], dtype=np.int32))
        all_layers.append(np.full(n_tokens, c['layer'], dtype=np.int32))
        all_ntokens.append(n_tokens)
        if 'full_scores' in c:
            all_scores.append(c['full_scores'])

    out_path = "/mnt/pd/routing_captures.npz"
    save_dict = {
        'topk_indices': np.concatenate(all_indices, axis=0),   # [total_tokens, K]
        'topk_weights': np.concatenate(all_weights, axis=0),   # [total_tokens, K]
        'step_ids': np.concatenate(all_steps, axis=0),         # [total_tokens]
        'layer_ids': np.concatenate(all_layers, axis=0),       # [total_tokens]
        'ntokens_per_entry': np.array(all_ntokens, dtype=np.int32),
        'entry_steps': np.array([c['step'] for c in captures], dtype=np.int32),
        'entry_layers': np.array([c['layer'] for c in captures], dtype=np.int32),
    }
    if all_scores:
        save_dict['full_scores'] = np.concatenate(all_scores, axis=0)

    np.savez_compressed(out_path, **save_dict)
    file_size = os.path.getsize(out_path)
    print(f"\n  Saved to: {out_path}")
    print(f"  File size: {file_size / 1024 / 1024:.1f} MB")
    print(f"  Total tokens × layers: {save_dict['topk_indices'].shape[0]}")

    # ---- Quick summary stats ----
    print(f"\n{'=' * 60}")
    print("Quick routing statistics (across all steps/layers):")
    indices = save_dict['topk_indices']
    print(f"  Expert index range: [{indices.min()}, {indices.max()}]")
    # Count how often each expert is selected
    expert_counts = np.bincount(indices.flatten(), minlength=256)
    print(f"  Most popular expert: {expert_counts.argmax()} "
          f"(selected {expert_counts.max()} times)")
    print(f"  Least popular expert: {expert_counts.argmin()} "
          f"(selected {expert_counts.min()} times)")
    imbalance = expert_counts.max() / max(expert_counts.mean(), 1)
    print(f"  Global imbalance ratio: {imbalance:.2f}x")


if __name__ == "__main__":
    main()
