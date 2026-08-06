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

"""The one place a value may exist.

Every shape constant, every flag value, every compiler frame and every
profile in this benchmark lives in this file and nowhere else. An arm
file containing a literal is a lint failure. A value here without a
provenance comment naming where it was read from is a review failure.
"""

# ---------------------------------------------------------------------------
# Shapes. Verified against each model's own config.json, not shape files.
# ---------------------------------------------------------------------------
SHAPES = {
    "qwen3_5_397b": {
        # Provenance: model config.json of Qwen3.5-397B-A17B-FP8
        # (hidden_size, moe_intermediate_size, num_experts,
        #  num_experts_per_tok).
        "hidden": 4096,
        "intermediate": 1024,
        "experts": 512,
        "top_k": 10,
        "weight_dtype": "float8_e4m3fn",
        "ep": 8,
        "capture": "routing_capture/qwen3_5_397b/group_sizes.jsonl.gz",
    },
    "gpt_oss_20b": {
        # openai/gpt-oss-20b, config.json at the hf_cache snapshot
        # 6cee5e81ee83917806bbde320786a8fb61efebee: hidden_size 2880,
        # intermediate_size 2880, num_local_experts 32,
        # num_experts_per_tok 4. Hidden is padded 2880 -> 2944 (23 lane
        # blocks) IDENTICALLY FOR EVERY ARM, because 2880 is not a
        # whole number of 128-wide lane blocks, and the intermediate is
        # padded the same one block (2880 -> 2944) because the upstream
        # production path's fused-activation grouped matmul refuses a
        # doubled expert width not divisible by 256; the model's expert
        # biases and its clamped swiglu are omitted identically (silu,
        # no bias) so every implementation runs the same math. Weights are
        # synthetic FP8, because the checkpoint's mxfp4 32-block form is
        # refused by name in the fused expert-parallel kernel.
        "hidden": 2944,
        "intermediate": 2944,
        "experts": 32,
        "top_k": 4,
        "weight_dtype": "float8_e4m3fn",
        "ep": 8,
        "capture": None,  # no recorded routing exists for this shape
    },
    "gpt_oss_20b_fp4": {
        # The same checkpoint config as gpt_oss_20b, at the model's native
        # four-bit width. Both dimensions pad to 3072 (six 512-wide scale
        # blocks, 24 lane blocks) because the four-bit form's scale blocks
        # are 512 wide and 2880 divides by neither 512 nor 128. Identical
        # padding for every arm; the SGLang arms have no four-bit form and
        # refuse at bind by name.
        "hidden": 3072,
        "intermediate": 3072,
        "experts": 32,
        "top_k": 4,
        "weight_dtype": "float4_e2m1fn",
        "ep": 8,
        "capture": None,  # no recorded routing exists for this shape
    },
    "gpt_oss_120b": {
        # unsloth/gpt-oss-120b-BF16, from the checkpoint's own
        # config.json: hidden_size 2880, intermediate_size 2880,
        # num_local_experts 128, num_experts_per_tok 4. Both widths are
        # padded 2880 -> 2944 IDENTICALLY FOR EVERY ARM, same rule and
        # reason as gpt_oss_20b above (128-wide lane blocks; the
        # production fused-activation grouped matmul refuses a doubled
        # expert width not divisible by 256). Expert biases and the
        # clamped swiglu are omitted identically (silu, no bias), and
        # weights are synthetic FP8.
        "hidden": 2944,
        "intermediate": 2944,
        "experts": 128,
        "top_k": 4,
        "weight_dtype": "float8_e4m3fn",
        "ep": 8,
        "capture": None,  # no recorded routing exists for this shape
    },
    "gpt_oss_120b_fp4": {
        # The 120B shape at the checkpoint's native four-bit expert
        # format, same padding rule as gpt_oss_20b_fp4: both widths
        # 2880 -> 3072 IDENTICALLY FOR EVERY ARM because the 512-wide
        # scale blocks require widths divisible by 512. Model facts as
        # gpt_oss_120b above, from the checkpoint's own config.json. The
        # kernel's own accounting fits this form inside its VMEM
        # budget (59.9 of 62.7 MiB, vmem_estimate_bytes at sixteen
        # local experts), where the FP8 form is over at any count.
        "hidden": 3072,
        "intermediate": 3072,
        "experts": 128,
        "top_k": 4,
        "weight_dtype": "float4_e2m1fn",
        "ep": 8,
        "capture": None,  # no recorded routing exists for this shape
    },
    "qwen3_30b": {
        # Provenance: model config.json of Qwen3-30B (hidden_size 2048,
        # moe_intermediate_size 768, num_experts 128,
        # num_experts_per_tok 8).
        "hidden": 2048,
        "intermediate": 768,
        "experts": 128,
        "top_k": 8,
        "weight_dtype": "float8_e4m3fn",
        "ep": 8,
        "capture": None,  # no recorded routing exists for this shape
    },
}

BATCHES = (64, 128, 256, 512, 1024, 2048, 4096, 8192)

# ---------------------------------------------------------------------------
# Operand layouts. What the weight master builds for a shape, where the form
# is decided by something other than a field of the shape itself.
# ---------------------------------------------------------------------------
# 512: the contraction rows one four-bit weight scale covers. The fused
# expert-parallel kernel's own constant, QB4 at
# tpu_inference/kernels/fused_moe/v2/host.py:32, which its layer entry
# applies to BOTH matmuls whenever the caller passes no rhs_qb of its own
# (kernels/fused_moe/v2/layer.py:144), and this kit passes none. The fused
# grouped matmul does not name a width; it accepts one by arithmetic, and
# 512 satisfies both of its conditions -- a block at least as wide as the
# MXU column and a whole number of lhs quantization blocks
# (kernels/megablox/gmm_fused.py:123-131). The serving weight loader takes
# its width from MOE_REQUANTIZE_BLOCK_SIZE (envs.py:36, read at
# layers/common/process_weights/moe_weights.py:738) and lays the resulting
# scales out as [experts, blocks, 1, channels] (moe_weights.py:331-341),
# which is the layout the master carries.
WEIGHT_SCALE_BLOCK = 512

# ---------------------------------------------------------------------------
# Compiler frames. A frame is the exact XLA option dict a cell compiles
# under, declared here, stamped on the cell, named on every table.
# ---------------------------------------------------------------------------
FRAMES = {
    # The two halves of the serving compile environment, and nothing else.
    #
    # THE JIT OPTIONS. All five entries are the serving deployment's own
    # jit options, verbatim. The offload minimum 67108864 is the "auto"
    # setting, which is the 64 MiB vmem capacity on this chip.
    #
    # EXCLUDED, and this is the whole point of naming the frame on every
    # table: the all-gather offload extras
    #   xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes = 8388608
    #   xla_lhs_prioritize_async_depth_over_stall = true
    # which one kernel was tuned against and which override this frame's own
    # all-gather offload minimum. They are that kernel's tuning rather than
    # the serving deployment's options, and a comparison that let one arm
    # bring them in would be measuring that arm's tuning.
    #
    # A FRAME HAS TWO HALVES, because the serving compile environment does.
    # The jit options are one and the runtime initialization flags are the
    # other, and they change device compile behaviour just as much. A frame
    # that carried only the options would leave the gate refusing the serving
    # runtime flags as if they were contamination, which inverts the whole
    # point of naming a frame.
    "serving": {
        "compiler_options": {
            "xla_tpu_all_gather_collective_matmul_mode":
                "post_spmd_conservative",
            "xla_tpu_reduce_scatter_collective_matmul_mode":
                "post_spmd_conservative",
            "xla_tpu_use_minor_sharding_for_major_trivial_input": "true",
            "xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes":
                "67108864",
            "xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes":
                "67108864",
        },
        # LIBTPU_INIT_ARGS, the runtime half.
        #
        # The last five are the serving deployment's own runtime flags,
        # verbatim and in order.
        #
        # The first is NAMED HERE ON PURPOSE. The serving tree's
        # tpu_inference/env_override.py prepends it to LIBTPU_INIT_ARGS at
        # import time, so a deployment picks it up without ever listing it.
        # A frame that depends on whether an import happened is not a frame,
        # so this kit declares the flag and does not care about import order.
        # That prepend does not deduplicate, so a tree that adds it again
        # produces the same token twice. The child-side check treats a
        # duplicate of a flag this frame already names as the no-op it is,
        # and refuses any token the frame does not name.
        "libtpu_init_args": [
            "--xla_tpu_use_dynamic_smem_negotiation=true",
            "--xla_tpu_use_minor_sharding_for_major_trivial_input=true",
            "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter"
            "=false",
            "--xla_tpu_ars_combiner_threshold_in_bytes=0",
            "--xla_tpu_enable_async_collective_merger=false",
            "--xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer"
            "=false",
        ],
    },
    # Neither half: whatever the compiler and the runtime do by default.
    "plain": {"compiler_options": {}, "libtpu_init_args": []},
}

# ---------------------------------------------------------------------------
# Profiles. A profile is the complete flag set an arm runs under. The
# config_origin stamped on the cell says whose choice the profile is.
# ---------------------------------------------------------------------------
PROFILES = {
    # Unmodified upstream tpu-inference at its own code defaults.
    # Provenance: the upstream envs.py defaults at UPSTREAM_PIN below,
    # ONEHOT_MOE_PERMUTE_THRESHOLD=0, VLLM_MOE_CHUNK_SIZE=0,
    # ENABLE_RS_KERNEL=False, MOE_ALL_GATHER_ACTIVATION_DTYPE="".
    "upstream-default": {
        "origin": "upstream-default",
        "flags": {
            "onehot_moe_permute_threshold": 0,
            "moe_chunk_size": 0,
            "scatter_results": False,
        },
    },
    # The configuration the serving deployment of the 397B model actually
    # runs. Each value below names the serving code that reads it, so the
    # value can be checked against the tree rather than taken on trust.
    "ours-served": {
        "origin": "ours-served",
        "flags": {
            # 32768, the serving value of ONEHOT_MOE_PERMUTE_THRESHOLD.
            # Declared at envs.py, consumed at layers/common/moe.py and
            # gated in fused_moe_gmm.py. One-hot matmul dispatch and combine
            # run while rows, which are tokens times top_k, stay at or below
            # it.
            "onehot_moe_permute_threshold": 32768,
            # 256, the serving value of VLLM_MOE_CHUNK_SIZE. Declared at
            # envs.py and wired from layers/vllm/interface/moe.py through
            # layers/common/moe.py to the fused_moe_gmm.py gate. Live at
            # prefill above 2048 tokens and structurally inert at decode.
            "moe_chunk_size": 256,
            # True as served. Set at layers/vllm/interface/moe.py by
            # is_attn_dp(mesh) under the enable_dp_attention serve argument.
            # The combine ends in a psum_scatter onto the eight owner
            # ranks.
            "scatter_results": True,
            # False. MOE_FP8_DISPATCH is not set by the serving
            # configuration and its envs.py default is False, so eight-bit
            # dispatch is off as served at every batch size.
            "fp8_dispatch": False,
        },
    },
    # An external kernel at the configuration its own selector returns.
    # There are no values here: the arm's verified selector mirror
    # produces them per (shape, batch), and the cell stamps
    # origin=owner-selector plus the resolved values in flags.
    "owner-selector": {"origin": "owner-selector", "flags": None},
    # An external kernel at its shipped defaults where it has no selector.
    "owner-default": {"origin": "owner-default", "flags": None},
    # A configuration this project chose for someone else's kernel. Renders
    # only under a "(Tuned By Us)" row a manifest explicitly names, never
    # under the owner's label. Flags come from the manifest that asks.
    "ours-tuned": {"origin": "ours-tuned", "flags": None},
}

# ---------------------------------------------------------------------------
# Pins. Commit plus per-file sha256 for every source not on this branch.
# The values live in pins.json beside this file; this maps arms to their
# pin entries. Our own arms bind THIS branch's tree and are content-hashed
# at run time rather than pinned here.
# ---------------------------------------------------------------------------
# vllm-project/tpu-inference, main.
UPSTREAM_PIN = "4d58cf1c5d819d809f88f25090cf45b0b6d0e422"
# The pins.json entry naming that same commit. A checkout that already holds
# the commit makes a worktree of it and never fetches; a clone that does not
# -- a shallow one, or a fork whose history does not reach upstream main --
# fetches this entry instead, so the production arm has a tree to bind from
# any clone. The entry pins no per-file hashes because the arm binds a whole
# tree rather than a named set of files, and git refuses a checkout of any
# commit but this one.
UPSTREAM_PIN_NAME = "upstream_pin"
PINS_FILE = "pins.json"

# ---------------------------------------------------------------------------
# Measurement constants.
# ---------------------------------------------------------------------------
TIERS = {
    "screen": {"iters": 300, "repeats": 1},
    "heavy": {"iters": 1000, "repeats": 3},
}
WARMUP = 30
REPLAY_STEPS_PER_CELL = 4
# Replay step selection excludes DEGENERATE records by rule, not by a call
# index. A record whose active-expert count equals top_k routed every token
# identically, which is a boot artifact and not a serving condition. The rule
# is general, and the chosen (capture, call) pairs are stamped on every cell
# so a reader sees which serving steps were used.
REPLAY_EXCLUDE_ACTIVE_EQ_TOPK = True
# A start-up sweep lights up to about 2.4 times the selection width, so a
# record whose active-expert count is under this many times top_k is
# eligible and anything wider is start-up traffic.
REPLAY_EXCLUDE_ACTIVE_FACTOR = 3
# env.lock is a pip freeze of the environment the cells were measured in.
# It is asserted before any cell is measured, not advisory.
ENV_LOCK_FILE = "env.lock"
DEVICE_REQUIRED = {"kind": "TPU v7", "chips": 8, "hosts": 1}
