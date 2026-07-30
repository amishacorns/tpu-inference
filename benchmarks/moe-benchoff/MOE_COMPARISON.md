# Mixture-of-Experts Kernel Comparison on TPU v7x

## Summary

This document compares six implementations of the expert-parallel mixture-of-experts (MoE) layer on TPU. The six are the production path, two kernels from this optimization work, and three external kernels, measured at the layer shapes of three models under recorded and random routing. No single implementation is fastest at every batch size, and transfer granularity decides the large ones. Every expert-parallel kernel returns each token's results to the core that owns it, and a transfer's cost is mostly fixed, so designs that pay it once per routed token fall behind as batches grow, while the design that plans its transport once and returns contiguous runs of tokens keeps its transfer count low and takes the large batches.

## Recommendation

**Serve with Fused GMM below the crossover batch size and PR 3288 at and above it.** The crossover is model dependent, and the Recommended row in each table is this switch at that model's shape. Both kernels have room to improve, and the crossover moves with them, so the switch point should be re-measured as either side is optimized.

## The Implementations

| Implementation | Description |
|---|---|
| Production | The path tpu-inference, the vLLM project's TPU serving stack, serves today. Two grouped matmul (GMM) calls per layer, with dispatch and combine done by XLA collectives. |
| [Fused GMM](https://github.com/amishacorns/tpu-inference/pull/1) | Production with the two GMM calls fused into one kernel, part of this optimization work and not yet in upstream tpu-inference. Dispatch and combine stay Production's XLA collectives. |
| [PR 3288](https://github.com/vllm-project/tpu-inference/pull/3288) | One fused expert pass keeping the intermediate in VMEM. Combine transport runs inside the kernel at one transfer per contiguous run of tokens. Runs in serving when its setting (`USE_MOE_FUSED_EP_KERNEL`) is on, engaging at and above a configured token count (`MOE_FUSED_EP_KERNEL_MIN_TOKENS`). Below that count the layer serves the existing path. It holds whole-expert weight buffers, so one wide eight-bit shape exceeds its memory budget and is refused. Tiling those buffers is future work. |
| [PR 3040](https://github.com/vllm-project/tpu-inference/pull/3040) | One Pallas call runs gather, both GMMs and the combine. The intermediate stays in VMEM and the combine moves one transfer per routed token. Tokens must arrive already replicated, so the kernel performs no dispatch of its own. |
| [SGLang-JAX V2](https://github.com/sgl-project/sglang-jax/tree/0381ccbf29602e42a7cf1a1e82c8d1ae6aa619f7/python/sgl_jax/srt/kernels/fused_moe/v2) | The second-generation expert-parallel kernel of SGLang-JAX, the SGLang serving project's JAX stack for TPU. All-to-all dispatch in the kernel at one transfer per routed token, each core computing its local experts one at a time and returning its results in one transfer per expert and destination. It computes only the experts that received tokens. |
| [SGLang-JAX V1](https://github.com/sgl-project/sglang-jax/tree/0381ccbf29602e42a7cf1a1e82c8d1ae6aa619f7/python/sgl_jax/srt/kernels/fused_moe/v1) | Their first-generation expert-parallel kernel. Same all-to-all design as V2, with a tuned-configuration table covering more shapes. It visits every local expert whether or not it received tokens. |

Every expert-parallel MoE implementation pays two costs, a fixed cost to plan and launch its transport and a per-token cost to move each token's results between cores. Production and Fused GMM pay almost no fixed cost, so they win small batches at most shapes. PR 3288 pays a fixed transport plan but moves results in contiguous runs, so its per-token cost is the lowest and it wins large batches. PR 3040 and the SGLang kernels pay per-token costs instead. PR 3040 moves each routed token's result separately, and the SGLang kernels move each routed token separately on dispatch. Where per-token transport has not yet grown, PR 3040 is fastest at several batches below the crossover, and its per-token combine takes over as batches grow.

## Qwen3.5 397B

512 experts, top-10 selection, hidden 4096, MoE intermediate 1024, FP8 weights, from the model's configuration file.

### Recorded Routing

Whole-layer program device time in microseconds (device time for one MoE layer call, host time excluded), routing replayed from a serving capture, four serving steps per measurement with the capture's start-up records excluded, averaged over the steps. SGLang-JAX's own stack routes this model to V1 rather than V2, and V1's tuned-configuration table has no entry at this model's shape, so V1 runs at its fixed default configuration. Both their kernels are measured at their own calling convention. The Production and Fused GMM rows rise at 2048 and fall again at 4096 because their dispatch changes form at a threshold on routed tokens (tokens times the selection width) crossed between the two batch sizes. The expert-parallel implementations do not use that dispatch path, so their rows stay smooth through the same range. The threshold could be tuned better for this model at these batch sizes. The Comparison row states how much time Production spends over Recommended at each batch.

| Implementation | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|
| Production | 231.7 | 251.8 | 350.6 | 485.3 | 752.6 | 1635.2 | 1461.3 | 2086.4 |
| Fused GMM | **202.1** | **220.5** | **314.9** | 445.9 | 728.4 | 1649.5 | 1521.2 | 2194.5 |
| PR 3288 | 243.9 | 263.0 | 378.2 | 493.3 | 549.6 | **635.1** | **897.9** | **1401.7** |
| PR 3040 | 229.0 | 250.8 | 353.0 | **435.5** | **547.4** | 818.2 | 1604.5 | 2600.7 |
| SGLang-JAX V2 | 231.4 | 292.3 | 419.4 | 838.0 | 1617.2 | 2887.6 | 5715.9 | SMEM OOM |
| SGLang-JAX V1 | 735.7 | 858.1 | 1331.4 | 2851.6 | 5379.4 | 9231.4 | 16126.5 | 31542.0 |
| Recommended | 202.1 | 220.5 | 314.9 | 445.9 | 549.6 | 635.1 | 897.9 | 1401.7 |

| Comparison | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|
| Recommended vs Production | +14.6% | +14.2% | +11.4% | +8.8% | +36.9% | +157.5% | +62.8% | +48.9% |

### Model Quality

With FP8 weights, expert outputs travel between cores in eight bits, so PR 3288's output is not bit-identical to Production's. The closest end-to-end check, from the serving evaluation harness rather than this benchmark, ran an earlier revision of this expert-parallel kernel against the same serving without it.

| MMLU-Pro, Qwen3.5 397B FP8 | Score |
|---|---|
| Kernel On | 0.8300 |
| Kernel Off | 0.8337 |

The switched-off score is the mean of repeated boots, and its boot-to-boot spread covers the switched-on score.

## Qwen3 30B

128 experts, top-8 selection, hidden 2048, MoE intermediate 768, FP8 weights, from the model's configuration file. No serving configuration exists for this model, so Production and Fused GMM run upstream at its own defaults, and their dispatch takes the ragged dispatch path at every batch. SGLang-JAX V2 refuses the shape outright (at the block configuration its layer falls back to for an untuned shape, the intermediate width must divide by 512, and this model's is 768) while SGLang-JAX V1 runs it, because V1's tuned-configuration table carries an entry for this shape where V2's does not.

### Random Routing

Whole-layer program device time in microseconds, one seeded routing draw per batch size, identical across implementations.

| Implementation | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|
| Production | 94.0 | 113.5 | 149.8 | 234.6 | 229.7 | 337.7 | 527.4 | 1019.3 |
| Fused GMM | **79.5** | **99.5** | 137.2 | 226.6 | 223.7 | 338.9 | 556.1 | 1050.5 |
| PR 3288 | 135.5 | 151.1 | 160.9 | 165.0 | **180.2** | **236.8** | **338.5** | **654.2** |
| PR 3040 | 106.7 | 111.3 | **119.6** | **139.8** | 212.7 | 340.3 | 579.9 | 1173.2 |
| SGLang-JAX V2 | Width Limit | Width Limit | Width Limit | Width Limit | Width Limit | Width Limit | Width Limit | Width Limit |
| SGLang-JAX V1 | 176.5 | 186.2 | 212.8 | 267.5 | 383.1 | 613.6 | 1063.6 | 2130.8 |
| Recommended | 79.5 | 99.5 | 137.2 | 165.0 | 180.2 | 236.8 | 338.5 | 654.2 |

| Comparison | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|
| Recommended vs Production | +18.3% | +14.1% | +9.2% | +42.2% | +27.5% | +42.6% | +55.8% | +55.8% |

## GPT-OSS 20B FP4

32 experts, top-4 selection, hidden 2880, MoE intermediate 2880, from the model's configuration file. Weights are FP4 (E2M1) with 512-wide scale blocks. Both dimensions are padded to 3072, identically for every implementation. The model's expert biases and its clamped activation are omitted identically (plain silu, no bias) so every implementation runs the same math. No serving configuration exists for this model, so Production and Fused GMM run upstream at its own defaults, and their dispatch takes the ragged dispatch path at every batch.

### Random Routing

Whole-layer program device time in microseconds, one seeded routing draw per batch size, identical across implementations.

| Implementation | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|
| Production | 100.6 | 118.8 | 151.0 | 235.3 | 261.9 | 359.2 | 602.0 | 1077.0 |
| Fused GMM | 95.1 | 92.0 | 125.5 | 211.2 | 240.2 | 360.3 | 590.9 | 1095.5 |
| PR 3288 | 114.5 | 117.8 | 117.5 | **120.4** | **160.6** | **225.0** | **348.5** | **616.4** |
| PR 3040 | **73.4** | **76.7** | **84.4** | 123.6 | 183.6 | 311.0 | 595.9 | 1220.8 |
| SGLang-JAX V2 | Eight-Bit Only | Eight-Bit Only | Eight-Bit Only | Eight-Bit Only | Eight-Bit Only | Eight-Bit Only | Eight-Bit Only | Eight-Bit Only |
| SGLang-JAX V1 | Eight-Bit Only | Eight-Bit Only | Eight-Bit Only | Eight-Bit Only | Eight-Bit Only | Eight-Bit Only | Eight-Bit Only | Eight-Bit Only |
| Recommended | 95.1 | 92.0 | 117.5 | 120.4 | 160.6 | 225.0 | 348.5 | 616.4 |

| Comparison | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|
| Recommended vs Production | +5.8% | +29.2% | +28.5% | +95.5% | +63.1% | +59.6% | +72.8% | +74.7% |

## GPT-OSS 120B FP4

128 experts, top-4 selection, hidden 2880, MoE intermediate 2880, from the model's configuration file. Weights are FP4 (E2M1) with 512-wide scale blocks. Both dimensions are padded to 3072, identically for every implementation. The model's expert biases and its clamped activation are omitted identically (plain silu, no bias) so every implementation runs the same math. No serving configuration exists for this model, so Production and Fused GMM run upstream at its own defaults, and their dispatch takes the ragged dispatch path at every batch.

### Random Routing

Whole-layer program device time in microseconds, one seeded routing draw per batch size, identical across implementations.

| Implementation | 128 | 512 | 2048 | 8192 |
|---|---|---|---|---|
| Production | 190.0 | 298.2 | 440.4 | 1226.1 |
| Fused GMM | **149.9** | 271.1 | 419.1 | 1235.0 |
| PR 3288 | 232.8 | 266.1 | **306.0** | **700.3** |
| PR 3040 | 161.4 | **182.9** | 375.2 | 1141.2 |
| SGLang-JAX V2 | Eight-Bit Only | Eight-Bit Only | Eight-Bit Only | Eight-Bit Only |
| SGLang-JAX V1 | Eight-Bit Only | Eight-Bit Only | Eight-Bit Only | Eight-Bit Only |
| Recommended | 149.9 | 266.1 | 306.0 | 700.3 |

| Comparison | 128 | 512 | 2048 | 8192 |
|---|---|---|---|---|
| Recommended vs Production | +26.8% | +12.1% | +43.9% | +75.1% |

## Feature Comparison

| | Production | Fused GMM | PR 3288 | PR 3040 | SGLang-JAX V2 | SGLang-JAX V1 |
|---|---|---|---|---|---|---|
| Expert Selection | Built-in scoring and top-k | Built-in scoring and top-k | Built-in scoring and top-k | Built-in scoring and top-k | External, indices precomputed | External, indices precomputed |
| Dispatch | Two all-gathers under the serving settings, none at upstream defaults | Two all-gathers under the serving settings, none at upstream defaults | Two all-gathers, tokens and routing metadata | All-gather to full replication, then in-kernel per-token reads | All-to-all, one transfer per routed token | All-to-all, one transfer per routed token |
| Expert Compute | Two grouped matmul calls | Both matmuls and activation in one kernel | Gather through combine in one kernel | Gather through combine in one kernel | Per-expert matmul pair | Per-expert matmul pair |
| Combine | Gather-reduce, then all-reduce or reduce-scatter | Gather-reduce, then all-reduce or reduce-scatter | Two transfers per expert-destination run | One transfer per routed token, two on FP8 | One transfer per expert-destination run | One transfer per expert-destination run |
| Combine Dtype | BF16 | BF16 | BF16, FP8 | BF16, FP8 | BF16 | BF16 |
| Empty Experts | Skipped entirely | Skipped entirely | Skipped entirely | Compute skipped, loop runs full length | Skipped entirely | Compute skipped, every expert visited |
| Weight Formats | BF16, FP8, FP4, INT8, INT4 | FP8, FP4, INT8, INT4 | BF16, FP8, FP4, INT8 | BF16, FP8, FP4 | BF16, FP8 | BF16, FP8 |
| Activation Quantization | In-kernel FP8 or INT8, 512-wide blocks | In-kernel FP8 or INT8, mandatory | In-kernel FP8, per token | In-kernel FP8, 256- or 512-wide blocks | In-kernel FP8, per token | None |
| Shape Limits | Tokens x top-k multiple of 16, SparseCore divisibility | Scales required, VMEM budget | Hidden 128-aligned, experts divisible by core count, VMEM budget | Tokens x top-k under 262,144, multiple of 16 | Hidden 128-aligned, block-width divisibility, SMEM budget | Alignment rules, VMEM and HBM budgets |

## Method

The instrument is whole-layer program device time from device profiles, never host timers, and every measurement runs the layer at eight-way expert parallelism. Every session opens and closes with a control measurement that brackets it, and the session's drift is recorded. Control readings agree across sessions to within a fraction of a percent, and every published value is a single reading bracketed by them.

The recorded-routing capture comes from a serving session of the 397B model under eight-way expert parallelism and evaluation traffic. A replayed measurement reproduces the captured routing exactly, with the capture's start-up records excluded. Token values are synthetic and only the routing is real, which is the quantity that determines an MoE layer's cost. Every implementation replays the same steps.

Random-routing measurements use one seeded draw per batch size, identical across implementations. **A random draw loads the experts nearly evenly, so the advantage of skipping empty experts, which real traffic rewards, barely appears in the random-routing tables.**

Every implementation runs as it ships, with its own defaults, its own configuration selector, and no tuning of ours substituted for its own. The dispatch settings are read only by Production and Fused GMM. PR 3288, PR 3040 and the SGLang kernels carry their own switches.

Production is measured at the upstream pin. For the 397B it runs the serving command's settings, and for the models with no serving configuration it runs upstream defaults. PR 3288 runs the served configuration. PR 3040 runs its shipped defaults. The SGLang kernels run through their own configuration selectors, and our copy of each selector was verified against their source at the pin.

Each implementation is timed at its own calling convention. The SGLang rows exclude expert-selection cost because their kernels take the selection precomputed, while every other row includes it, a difference in their favor that grows with the batch. PR 3040's measured program includes the all-gather that replicates every token to every core, the transport its kernel requires the caller to have done. On a serving deployment with data parallelism its driver takes the tokens already distributed, so that cost belongs to this benchmark's mesh rather than to the kernel.

PR 3040's release omits its device-tuned block-size tables by its own note and says to retune per device, so its defaults are untuned. Hand-written tile entries tried in its place at the largest batch ran faster than the shipped default while still trailing PR 3288 by a wide margin. Its FP8 combine and its sequence-parallel switch both measured slower than its defaults, and the tables keep the shipped configuration.

Every measured value carries its complete configuration, source hashes, compiler options and routing provenance, and every number in the timing tables is checked against those records. Where an implementation refuses a configuration, the refusal is the reported result, attributed to the kernel, the shape, or the harness, and the table entry states the reason in a few words. A refusal attributed to the harness never renders as an implementation's result.
