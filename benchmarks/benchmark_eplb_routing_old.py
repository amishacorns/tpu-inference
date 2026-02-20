#!/usr/bin/env python3
"""EPLB routing analysis: compute vs communication cost of expert sharding.

Uses real DeepSeek-R1 router weights (layer 4) with e_score_correction_bias.
Routing: logits = x @ W^T → scores = sigmoid(logits) + e_correction → top-k.

Two cost metrics per device:
  - Compute load: total token-slots (each token × each local expert it hits).
    This is proportional to GMM FLOPs.
  - Comm load: unique tokens that must be sent to the device via all-to-all.
    A token sent once is replicated on-chip for all its local experts.
"""

import time
import json
import numpy as np
from scipy.stats import spearmanr
import torch
from safetensors import safe_open
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

PLOT_DIR = '/mnt/pd/eplb_plots'

# ── Config ──
NUM_EXPERTS = 256
TOP_K = 8
HIDDEN = 7168
EP_SIZE = 8
EXPERTS_PER_DEV = NUM_EXPERTS // EP_SIZE  # 32

NUM_BATCHES = 200
BATCH_SIZE = 1024

CKPT_DIR = '/mnt/pd/checkpoints/deepseek-r1-fp4-mlp-256'
LAYER = 4
SHARD_IDX = 4  # shard containing layer 4 router weights


def load_router():
    """Load real DeepSeek-R1 router weights from checkpoint.

    Returns:
        W: gate weight [E, H] as float32
        e_corr: e_score_correction_bias [E] as float32
    """
    path = f'{CKPT_DIR}/model-{SHARD_IDX:05d}-of-000163.safetensors'
    print(f"  Loading router from {path}")
    with safe_open(path, framework='pt') as f:
        W = f.get_tensor(f'model.layers.{LAYER}.mlp.gate.weight')
        e_corr = f.get_tensor(f'model.layers.{LAYER}.mlp.gate.e_score_correction_bias')
    W = W.float().numpy()       # [E=256, H=7168]
    e_corr = e_corr.float().numpy()  # [E=256]
    print(f"  gate.weight: {W.shape}, e_correction: {e_corr.shape}")
    print(f"  e_correction range: [{e_corr.min():.4f}, {e_corr.max():.4f}], "
          f"mean={e_corr.mean():.4f}")
    return W, e_corr


def route(hidden, W, e_corr, top_k=TOP_K):
    """DeepSeek routing: sigmoid(x @ W^T) + e_correction → top-k indices.

    Matches deepseek_v3.py:
      scores = sigmoid(gate_proj(x))
      scores += bias_E          # e_score_correction
      topk_indices = topk(scores, k)
    """
    logits = hidden @ W.T                                    # [T, E]
    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))  # sigmoid
    scores += e_corr[None, :]                                # + correction
    idx = np.argpartition(scores, -top_k, axis=-1)[:, -top_k:]
    row = np.arange(hidden.shape[0])[:, None]
    vals = scores[row, idx]
    order = np.argsort(-vals, axis=-1)
    idx = np.take_along_axis(idx, order, axis=1)
    return idx  # [T, top_k]


def collect(W, e_corr, num_batches=NUM_BATCHES, batch_size=BATCH_SIZE, seed=0):
    """Run router over random batches.

    Returns:
        counts: [num_batches, E] expert token-slot counts per batch
        total:  [E] aggregate expert token-slot counts
        all_ids: list of [batch_size, top_k] per-token expert indices per batch
    """
    rng = np.random.RandomState(seed)
    counts = np.zeros((num_batches, NUM_EXPERTS), dtype=np.int64)
    all_ids = []
    for i in range(num_batches):
        x = rng.randn(batch_size, HIDDEN).astype(np.float32) * 0.1
        ids = route(x, W, e_corr)  # [batch_size, top_k]
        counts[i] = np.bincount(ids.flatten(), minlength=NUM_EXPERTS)
        all_ids.append(ids)
    return counts, counts.sum(axis=0), all_ids


def device_loads_compute(expert_counts, assignment):
    """Compute load: total token-slots per device (GMM work)."""
    loads = np.zeros(EP_SIZE, dtype=np.float64)
    for e in range(NUM_EXPERTS):
        loads[assignment[e]] += expert_counts[e]
    return loads


def device_loads_comm(token_ids, assignment):
    """Comm load: unique tokens per device (all-to-all volume).

    token_ids: [T, top_k] expert indices per token.
    assignment: [E] → device.
    Returns: [EP_SIZE] unique token count per device.
    """
    T = token_ids.shape[0]
    # For each token, find which devices it touches
    dev_ids = assignment[token_ids]  # [T, top_k] → device per slot
    loads = np.zeros(EP_SIZE, dtype=np.float64)
    for d in range(EP_SIZE):
        # Count tokens that have at least one expert on device d
        hits = np.any(dev_ids == d, axis=1)  # [T] bool
        loads[d] = hits.sum()
    return loads


def static_assignment():
    """Linear: experts 0-31→dev0, 32-63→dev1, ..."""
    return np.repeat(np.arange(EP_SIZE), EXPERTS_PER_DEV)


def optimal_assignment(expert_counts):
    """Greedy bin-pack: heaviest expert → lightest device (32 per device).
    Minimizes max compute load (token-slots)."""
    order = np.argsort(-expert_counts)
    assign = np.zeros(NUM_EXPERTS, dtype=np.int64)
    dev_load = np.zeros(EP_SIZE, dtype=np.float64)
    dev_count = np.zeros(EP_SIZE, dtype=np.int64)
    for e in order:
        candidates = np.where(dev_count < EXPERTS_PER_DEV)[0]
        if len(candidates) == 0:
            candidates = np.arange(EP_SIZE)
        best = candidates[np.argmin(dev_load[candidates])]
        assign[e] = best
        dev_load[best] += expert_counts[e]
        dev_count[best] += 1
    return assign


def imb(loads):
    """Imbalance ratio: max / mean."""
    return loads.max() / loads.mean() if loads.mean() > 0 else 1.0


def main():
    print("Loading DeepSeek-R1 router weights (layer 4)...")
    W, e_corr = load_router()

    print("\nCollecting routing patterns...")
    t0 = time.time()
    per_batch, total, all_ids = collect(W, e_corr)
    dt = time.time() - t0
    print(f"  {NUM_BATCHES}×{BATCH_SIZE} tokens, {dt:.1f}s")
    print(f"  Routed token-slots: {total.sum()}")

    # ── Expert-level stats ──
    print(f"\n{'='*60}")
    print(f"  Expert Load Distribution (256 experts)")
    print(f"{'='*60}")
    print(f"  mean={total.mean():.0f}  std={total.std():.0f}  "
          f"CV={total.std()/total.mean():.4f}")
    mn, mx = total.min(), total.max()
    print(f"  min={mn} (e{total.argmin()})  max={mx} (e{total.argmax()})  "
          f"ratio={mx/mn if mn>0 else 'inf':.2f}x")
    top5 = np.argsort(-total)[:5]
    print(f"  top-5: {[(int(e), int(total[e])) for e in top5]}")

    # ── Build co-occurrence matrix ──
    print(f"\nBuilding co-occurrence matrix...")
    t0_co = time.time()
    cooccur = np.zeros((NUM_EXPERTS, NUM_EXPERTS), dtype=np.int64)
    for ids in all_ids:
        for t in range(ids.shape[0]):
            experts = ids[t]  # [top_k]
            for i in range(TOP_K):
                for j in range(i+1, TOP_K):
                    cooccur[experts[i], experts[j]] += 1
                    cooccur[experts[j], experts[i]] += 1
    print(f"  Done in {time.time() - t0_co:.1f}s")

    # ── Two assignment strategies ──
    sa = static_assignment()
    oa = optimal_assignment(total)

    # Aggregate all token IDs for comm measurement
    all_ids_cat = np.concatenate(all_ids, axis=0)  # [total_tokens, top_k]

    # Compute all loads upfront
    sl_comp = device_loads_compute(total, sa)
    ol_comp = device_loads_compute(total, oa)

    sl_comm = device_loads_comm(all_ids_cat, sa)
    ol_comm = device_loads_comm(all_ids_cat, oa)

    # ── 1. Compute Load comparison ──
    print(f"\n{'='*70}")
    print(f"  1. Compute Load: token-slots per device (GMM work)")
    print(f"{'='*70}")
    print(f"  {'':>5} {'static':>10} {'optimal':>10}")
    print(f"  {'-'*30}")
    for d in range(EP_SIZE):
        s_bar = '█' * int(sl_comp[d] / sl_comp.max() * 30)
        print(f"  dev{d}: {sl_comp[d]:>9.0f}  {ol_comp[d]:>9.0f}  {s_bar}")
    print(f"  max:  {sl_comp.max():>9.0f}  {ol_comp.max():>9.0f}")
    print(f"  imb:  {imb(sl_comp):>9.4f}x {imb(ol_comp):>9.4f}x")
    reshard_comp_pct = (sl_comp.max() - ol_comp.max()) / sl_comp.max() * 100
    print(f"  → Resharding reduces max compute by {reshard_comp_pct:.1f}%")

    # ── 2. Comm Load comparison ──
    print(f"\n{'='*70}")
    print(f"  2. Comm Load: unique tokens per device (all-to-all volume)")
    print(f"{'='*70}")
    print(f"  {'':>5} {'static':>10} {'optimal':>10}")
    print(f"  {'-'*30}")
    for d in range(EP_SIZE):
        s_bar = '█' * int(sl_comm[d] / sl_comm.max() * 30)
        print(f"  dev{d}: {sl_comm[d]:>9.0f}  {ol_comm[d]:>9.0f}  {s_bar}")
    total_tokens = all_ids_cat.shape[0]
    print(f"  max:  {sl_comm.max():>9.0f}  {ol_comm.max():>9.0f}")
    print(f"  imb:  {imb(sl_comm):>9.4f}x {imb(ol_comm):>9.4f}x")
    reshard_comm_pct = (sl_comm.max() - ol_comm.max()) / sl_comm.max() * 100
    print(f"  → Resharding reduces max comm by {reshard_comm_pct:.1f}%")
    print(f"  Total tokens: {total_tokens}  "
          f"Mean unique/dev: {sl_comm.mean():.0f}  "
          f"(each token goes to {sl_comm.sum()/total_tokens:.1f} devices on avg)")

    # ── 3. Compute vs Comm comparison ──
    print(f"\n{'='*70}")
    print(f"  3. Compute vs Comm: fan-out (optimal assignment)")
    print(f"{'='*70}")
    print(f"  {'':>5} {'tok_slots':>10} {'uniq_tok':>10} {'fan_out':>8}")
    print(f"  {'-'*40}")
    for d in range(EP_SIZE):
        fo = ol_comp[d] / ol_comm[d] if ol_comm[d] > 0 else 0
        print(f"  dev{d}: {ol_comp[d]:>9.0f}  {ol_comm[d]:>9.0f}  {fo:>7.2f}x")
    total_fo = ol_comp.sum() / ol_comm.sum()
    print(f"  mean fan-out: {total_fo:.2f}x  "
          f"(each token arriving at a device is used by {total_fo:.2f} local experts on avg)")

    # ── 4. Per-batch variation (both metrics) ──
    print(f"\n{'='*70}")
    print(f"  4. Per-Batch Imbalance (temporal variation)")
    print(f"{'='*70}")
    bis_comp = np.array([imb(device_loads_compute(per_batch[i], sa)) for i in range(NUM_BATCHES)])
    bio_comp = np.array([imb(device_loads_compute(per_batch[i], oa)) for i in range(NUM_BATCHES)])
    bis_comm = np.array([imb(device_loads_comm(all_ids[i], sa)) for i in range(NUM_BATCHES)])
    bio_comm = np.array([imb(device_loads_comm(all_ids[i], oa)) for i in range(NUM_BATCHES)])
    print(f"  Compute (token-slots):")
    print(f"    Static:  mean={bis_comp.mean():.3f}x  std={bis_comp.std():.3f}  max={bis_comp.max():.3f}x")
    print(f"    Optimal: mean={bio_comp.mean():.3f}x  std={bio_comp.std():.3f}  max={bio_comp.max():.3f}x")
    print(f"  Comm (unique tokens):")
    print(f"    Static:  mean={bis_comm.mean():.3f}x  std={bis_comm.std():.3f}  max={bis_comm.max():.3f}x")
    print(f"    Optimal: mean={bio_comm.mean():.3f}x  std={bio_comm.std():.3f}  max={bio_comm.max():.3f}x")

    # ── 5. Expert Rank Stability ──
    print(f"\n{'='*70}")
    print(f"  5. Expert Rank Stability Across Batches")
    print(f"{'='*70}")

    # Spearman rank correlation between batch pairs
    pair_rng = np.random.RandomState(99)
    pairs = pair_rng.choice(NUM_BATCHES, size=(50, 2), replace=True)
    rhos = [spearmanr(per_batch[i], per_batch[j]).correlation
            for i, j in pairs if i != j]
    print(f"  Spearman rank corr (50 batch pairs):")
    print(f"    mean={np.mean(rhos):.4f}  std={np.std(rhos):.4f}  "
          f"min={np.min(rhos):.4f}  max={np.max(rhos):.4f}")

    # Top-K overlap stability
    for K in [10, 20, 50]:
        global_topK = set(np.argsort(-total)[:K])
        overlaps = [len(global_topK & set(np.argsort(-per_batch[i])[:K])) / K
                     for i in range(NUM_BATCHES)]
        overlaps = np.array(overlaps)
        print(f"  Top-{K:>2} overlap w/ global: "
              f"mean={overlaps.mean():.3f}  std={overlaps.std():.3f}  "
              f"min={overlaps.min():.3f}")

    # Per-expert CV
    expert_means = per_batch.mean(axis=0).astype(np.float64)
    expert_stds = per_batch.std(axis=0).astype(np.float64)
    expert_cv = np.where(expert_means > 0, expert_stds / expert_means, 0)
    print(f"  Per-expert CV: mean={expert_cv.mean():.3f}  "
          f"median={np.median(expert_cv):.3f}")

    # Top-10 rank stability
    top10 = np.argsort(-total)[:10]
    print(f"\n  Top-10 experts rank stability:")
    print(f"  {'expert':>7} {'total':>8} {'mean':>8} {'std':>8} "
          f"{'CV':>6} {'min_rk':>7} {'med_rk':>7} {'max_rk':>7}")
    for e in top10:
        ranks = np.array([int(np.where(np.argsort(-per_batch[i]) == e)[0][0]) + 1
                          for i in range(NUM_BATCHES)])
        print(f"  {e:>7} {total[e]:>8} {expert_means[e]:>8.1f} "
              f"{expert_stds[e]:>8.1f} {expert_cv[e]:>6.3f} "
              f"{ranks.min():>7} {int(np.median(ranks)):>7} {ranks.max():>7}")

    # Bottom-10 rank stability
    bot10 = np.argsort(total)[:10]
    print(f"\n  Bottom-10 experts rank stability:")
    print(f"  {'expert':>7} {'total':>8} {'mean':>8} {'std':>8} "
          f"{'CV':>6} {'min_rk':>7} {'med_rk':>7} {'max_rk':>7}")
    for e in bot10:
        ranks = np.array([int(np.where(np.argsort(-per_batch[i]) == e)[0][0]) + 1
                          for i in range(NUM_BATCHES)])
        print(f"  {e:>7} {total[e]:>8} {expert_means[e]:>8.1f} "
              f"{expert_stds[e]:>8.1f} {expert_cv[e]:>6.3f} "
              f"{ranks.min():>7} {int(np.median(ranks)):>7} {ranks.max():>7}")

    # ── 6. Expert Co-selection (which experts share tokens?) ──
    print(f"\n{'='*70}")
    print(f"  6. Expert Co-selection Analysis")
    print(f"{'='*70}")
    # Top co-selected pairs
    upper = np.triu(cooccur, k=1)
    flat_idx = np.argsort(-upper.flatten())[:15]
    pairs_top = [(idx // NUM_EXPERTS, idx % NUM_EXPERTS) for idx in flat_idx]
    print(f"  Top-15 co-selected expert pairs:")
    print(f"  {'e_i':>5} {'e_j':>5} {'co_count':>10} {'same_static':>16} {'same_optimal':>16}")
    for ei, ej in pairs_top:
        cc = cooccur[ei, ej]
        sd_s = 'yes' if sa[ei] == sa[ej] else 'no'
        sd_o = 'yes' if oa[ei] == oa[ej] else 'no'
        print(f"  {ei:>5} {ej:>5} {cc:>10} {sd_s:>16} {sd_o:>16}")

    # How many co-selected pairs land on same device?
    total_pairs = 0
    same_dev_static = 0
    same_dev_optimal = 0
    weighted_same_static = 0
    weighted_same_optimal = 0
    for ei in range(NUM_EXPERTS):
        for ej in range(ei+1, NUM_EXPERTS):
            if cooccur[ei, ej] > 0:
                total_pairs += 1
                w = cooccur[ei, ej]
                if sa[ei] == sa[ej]:
                    same_dev_static += 1
                    weighted_same_static += w
                if oa[ei] == oa[ej]:
                    same_dev_optimal += 1
                    weighted_same_optimal += w
    total_co_weight = upper.sum()
    print(f"\n  Co-located pair fraction (unweighted):")
    print(f"    static:  {same_dev_static}/{total_pairs} ({same_dev_static/total_pairs*100:.1f}%)")
    print(f"    optimal: {same_dev_optimal}/{total_pairs} ({same_dev_optimal/total_pairs*100:.1f}%)")
    print(f"  Co-located pair fraction (weighted by co-count):")
    print(f"    static:  {weighted_same_static/total_co_weight*100:.1f}%")
    print(f"    optimal: {weighted_same_optimal/total_co_weight*100:.1f}%")
    print(f"  Expected if random: {1/EP_SIZE*100:.1f}% (1/{EP_SIZE} chance two experts on same device)")

    # ── 7. Summary ──
    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    print(f"                      {'static':>12} {'optimal':>12} {'reduction':>10}")
    print(f"  {'-'*50}")
    print(f"  Compute (max slots) {sl_comp.max():>12.0f} {ol_comp.max():>12.0f} {reshard_comp_pct:>9.1f}%")
    print(f"  Comm (max uniq tok) {sl_comm.max():>12.0f} {ol_comm.max():>12.0f} {reshard_comm_pct:>9.1f}%")
    print(f"  Compute imbalance   {imb(sl_comp):>12.4f}x {imb(ol_comp):>12.4f}x")
    print(f"  Comm imbalance      {imb(sl_comm):>12.4f}x {imb(ol_comm):>12.4f}x")
    print(f"  Fan-out (avg)       {'':>12} {total_fo:>12.2f}x")

    # ── Generate plots ──
    print(f"\nGenerating plots...")
    import os
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_all(
        total=total, per_batch=per_batch, all_ids=all_ids,
        sa=sa, oa=oa,
        sl_comp=sl_comp, ol_comp=ol_comp,
        sl_comm=sl_comm, ol_comm=ol_comm,
        bis_comp=bis_comp, bio_comp=bio_comp,
        bis_comm=bis_comm, bio_comm=bio_comm,
        cooccur=cooccur,
    )
    print(f"  Plots saved to {PLOT_DIR}/")

    with open('/mnt/pd/eplb_routing_analysis.json', 'w') as f:
        json.dump({
            'config': {'E': NUM_EXPERTS, 'K': TOP_K, 'EP': EP_SIZE},
            'expert_cv': float(total.std()/total.mean()),
            'static_compute_max': float(sl_comp.max()),
            'optimal_compute_max': float(ol_comp.max()),
            'reshard_compute_pct': float(reshard_comp_pct),
            'static_comm_max': float(sl_comm.max()),
            'optimal_comm_max': float(ol_comm.max()),
            'reshard_comm_pct': float(reshard_comm_pct),
            'avg_fan_out': float(total_fo),
        }, f, indent=2)
    print(f"\nSaved to /mnt/pd/eplb_routing_analysis.json")


def plot_all(*, total, per_batch, all_ids,
             sa, oa,
             sl_comp, ol_comp,
             sl_comm, ol_comm,
             bis_comp, bio_comp,
             bis_comm, bio_comm,
             cooccur):
    """Generate all analysis plots."""
    COLORS = {'static': '#5B8BD0', 'optimal': '#2EAA4A'}
    devs = [f'D{d}' for d in range(EP_SIZE)]

    # ── Fig 1: Expert Load Distribution (sorted bar + CDF) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sorted_total = np.sort(total)[::-1]
    ax1.bar(range(NUM_EXPERTS), sorted_total, width=1.0, color='#5B8BD0', alpha=0.8)
    ax1.axhline(total.mean(), color='red', ls='--', lw=1.5, label=f'mean={total.mean():.0f}')
    ax1.set_xlabel('Expert (sorted by load)', fontsize=11)
    ax1.set_ylabel('Token-slots', fontsize=11)
    ax1.set_title('Expert Load Distribution (sorted)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim(-1, NUM_EXPERTS)

    # CDF
    cdf_x = np.sort(total)
    cdf_y = np.arange(1, NUM_EXPERTS + 1) / NUM_EXPERTS
    ax2.plot(cdf_x, cdf_y, lw=2, color='#5B8BD0')
    ax2.axvline(total.mean(), color='red', ls='--', lw=1.5, label=f'mean={total.mean():.0f}')
    ax2.axvline(np.median(total), color='orange', ls='--', lw=1.5, label=f'median={np.median(total):.0f}')
    ax2.set_xlabel('Token-slots per expert', fontsize=11)
    ax2.set_ylabel('CDF', fontsize=11)
    ax2.set_title('Expert Load CDF', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/1_expert_load_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 2: Device Load Comparison (grouped bars) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(EP_SIZE)
    w = 0.35
    # Compute load
    ax1.bar(x - w/2, sl_comp, w, label='Static', color=COLORS['static'], alpha=0.85)
    ax1.bar(x + w/2, ol_comp, w, label='Optimal', color=COLORS['optimal'], alpha=0.85)
    ax1.axhline(sl_comp.mean(), color=COLORS['static'], ls=':', lw=1, alpha=0.6)
    ax1.set_xticks(x); ax1.set_xticklabels(devs)
    ax1.set_ylabel('Token-slots', fontsize=11)
    ax1.set_title('Compute Load per Device', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # Comm load
    ax2.bar(x - w/2, sl_comm, w, label='Static', color=COLORS['static'], alpha=0.85)
    ax2.bar(x + w/2, ol_comm, w, label='Optimal', color=COLORS['optimal'], alpha=0.85)
    ax2.axhline(sl_comm.mean(), color=COLORS['static'], ls=':', lw=1, alpha=0.6)
    ax2.set_xticks(x); ax2.set_xticklabels(devs)
    ax2.set_ylabel('Unique tokens', fontsize=11)
    ax2.set_title('Communication Load per Device', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/2_device_load_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 3: Per-Batch Imbalance Over Time ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    batches = np.arange(NUM_BATCHES)
    ax1.plot(batches, bis_comp, lw=0.8, alpha=0.7, color=COLORS['static'], label='Static')
    ax1.plot(batches, bio_comp, lw=0.8, alpha=0.7, color=COLORS['optimal'], label='Optimal')
    ax1.axhline(1.0, color='black', ls='-', lw=0.5, alpha=0.3)
    ax1.set_ylabel('Imbalance (max/mean)', fontsize=11)
    ax1.set_title('Compute Imbalance per Batch', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.2)

    ax2.plot(batches, bis_comm, lw=0.8, alpha=0.7, color=COLORS['static'], label='Static')
    ax2.plot(batches, bio_comm, lw=0.8, alpha=0.7, color=COLORS['optimal'], label='Optimal')
    ax2.axhline(1.0, color='black', ls='-', lw=0.5, alpha=0.3)
    ax2.set_xlabel('Batch index', fontsize=11)
    ax2.set_ylabel('Imbalance (max/mean)', fontsize=11)
    ax2.set_title('Communication Imbalance per Batch', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/3_per_batch_imbalance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 4: Co-selection Heatmap ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sort experts by total load for better visual structure
    sort_idx = np.argsort(-total)
    cooccur_sorted = cooccur[np.ix_(sort_idx, sort_idx)]

    # Full heatmap (log scale)
    mask = cooccur_sorted.copy().astype(float)
    mask[mask == 0] = np.nan
    im = axes[0].imshow(mask, cmap='inferno', norm=LogNorm(vmin=1, vmax=cooccur.max()),
                        aspect='auto', interpolation='nearest')
    axes[0].set_title('Expert Co-selection (log scale)\n(sorted by popularity)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Expert (sorted)', fontsize=10)
    axes[0].set_ylabel('Expert (sorted)', fontsize=10)
    plt.colorbar(im, ax=axes[0], label='Co-selection count', shrink=0.8)

    # Zoomed top-32 (the experts that matter most)
    top32 = sort_idx[:32]
    cooccur_top = cooccur[np.ix_(top32, top32)]
    im2 = axes[1].imshow(cooccur_top, cmap='inferno', aspect='auto', interpolation='nearest')
    axes[1].set_title('Top-32 Experts Co-selection\n(sorted by popularity)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Expert rank', fontsize=10)
    axes[1].set_ylabel('Expert rank', fontsize=10)
    # Label ticks with expert IDs
    tick_pos = list(range(0, 32, 4))
    axes[1].set_xticks(tick_pos)
    axes[1].set_xticklabels([f'e{top32[i]}' for i in tick_pos], fontsize=7, rotation=45)
    axes[1].set_yticks(tick_pos)
    axes[1].set_yticklabels([f'e{top32[i]}' for i in tick_pos], fontsize=7)
    plt.colorbar(im2, ax=axes[1], label='Co-selection count', shrink=0.8)

    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/4_coselection_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 5: Expert Rank Stability ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: rank trajectories of top-10 experts across batches
    top10 = np.argsort(-total)[:10]
    cmap = plt.cm.tab10
    for rank_i, e in enumerate(top10):
        ranks_across = np.array([
            int(np.where(np.argsort(-per_batch[i]) == e)[0][0]) + 1
            for i in range(NUM_BATCHES)
        ])
        ax1.plot(range(NUM_BATCHES), ranks_across, lw=0.9, alpha=0.8,
                 color=cmap(rank_i), label=f'e{e}')
    ax1.set_xlabel('Batch index', fontsize=11)
    ax1.set_ylabel('Rank (1=most popular)', fontsize=11)
    ax1.set_title('Top-10 Expert Rank Over Batches', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.legend(fontsize=7, ncol=2, loc='lower right')
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(30, 0)

    # Right: per-expert CV vs mean load (scatter)
    expert_means = per_batch.mean(axis=0).astype(np.float64)
    expert_stds = per_batch.std(axis=0).astype(np.float64)
    expert_cv = np.where(expert_means > 0, expert_stds / expert_means, 0)
    sc = ax2.scatter(expert_means, expert_cv, s=12, alpha=0.6, c=total,
                     cmap='viridis', edgecolors='none')
    plt.colorbar(sc, ax=ax2, label='Total token-slots', shrink=0.8)
    # Annotate top-5
    top5 = np.argsort(-total)[:5]
    for e in top5:
        ax2.annotate(f'e{e}', (expert_means[e], expert_cv[e]),
                     fontsize=7, fontweight='bold', color='red',
                     xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('Mean batch load', fontsize=11)
    ax2.set_ylabel('CV (std/mean)', fontsize=11)
    ax2.set_title('Expert Load Stability (CV vs Mean)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/5_rank_stability.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 6: Summary Comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Imbalance comparison bar chart
    strategies = ['Static', 'Optimal']
    comp_imbs = [imb(sl_comp), imb(ol_comp)]
    comm_imbs = [imb(sl_comm), imb(ol_comm)]
    x = np.arange(2)
    w = 0.3
    axes[0].bar(x - w/2, comp_imbs, w, label='Compute', color='#5B8BD0', alpha=0.85)
    axes[0].bar(x + w/2, comm_imbs, w, label='Comm', color='#E8A838', alpha=0.85)
    axes[0].axhline(1.0, color='black', ls='--', lw=0.8, alpha=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(strategies, fontsize=10)
    axes[0].set_ylabel('Imbalance (max/mean)', fontsize=11)
    axes[0].set_title('Imbalance by Strategy', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9)
    for i, (cv, ccv) in enumerate(zip(comp_imbs, comm_imbs)):
        axes[0].text(i - w/2, cv + 0.005, f'{cv:.3f}x', ha='center', fontsize=9, fontweight='bold')
        axes[0].text(i + w/2, ccv + 0.005, f'{ccv:.3f}x', ha='center', fontsize=9, fontweight='bold')

    # (b) Max load comparison
    comp_maxs = [sl_comp.max(), ol_comp.max()]
    comm_maxs = [sl_comm.max(), ol_comm.max()]
    axes[1].bar(x - w/2, [v/1000 for v in comp_maxs], w, label='Compute (K)', color='#5B8BD0', alpha=0.85)
    axes[1].bar(x + w/2, [v/1000 for v in comm_maxs], w, label='Comm (K)', color='#E8A838', alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(strategies, fontsize=10)
    axes[1].set_ylabel('Max load (thousands)', fontsize=11)
    axes[1].set_title('Max Device Load by Strategy', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9)
    for i, (cv, ccv) in enumerate(zip(comp_maxs, comm_maxs)):
        axes[1].text(i - w/2, cv/1000 + 2, f'{cv/1000:.0f}K', ha='center', fontsize=9, fontweight='bold')
        axes[1].text(i + w/2, ccv/1000 + 2, f'{ccv/1000:.0f}K', ha='center', fontsize=9, fontweight='bold')

    # (c) Co-location & fan-out
    upper = np.triu(cooccur, k=1)
    total_co_weight = upper.sum()
    coloc = []
    fanouts = []
    for assign, lcomp, lcomm in [(sa, sl_comp, sl_comm), (oa, ol_comp, ol_comm)]:
        wt = 0
        for ei in range(NUM_EXPERTS):
            for ej in range(ei+1, NUM_EXPERTS):
                if assign[ei] == assign[ej]:
                    wt += cooccur[ei, ej]
        coloc.append(wt / total_co_weight * 100)
        fanouts.append(lcomp.sum() / lcomm.sum())
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    bars1 = ax3.bar(x - w/2, coloc, w, label='Co-location %', color='#2EAA4A', alpha=0.85)
    bars2 = ax3_twin.bar(x + w/2, fanouts, w, label='Fan-out', color='#9B59B6', alpha=0.85)
    ax3.axhline(12.5, color='gray', ls='--', lw=0.8, alpha=0.6, label='Random (12.5%)')
    ax3.set_xticks(x); ax3.set_xticklabels(strategies, fontsize=10)
    ax3.set_ylabel('Weighted co-location %', fontsize=11, color='#2EAA4A')
    ax3_twin.set_ylabel('Avg fan-out', fontsize=11, color='#9B59B6')
    ax3.set_title('Co-location & Fan-out', fontsize=13, fontweight='bold')
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
    for i, (c, f) in enumerate(zip(coloc, fanouts)):
        ax3.text(i - w/2, c + 0.5, f'{c:.1f}%', ha='center', fontsize=9, fontweight='bold')
        ax3_twin.text(i + w/2, f + 0.02, f'{f:.2f}x', ha='center', fontsize=9, fontweight='bold')

    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/6_summary_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Generated 6 plots in {PLOT_DIR}/")


if __name__ == '__main__':
    main()
