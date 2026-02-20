#!/usr/bin/env python3
"""EPLB routing analysis: compute vs communication cost of expert sharding.

Loops over ALL 59 MoE layers (3-61) in DeepSeek-R1, computes static vs
optimal (greedy bin-pack) imbalance for both compute and communication,
then produces a cross-layer summary plot and detailed per-layer analysis.

Routing: logits = x @ W^T -> scores = sigmoid(logits) + e_correction -> top-k.

Two cost metrics per device:
  - Compute load: total token-slots (each token x each local expert it hits).
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

CKPT_DIR = '/mnt/pd/checkpoints/deepseek-r1-fp4-mlp-256'

# Sweep config (fewer batches for speed across 59 layers)
SWEEP_BATCHES = 50
SWEEP_BATCH_SIZE = 1024

# Detail config (full analysis for one representative layer)
DETAIL_BATCHES = 200
DETAIL_BATCH_SIZE = 1024
DETAIL_LAYER = 4

# Layer -> shard index mapping (all 59 MoE layers in DeepSeek-R1)
LAYER_SHARD_MAP = {
    3: 1, 4: 4, 5: 7, 6: 9, 7: 13, 8: 15, 9: 18, 10: 21, 11: 23,
    12: 26, 13: 29, 14: 31, 15: 35, 16: 37, 17: 40, 18: 43, 19: 45,
    20: 48, 21: 51, 22: 53, 23: 57, 24: 59, 25: 62, 26: 65, 27: 67,
    28: 70, 29: 73, 30: 75, 31: 79, 32: 81, 33: 84, 34: 87, 35: 89,
    36: 92, 37: 95, 38: 97, 39: 101, 40: 103, 41: 106, 42: 109,
    43: 111, 44: 114, 45: 117, 46: 119, 47: 123, 48: 125, 49: 128,
    50: 131, 51: 133, 52: 136, 53: 139, 54: 142, 55: 144, 56: 147,
    57: 150, 58: 152, 59: 155, 60: 158, 61: 160,
}


def load_router(layer, shard_idx, verbose=True):
    """Load real DeepSeek-R1 router weights from checkpoint."""
    path = f'{CKPT_DIR}/model-{shard_idx:05d}-of-000163.safetensors'
    if verbose:
        print(f"  Loading router from {path}")
    with safe_open(path, framework='pt') as f:
        W = f.get_tensor(f'model.layers.{layer}.mlp.gate.weight')
        e_corr = f.get_tensor(f'model.layers.{layer}.mlp.gate.e_score_correction_bias')
    W = W.float().numpy()
    e_corr = e_corr.float().numpy()
    if verbose:
        print(f"  gate.weight: {W.shape}, e_correction: {e_corr.shape}")
        print(f"  e_correction range: [{e_corr.min():.4f}, {e_corr.max():.4f}], "
              f"mean={e_corr.mean():.4f}")
    return W, e_corr


def route(hidden, W, e_corr, top_k=TOP_K):
    """DeepSeek routing: sigmoid(x @ W^T) + e_correction -> top-k indices."""
    logits = hidden @ W.T
    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    scores += e_corr[None, :]
    idx = np.argpartition(scores, -top_k, axis=-1)[:, -top_k:]
    row = np.arange(hidden.shape[0])[:, None]
    vals = scores[row, idx]
    order = np.argsort(-vals, axis=-1)
    idx = np.take_along_axis(idx, order, axis=1)
    return idx


def collect(W, e_corr, num_batches=200, batch_size=1024, seed=0):
    """Run router over random batches. Returns counts, total, all_ids."""
    rng = np.random.RandomState(seed)
    counts = np.zeros((num_batches, NUM_EXPERTS), dtype=np.int64)
    all_ids = []
    for i in range(num_batches):
        x = rng.randn(batch_size, HIDDEN).astype(np.float32) * 0.1
        ids = route(x, W, e_corr)
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
    """Comm load: unique tokens per device (all-to-all volume)."""
    T = token_ids.shape[0]
    dev_ids = assignment[token_ids]
    loads = np.zeros(EP_SIZE, dtype=np.float64)
    for d in range(EP_SIZE):
        hits = np.any(dev_ids == d, axis=1)
        loads[d] = hits.sum()
    return loads


def static_assignment():
    """Linear: experts 0-31->dev0, 32-63->dev1, ..."""
    return np.repeat(np.arange(EP_SIZE), EXPERTS_PER_DEV)


def optimal_assignment(expert_counts):
    """Greedy bin-pack: heaviest expert -> lightest device (32 per device)."""
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


# ── Lightweight per-layer analysis ────────────────────────────────────────

def analyze_layer(layer, shard_idx):
    """Analyze one layer, return dict of imbalance metrics."""
    W, e_corr = load_router(layer, shard_idx, verbose=False)
    per_batch, total, all_ids = collect(W, e_corr,
                                        num_batches=SWEEP_BATCHES,
                                        batch_size=SWEEP_BATCH_SIZE)
    all_ids_cat = np.concatenate(all_ids, axis=0)

    sa = static_assignment()
    oa = optimal_assignment(total)

    sl_comp = device_loads_compute(total, sa)
    ol_comp = device_loads_compute(total, oa)
    sl_comm = device_loads_comm(all_ids_cat, sa)
    ol_comm = device_loads_comm(all_ids_cat, oa)

    fan_out = ol_comp.sum() / ol_comm.sum() if ol_comm.sum() > 0 else 1.0
    expert_cv = float(total.std() / total.mean()) if total.mean() > 0 else 0.0

    return {
        'layer': layer,
        'static_compute_imb': imb(sl_comp),
        'optimal_compute_imb': imb(ol_comp),
        'static_comm_imb': imb(sl_comm),
        'optimal_comm_imb': imb(ol_comm),
        'compute_reduction_pct': (sl_comp.max() - ol_comp.max()) / sl_comp.max() * 100,
        'comm_reduction_pct': (sl_comm.max() - ol_comm.max()) / sl_comm.max() * 100,
        'fan_out': fan_out,
        'expert_cv': expert_cv,
    }


# ── Cross-layer summary plot ─────────────────────────────────────────────

def plot_cross_layer_summary(results, plot_dir):
    """Plot static vs optimal across all layers."""
    import os
    os.makedirs(plot_dir, exist_ok=True)

    layers = [r['layer'] for r in results]
    static_comp = [r['static_compute_imb'] for r in results]
    opt_comp = [r['optimal_compute_imb'] for r in results]
    static_comm = [r['static_comm_imb'] for r in results]
    opt_comm = [r['optimal_comm_imb'] for r in results]
    comp_red = [r['compute_reduction_pct'] for r in results]
    comm_red = [r['comm_reduction_pct'] for r in results]
    cvs = [r['expert_cv'] for r in results]
    fos = [r['fan_out'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # (a) Compute imbalance
    ax = axes[0, 0]
    ax.plot(layers, static_comp, 'o-', ms=4, lw=1.2, color='#D35400',
            label='Static', alpha=0.85)
    ax.plot(layers, opt_comp, 's-', ms=4, lw=1.2, color='#2EAA4A',
            label='Optimal (bin-pack)', alpha=0.85)
    ax.axhline(1.0, color='black', ls='--', lw=0.7, alpha=0.4)
    ax.set_xlabel('Layer'); ax.set_ylabel('Compute Imbalance (max/mean)')
    ax.set_title('Compute Imbalance Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0.99)

    # (b) Comm imbalance
    ax = axes[0, 1]
    ax.plot(layers, static_comm, 'o-', ms=4, lw=1.2, color='#D35400',
            label='Static', alpha=0.85)
    ax.plot(layers, opt_comm, 's-', ms=4, lw=1.2, color='#2EAA4A',
            label='Optimal (bin-pack)', alpha=0.85)
    ax.axhline(1.0, color='black', ls='--', lw=0.7, alpha=0.4)
    ax.set_xlabel('Layer'); ax.set_ylabel('Comm Imbalance (max/mean)')
    ax.set_title('Communication Imbalance Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0.99)

    # (c) Reduction % bars
    ax = axes[1, 0]
    x = np.arange(len(layers))
    w = 0.35
    ax.bar(x - w/2, comp_red, w, label='Compute', color='#5B8BD0', alpha=0.85)
    ax.bar(x + w/2, comm_red, w, label='Comm', color='#E8A838', alpha=0.85)
    ax.axhline(np.mean(comp_red), color='#5B8BD0', ls=':', lw=1.5, alpha=0.7,
               label=f'Comp mean={np.mean(comp_red):.1f}%')
    ax.axhline(np.mean(comm_red), color='#E8A838', ls=':', lw=1.5, alpha=0.7,
               label=f'Comm mean={np.mean(comm_red):.1f}%')
    ax.set_xticks(x[::5])
    ax.set_xticklabels([layers[i] for i in range(0, len(layers), 5)])
    ax.set_xlabel('Layer'); ax.set_ylabel('Max-load Reduction (%)')
    ax.set_title('Resharding Benefit per Layer', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right'); ax.grid(True, alpha=0.3, axis='y')

    # (d) Expert CV + fan-out
    ax = axes[1, 1]
    ax.plot(layers, cvs, 'o-', ms=4, lw=1.2, color='#8E44AD',
            label='Expert Load CV', alpha=0.85)
    ax.set_xlabel('Layer'); ax.set_ylabel('Expert Load CV', color='#8E44AD')
    ax.tick_params(axis='y', labelcolor='#8E44AD')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(layers, fos, 's-', ms=4, lw=1.2, color='#16A085',
             label='Fan-out', alpha=0.85)
    ax2.set_ylabel('Fan-out', color='#16A085')
    ax2.tick_params(axis='y', labelcolor='#16A085')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')
    ax.set_title('Expert Load Variability & Fan-out', fontsize=14, fontweight='bold')

    fig.suptitle(f'DeepSeek-R1 EPLB Routing — All {len(layers)} MoE Layers\n'
                 f'({NUM_EXPERTS} experts, top-{TOP_K}, EP={EP_SIZE}, '
                 f'{SWEEP_BATCHES}x{SWEEP_BATCH_SIZE} tokens/layer)',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{plot_dir}/0_cross_layer_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Summary plot saved to {plot_dir}/0_cross_layer_summary.png")


# ── Detailed single-layer analysis ───────────────────────────────────────

def detail_analysis(layer, shard_idx, plot_dir):
    """Full detailed analysis + 6 plots for one layer."""
    num_batches = DETAIL_BATCHES
    batch_size = DETAIL_BATCH_SIZE

    print(f"\n{'#'*70}")
    print(f"  DETAILED ANALYSIS — Layer {layer}")
    print(f"{'#'*70}")

    W, e_corr = load_router(layer, shard_idx)
    per_batch, total, all_ids = collect(W, e_corr, num_batches, batch_size)
    all_ids_cat = np.concatenate(all_ids, axis=0)

    print(f"  Expert stats: CV={total.std()/total.mean():.4f}  "
          f"max/min={total.max()}/{total.min()} ({total.max()/total.min():.2f}x)")

    sa = static_assignment()
    oa = optimal_assignment(total)

    sl_comp = device_loads_compute(total, sa)
    ol_comp = device_loads_compute(total, oa)
    sl_comm = device_loads_comm(all_ids_cat, sa)
    ol_comm = device_loads_comm(all_ids_cat, oa)

    # Co-occurrence matrix
    cooccur = np.zeros((NUM_EXPERTS, NUM_EXPERTS), dtype=np.int64)
    for ids in all_ids:
        for t in range(ids.shape[0]):
            experts = ids[t]
            for i in range(TOP_K):
                for j in range(i+1, TOP_K):
                    cooccur[experts[i], experts[j]] += 1
                    cooccur[experts[j], experts[i]] += 1

    rc_pct = (sl_comp.max() - ol_comp.max()) / sl_comp.max() * 100
    rcm_pct = (sl_comm.max() - ol_comm.max()) / sl_comm.max() * 100
    total_fo = ol_comp.sum() / ol_comm.sum()

    print(f"  Compute: static {imb(sl_comp):.4f}x -> optimal {imb(ol_comp):.4f}x ({rc_pct:.1f}% reduction)")
    print(f"  Comm:    static {imb(sl_comm):.4f}x -> optimal {imb(ol_comm):.4f}x ({rcm_pct:.1f}% reduction)")
    print(f"  Fan-out: {total_fo:.2f}x")

    # Per-batch imbalance
    bis_comp = np.array([imb(device_loads_compute(per_batch[i], sa)) for i in range(num_batches)])
    bio_comp = np.array([imb(device_loads_compute(per_batch[i], oa)) for i in range(num_batches)])
    bis_comm = np.array([imb(device_loads_comm(all_ids[i], sa)) for i in range(num_batches)])
    bio_comm = np.array([imb(device_loads_comm(all_ids[i], oa)) for i in range(num_batches)])

    # Generate 6 detail plots
    plot_all(
        layer=layer, total=total, per_batch=per_batch, all_ids=all_ids,
        sa=sa, oa=oa, sl_comp=sl_comp, ol_comp=ol_comp,
        sl_comm=sl_comm, ol_comm=ol_comm,
        bis_comp=bis_comp, bio_comp=bio_comp,
        bis_comm=bis_comm, bio_comm=bio_comm,
        cooccur=cooccur, num_batches=num_batches,
    )


def plot_all(*, layer, total, per_batch, all_ids,
             sa, oa, sl_comp, ol_comp, sl_comm, ol_comm,
             bis_comp, bio_comp, bis_comm, bio_comm,
             cooccur, num_batches):
    """Generate 6 detail plots for a single layer."""
    COLORS = {'static': '#5B8BD0', 'optimal': '#2EAA4A'}
    devs = [f'D{d}' for d in range(EP_SIZE)]
    L = layer

    # ── Fig 1: Expert Load Distribution (sorted bar + CDF) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sorted_total = np.sort(total)[::-1]
    ax1.bar(range(NUM_EXPERTS), sorted_total, width=1.0, color='#5B8BD0', alpha=0.8)
    ax1.axhline(total.mean(), color='red', ls='--', lw=1.5, label=f'mean={total.mean():.0f}')
    ax1.set_xlabel('Expert (sorted by load)', fontsize=11)
    ax1.set_ylabel('Token-slots', fontsize=11)
    ax1.set_title(f'Expert Load Distribution — Layer {L}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim(-1, NUM_EXPERTS)
    cdf_x = np.sort(total)
    cdf_y = np.arange(1, NUM_EXPERTS + 1) / NUM_EXPERTS
    ax2.plot(cdf_x, cdf_y, lw=2, color='#5B8BD0')
    ax2.axvline(total.mean(), color='red', ls='--', lw=1.5, label=f'mean={total.mean():.0f}')
    ax2.axvline(np.median(total), color='orange', ls='--', lw=1.5, label=f'median={np.median(total):.0f}')
    ax2.set_xlabel('Token-slots per expert', fontsize=11)
    ax2.set_ylabel('CDF', fontsize=11)
    ax2.set_title(f'Expert Load CDF — Layer {L}', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/1_expert_load_L{L}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 2: Device Load Comparison (grouped bars) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(EP_SIZE)
    w = 0.35
    ax1.bar(x - w/2, sl_comp, w, label='Static', color=COLORS['static'], alpha=0.85)
    ax1.bar(x + w/2, ol_comp, w, label='Optimal', color=COLORS['optimal'], alpha=0.85)
    ax1.axhline(sl_comp.mean(), color=COLORS['static'], ls=':', lw=1, alpha=0.6)
    ax1.set_xticks(x); ax1.set_xticklabels(devs)
    ax1.set_ylabel('Token-slots', fontsize=11)
    ax1.set_title(f'Compute Load — Layer {L}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax2.bar(x - w/2, sl_comm, w, label='Static', color=COLORS['static'], alpha=0.85)
    ax2.bar(x + w/2, ol_comm, w, label='Optimal', color=COLORS['optimal'], alpha=0.85)
    ax2.axhline(sl_comm.mean(), color=COLORS['static'], ls=':', lw=1, alpha=0.6)
    ax2.set_xticks(x); ax2.set_xticklabels(devs)
    ax2.set_ylabel('Unique tokens', fontsize=11)
    ax2.set_title(f'Communication Load — Layer {L}', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/2_device_load_L{L}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 3: Per-Batch Imbalance Over Time ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    batches = np.arange(num_batches)
    ax1.plot(batches, bis_comp, lw=0.8, alpha=0.7, color=COLORS['static'], label='Static')
    ax1.plot(batches, bio_comp, lw=0.8, alpha=0.7, color=COLORS['optimal'], label='Optimal')
    ax1.axhline(1.0, color='black', ls='-', lw=0.5, alpha=0.3)
    ax1.set_ylabel('Imbalance (max/mean)', fontsize=11)
    ax1.set_title(f'Compute Imbalance per Batch — Layer {L}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.2)
    ax2.plot(batches, bis_comm, lw=0.8, alpha=0.7, color=COLORS['static'], label='Static')
    ax2.plot(batches, bio_comm, lw=0.8, alpha=0.7, color=COLORS['optimal'], label='Optimal')
    ax2.axhline(1.0, color='black', ls='-', lw=0.5, alpha=0.3)
    ax2.set_xlabel('Batch index', fontsize=11)
    ax2.set_ylabel('Imbalance (max/mean)', fontsize=11)
    ax2.set_title(f'Communication Imbalance per Batch — Layer {L}', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/3_per_batch_imbalance_L{L}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 4: Co-selection Heatmap ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sort_idx = np.argsort(-total)
    cooccur_sorted = cooccur[np.ix_(sort_idx, sort_idx)]
    mask = cooccur_sorted.copy().astype(float)
    mask[mask == 0] = np.nan
    im = axes[0].imshow(mask, cmap='inferno', norm=LogNorm(vmin=1, vmax=cooccur.max()),
                        aspect='auto', interpolation='nearest')
    axes[0].set_title(f'Co-selection (log) — Layer {L}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Expert (sorted)', fontsize=10)
    axes[0].set_ylabel('Expert (sorted)', fontsize=10)
    plt.colorbar(im, ax=axes[0], label='Co-selection count', shrink=0.8)
    top32 = sort_idx[:32]
    cooccur_top = cooccur[np.ix_(top32, top32)]
    im2 = axes[1].imshow(cooccur_top, cmap='inferno', aspect='auto', interpolation='nearest')
    axes[1].set_title(f'Top-32 Co-selection — Layer {L}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Expert rank', fontsize=10)
    axes[1].set_ylabel('Expert rank', fontsize=10)
    tick_pos = list(range(0, 32, 4))
    axes[1].set_xticks(tick_pos)
    axes[1].set_xticklabels([f'e{top32[i]}' for i in tick_pos], fontsize=7, rotation=45)
    axes[1].set_yticks(tick_pos)
    axes[1].set_yticklabels([f'e{top32[i]}' for i in tick_pos], fontsize=7)
    plt.colorbar(im2, ax=axes[1], label='Co-selection count', shrink=0.8)
    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/4_coselection_L{L}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 5: Expert Rank Stability ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    top10 = np.argsort(-total)[:10]
    cmap = plt.cm.tab10
    for rank_i, e in enumerate(top10):
        ranks_across = np.array([
            int(np.where(np.argsort(-per_batch[i]) == e)[0][0]) + 1
            for i in range(num_batches)
        ])
        ax1.plot(range(num_batches), ranks_across, lw=0.9, alpha=0.8,
                 color=cmap(rank_i), label=f'e{e}')
    ax1.set_xlabel('Batch index', fontsize=11)
    ax1.set_ylabel('Rank (1=most popular)', fontsize=11)
    ax1.set_title(f'Top-10 Expert Rank — Layer {L}', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.legend(fontsize=7, ncol=2, loc='lower right')
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(30, 0)
    expert_means = per_batch.mean(axis=0).astype(np.float64)
    expert_stds = per_batch.std(axis=0).astype(np.float64)
    expert_cv = np.where(expert_means > 0, expert_stds / expert_means, 0)
    sc = ax2.scatter(expert_means, expert_cv, s=12, alpha=0.6, c=total,
                     cmap='viridis', edgecolors='none')
    plt.colorbar(sc, ax=ax2, label='Total token-slots', shrink=0.8)
    top5 = np.argsort(-total)[:5]
    for e in top5:
        ax2.annotate(f'e{e}', (expert_means[e], expert_cv[e]),
                     fontsize=7, fontweight='bold', color='red',
                     xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('Mean batch load', fontsize=11)
    ax2.set_ylabel('CV (std/mean)', fontsize=11)
    ax2.set_title(f'Expert Load Stability — Layer {L}', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/5_rank_stability_L{L}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 6: Summary Comparison ──
    fig, axes_s = plt.subplots(1, 2, figsize=(12, 5))
    strategies = ['Static', 'Optimal']
    comp_imbs = [imb(sl_comp), imb(ol_comp)]
    comm_imbs = [imb(sl_comm), imb(ol_comm)]
    x = np.arange(2)
    w = 0.3
    axes_s[0].bar(x - w/2, comp_imbs, w, label='Compute', color='#5B8BD0', alpha=0.85)
    axes_s[0].bar(x + w/2, comm_imbs, w, label='Comm', color='#E8A838', alpha=0.85)
    axes_s[0].axhline(1.0, color='black', ls='--', lw=0.8, alpha=0.5)
    axes_s[0].set_xticks(x); axes_s[0].set_xticklabels(strategies, fontsize=10)
    axes_s[0].set_ylabel('Imbalance (max/mean)', fontsize=11)
    axes_s[0].set_title(f'Imbalance — Layer {L}', fontsize=13, fontweight='bold')
    axes_s[0].legend(fontsize=9)
    for i, (cv, ccv) in enumerate(zip(comp_imbs, comm_imbs)):
        axes_s[0].text(i - w/2, cv + 0.005, f'{cv:.3f}x', ha='center', fontsize=9, fontweight='bold')
        axes_s[0].text(i + w/2, ccv + 0.005, f'{ccv:.3f}x', ha='center', fontsize=9, fontweight='bold')
    comp_maxs = [sl_comp.max(), ol_comp.max()]
    comm_maxs = [sl_comm.max(), ol_comm.max()]
    axes_s[1].bar(x - w/2, [v/1000 for v in comp_maxs], w, label='Compute (K)', color='#5B8BD0', alpha=0.85)
    axes_s[1].bar(x + w/2, [v/1000 for v in comm_maxs], w, label='Comm (K)', color='#E8A838', alpha=0.85)
    axes_s[1].set_xticks(x); axes_s[1].set_xticklabels(strategies, fontsize=10)
    axes_s[1].set_ylabel('Max load (thousands)', fontsize=11)
    axes_s[1].set_title(f'Max Device Load — Layer {L}', fontsize=13, fontweight='bold')
    axes_s[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/6_summary_L{L}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Generated 6 detail plots for layer {L} in {PLOT_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import os
    os.makedirs(PLOT_DIR, exist_ok=True)

    layers_sorted = sorted(LAYER_SHARD_MAP.keys())
    num_layers = len(layers_sorted)

    print(f"{'='*70}")
    print(f"  DeepSeek-R1 EPLB Routing Analysis — {num_layers} MoE Layers")
    print(f"  {NUM_EXPERTS} experts, top-{TOP_K}, EP_SIZE={EP_SIZE}")
    print(f"  Sweep: {SWEEP_BATCHES}x{SWEEP_BATCH_SIZE} tokens/layer")
    print(f"  Detail layer: {DETAIL_LAYER} ({DETAIL_BATCHES}x{DETAIL_BATCH_SIZE} tokens)")
    print(f"{'='*70}\n")

    # ── Phase 1: Cross-layer sweep ──
    results = []
    t0 = time.time()
    for idx, layer in enumerate(layers_sorted):
        shard = LAYER_SHARD_MAP[layer]
        t1 = time.time()
        r = analyze_layer(layer, shard)
        dt = time.time() - t1
        results.append(r)
        print(f"  [{idx+1:2d}/{num_layers}] Layer {layer:2d}: "
              f"comp {r['static_compute_imb']:.3f}->{r['optimal_compute_imb']:.3f}x "
              f"({r['compute_reduction_pct']:+.1f}%)  "
              f"comm {r['static_comm_imb']:.3f}->{r['optimal_comm_imb']:.3f}x "
              f"({r['comm_reduction_pct']:+.1f}%)  "
              f"CV={r['expert_cv']:.3f}  fan={r['fan_out']:.2f}x  "
              f"[{dt:.1f}s]")

    sweep_time = time.time() - t0
    print(f"\n  Sweep: {sweep_time:.0f}s total, {sweep_time/num_layers:.1f}s/layer")

    # ── Aggregate stats ──
    comp_reds = [r['compute_reduction_pct'] for r in results]
    comm_reds = [r['comm_reduction_pct'] for r in results]
    s_comp = [r['static_compute_imb'] for r in results]
    o_comp = [r['optimal_compute_imb'] for r in results]
    s_comm = [r['static_comm_imb'] for r in results]
    o_comm = [r['optimal_comm_imb'] for r in results]
    cvs = [r['expert_cv'] for r in results]

    print(f"\n{'='*70}")
    print(f"  Cross-Layer Summary ({num_layers} MoE layers)")
    print(f"{'='*70}")
    print(f"  {'Metric':>30} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    print(f"  {'-'*65}")
    for name, vals in [
        ('Static compute imb', s_comp),
        ('Optimal compute imb', o_comp),
        ('Compute reduction %', comp_reds),
        ('Static comm imb', s_comm),
        ('Optimal comm imb', o_comm),
        ('Comm reduction %', comm_reds),
        ('Expert load CV', cvs),
        ('Fan-out', [r['fan_out'] for r in results]),
    ]:
        v = np.array(vals)
        print(f"  {name:>30} {v.mean():>8.3f} {v.std():>8.3f} "
              f"{v.min():>8.3f} {v.max():>8.3f}")

    # ── Phase 2: Cross-layer summary plot ──
    plot_cross_layer_summary(results, PLOT_DIR)

    # ── Phase 3: Detail analysis for one layer ──
    detail_analysis(DETAIL_LAYER, LAYER_SHARD_MAP[DETAIL_LAYER], PLOT_DIR)

    # ── Save results JSON ──
    out = {
        'config': {
            'E': NUM_EXPERTS, 'K': TOP_K, 'EP': EP_SIZE,
            'sweep_batches': SWEEP_BATCHES, 'sweep_batch_size': SWEEP_BATCH_SIZE,
        },
        'per_layer': results,
        'aggregate': {
            'mean_compute_reduction_pct': float(np.mean(comp_reds)),
            'mean_comm_reduction_pct': float(np.mean(comm_reds)),
            'mean_static_compute_imb': float(np.mean(s_comp)),
            'mean_optimal_compute_imb': float(np.mean(o_comp)),
            'mean_expert_cv': float(np.mean(cvs)),
        },
    }
    with open('/mnt/pd/eplb_routing_analysis.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to /mnt/pd/eplb_routing_analysis.json")


if __name__ == '__main__':
    main()
