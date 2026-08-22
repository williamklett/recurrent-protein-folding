#!/usr/bin/env python3
"""Characterize Contact Convergence and Biochemical Signal.

Computes contact map accuracy (vs Native) and convergence (vs Final)
using existing layer-wise distogram probes. Includes residual velocity analysis.
"""

import argparse
import csv
import json
import os
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForProteinFolding
from scipy.stats import spearmanr, pearsonr

from recurrent_folding.esmfold_probes import PairDistogramBottleneck

# Constants
CONTACT_THRESHOLD = 8.0
SEQ_SEP = 6
PROBE_DIR = Path("runs/general_belief_replay_benchmark20_cw30")
RUNS_DIR = Path("runs")

# Global probe cache
PROBES = {}

def get_distogram_bin_centers(device):
    start, end, bins = 2.3125, 21.6875, 64
    width = (end - start) / bins
    return torch.linspace(start + width/2, end - width/2, bins, device=device)

def get_probe(layer, device):
    if layer in PROBES: return PROBES[layer]
    path = PROBE_DIR / f"layer_{layer:02d}" / "probe.pt"
    if not path.exists(): return None
    probe = PairDistogramBottleneck(pair_dim=128, bins=64, hidden_dim=256).to(device)
    probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    probe.eval()
    for p in probe.parameters(): p.requires_grad = False
    PROBES[layer] = probe
    return probe

def compute_precision_at_k(probs, gt_contacts, k):
    idx = np.argsort(probs)[::-1]
    top_k = idx[:k]
    return np.mean(gt_contacts[top_k])

def characterize_protein(name, run_dir, model, tokenizer, device):
    cache_path = run_dir / "esmfold_activations.pt"
    if not cache_path.exists(): return None
    
    cache = torch.load(cache_path, map_location="cpu")
    z_all = cache["z"].squeeze(1).to(device)
    L = z_all.shape[1]
    
    gt_path = run_dir / "gt_ca_distances.npz"
    if not gt_path.exists(): return None
    gt_dists_all = np.load(gt_path)["distances"]
    
    iu = np.triu_indices(L, k=SEQ_SEP+1)
    gt_dists_flat = gt_dists_all[iu]
    gt_contacts_flat = (gt_dists_flat < CONTACT_THRESHOLD).astype(float)
    
    centers = get_distogram_bin_centers(device)
    bin_idx = (centers < CONTACT_THRESHOLD)

    p_contact_layers = {}
    residuals = {}
    velocities_total = []
    velocities_resid = []
    
    i_idx_t = torch.arange(L, device=device).repeat_interleave(L)
    j_idx_t = torch.arange(L, device=device).repeat(L)

    for t in range(1, 47):
        probe = get_probe(t, device)
        if probe is None: continue
        
        with torch.no_grad():
            z_t = z_all[t-1].float()
            logits, reconstructed = probe(z_t.reshape(-1, 128), i_idx_t, j_idx_t, L)
            reconstructed = reconstructed.reshape(L, L, 128)
            residuals[t] = (z_t - reconstructed).cpu()
            
            logits = logits.reshape(L, L, 64)
            logits = 0.5 * (logits + logits.transpose(0, 1))
            probs = F.softmax(logits, dim=-1)
            p_contact = probs[:, :, bin_idx].sum(dim=-1).cpu().numpy()
            p_contact_layers[t] = p_contact

    # Calculate Velocities
    for t in sorted(residuals.keys()):
        if (t+1) in residuals:
            delta_z = z_all[t].cpu() - z_all[t-1].cpu()
            velocities_total.append(float(delta_z.norm()))
            delta_r = residuals[t+1] - residuals[t]
            velocities_resid.append(float(delta_r.norm()))

    if not p_contact_layers or 17 not in p_contact_layers: return None
    
    final_layer = max(p_contact_layers.keys())
    p_final_flat = p_contact_layers[final_layer][iu]
    
    with torch.no_grad():
        final_probe = get_probe(final_layer, device)
        z_final = z_all[final_layer-1].float()
        logits_final = final_probe.encode(z_final.reshape(-1, 128), i_idx_t, j_idx_t, L).reshape(L, L, 64)
        logits_final = 0.5 * (logits_final + logits_final.transpose(0, 1))
        probs_final = F.softmax(logits_final, dim=-1)
        e_dist_final = (probs_final * centers).sum(dim=-1).cpu().numpy()[iu]
        final_mae = np.mean(np.abs(e_dist_final - gt_dists_flat))

    results = []
    from sklearn.metrics import roc_auc_score
    avg_early_vel_resid = np.mean(velocities_resid[:16]) if len(velocities_resid) >= 16 else 0

    for t in sorted(p_contact_layers.keys()):
        p_flat = p_contact_layers[t][iu]
        auroc = roc_auc_score(gt_contacts_flat, p_flat)
        corr_final = np.corrcoef(p_flat, p_final_flat)[0, 1]
        v_idx = t - 1
        v_total = velocities_total[v_idx] if v_idx < len(velocities_total) else 0
        v_resid = velocities_resid[v_idx] if v_idx < len(velocities_resid) else 0

        results.append({
            "protein": name,
            "layer": t,
            "auroc": auroc,
            "corr_vs_final": corr_final,
            "final_mae": final_mae,
            "velocity_total": v_total,
            "velocity_resid": v_resid,
            "avg_early_velocity": avg_early_vel_resid
        })
    
    print(f"  {name} | MAE {final_mae:.2f}Å | AUROC@L17 {results[16]['auroc']:.3f} | ResidVel {avg_early_vel_resid:.1f}", flush=True)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", type=str, default="runs/contact_convergence.csv")
    parser.add_argument("--out-png", type=str, default="runs/contact_convergence.png")
    parser.add_argument("--out-vel-png", type=str, default="runs/velocity_residual_grid.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to(device)
    model.eval()

    all_results = []
    print("Loading cached activation paths...")
    path_file = Path("all_activation_paths.txt")
    if path_file.exists():
        with path_file.open() as f:
            activation_files = [Path(line.strip()) for line in f if line.strip()]
    else:
        activation_files = list(RUNS_DIR.glob("**/esmfold_activations.pt"))
        
    protein_dirs = [f.parent for f in activation_files]
    print(f"Found {len(protein_dirs)} cached proteins.")
    
    out_csv = Path(args.out_csv)
    abs_runs_dir = RUNS_DIR.resolve()

    with out_csv.open("w", newline="") as f:
        writer = None
        for run_dir in protein_dirs:
            run_dir = run_dir.resolve()
            name = run_dir.name.replace("benchmark_", "").replace("hard_synthetic_miss_", "miss_").replace("synthetic_design_", "design_")
            if "analysis" in name or name == "7ACN": continue
            
            rel_path = run_dir.relative_to(abs_runs_dir) if run_dir.is_relative_to(abs_runs_dir) else run_dir.name
            print(f"Processing {name} ({rel_path})...", flush=True)
            
            try:
                res = characterize_protein(name, run_dir, model, tokenizer, device)
                if res:
                    all_results.extend(res)
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=res[0].keys())
                        writer.writeheader()
                    writer.writerows(res)
                    f.flush()
            except Exception as e:
                print(f"Error processing {name}: {e}")
            torch.cuda.empty_cache()

    if not all_results: return

    import matplotlib.pyplot as plt
    prot_metrics = {r['protein']: {
        'mae': r['final_mae'],
        'auroc_17': next(x['auroc'] for x in all_results if x['protein'] == r['protein'] and x['layer'] == 17),
        'early_vel': r['avg_early_velocity']
    } for r in all_results}
    
    proteins_sorted = sorted(prot_metrics.keys(), key=lambda n: prot_metrics[n]['mae'])
    n_prots = len(proteins_sorted)
    n_cols = 5
    n_rows = math.ceil(n_prots / n_cols)
    
    fig_v, axes_v = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.2), sharex=True, sharey=False)
    axes_v_flat = axes_v.flatten()
    max_v_resid = max([r['velocity_resid'] for r in all_results])

    for i, name in enumerate(proteins_sorted):
        ax = axes_v_flat[i]
        p_res = sorted([r for r in all_results if r['protein'] == name], key=lambda x: x['layer'])
        layers = [r['layer'] for r in p_res]
        v_resids = [r['velocity_resid'] for r in p_res]
        mae = prot_metrics[name]['mae']
        color = "#2e7d32" if mae < 2.0 else ("#c62828" if mae > 10.0 else "#ef6c00")
        
        ax.plot(layers, v_resids, color='#7c3aed', marker='o', markersize=2, lw=1.0)
        ax.set_ylim(0, max_v_resid * 1.1)
        ax.set_title(f"{i+1:d}. {name}", loc='left', fontsize=10, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)
        ax.axvspan(0, 15, color='#fef3c7', alpha=0.15)
        ax.text(0.95, 0.90, f"MAE: {mae:.1f}Å", transform=ax.transAxes, ha='right', fontsize=8, fontweight='bold')
        ax.grid(alpha=0.2, ls='--')

    for j in range(i + 1, len(axes_v_flat)): axes_v_flat[j].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(args.out_vel_png, dpi=180)

    maes = [p['mae'] for p in prot_metrics.values()]
    aurocs_17 = [p['auroc_17'] for p in prot_metrics.values()]
    vels = [p['early_vel'] for p in prot_metrics.values()]
    c_vel_mae = pearsonr(vels, maes)[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.scatter(aurocs_17, maes, c='#1d4ed8', alpha=0.6)
    ax1.set_title(f"AUROC@L17 vs Final MAE")
    ax1.set_xlabel("AUROC at Layer 17")
    ax1.set_ylabel("MAE (Å)")
    ax1.grid(alpha=0.3)
    ax2.scatter(vels, maes, c='#c62828', alpha=0.6)
    ax2.set_title(f"Early Resid Velocity vs Final MAE (r={c_vel_mae:.2f})")
    ax2.set_xlabel("Avg ||Δ(z-ẑ)|| (L1-17)")
    ax2.set_ylabel("MAE (Å)")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("runs/velocity_analysis_residual.png", dpi=150)

if __name__ == "__main__":
    main()
