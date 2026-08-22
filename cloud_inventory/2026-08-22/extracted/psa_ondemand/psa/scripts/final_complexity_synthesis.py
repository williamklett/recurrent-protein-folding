
import os
import numpy as np
import torch
import csv
import json
from pathlib import Path
from scipy.stats import pearsonr

ROOT = Path("/Users/williamklett/Desktop/Natural Bio Encoder")
MANIFEST = ROOT / "data" / "manifest.csv"
PHYS_DIR = ROOT / "data" / "physics"
DIST_DIR = ROOT / "data" / "distograms"
OUT_PATH = ROOT / "runs/final_complexity_synthesis.json"

def get_geom_signatures(dist_matrix):
    L = dist_matrix.shape[0]
    D2 = dist_matrix.astype(np.float32) ** 2
    row_means = D2.mean(axis=1, keepdims=True)
    col_means = D2.mean(axis=0, keepdims=True)
    total_mean = D2.mean()
    G = -0.5 * (D2 - row_means - col_means + total_mean)
    eigvals = np.linalg.eigvalsh(G)
    
    # Sort by magnitude
    mags = np.sort(np.abs(eigvals))[::-1]
    cum_var = np.cumsum(mags) / np.sum(mags)
    eff_rank = int(np.searchsorted(cum_var, 0.90) + 1)
    
    pos_energy = np.sum(eigvals[eigvals > 0])
    neg_energy = np.sum(np.abs(eigvals[eigvals < 0]))
    neg_frac = float(neg_energy / (pos_energy + neg_energy)) if (pos_energy + neg_energy) > 0 else 0.0
    
    return eff_rank, neg_frac

def calculate_complexity(gt_dist, contact_mask):
    seq_sep = np.abs(np.where(contact_mask)[0] - np.where(contact_mask)[1])
    aco = float(np.mean(seq_sep)) if len(seq_sep) > 0 else 0.0
    
    # Absolute Number of Crossings
    contacts = np.where(contact_mask)
    pairs = list(zip(contacts[0], contacts[1]))
    n = len(pairs)
    crossings = 0
    if n >= 2:
        for idx1 in range(n):
            i, j = pairs[idx1]
            for idx2 in range(idx1 + 1, n):
                k, l = pairs[idx2]
                if i > k: i, j, k, l = k, l, i, j
                if i < k < j < l: crossings += 1
    return aco, crossings

def main():
    with open(MANIFEST, 'r') as f:
        proteins = list(csv.DictReader(f))

    results = []
    for p in proteins:
        pdb_id = p['pdb_id'].lower()
        if not (PHYS_DIR / f"{pdb_id}.npz").exists() or not (DIST_DIR / f"{pdb_id}.pt").exists(): continue
            
        phys_data = np.load(PHYS_DIR / f"{pdb_id}.npz")
        gt_dist = phys_data['distances'] if 'distances' in phys_data else phys_data['d_ca']
        contact_mask = np.triu(gt_dist < 8.0, k=3)
        
        aco, abs_cross = calculate_complexity(gt_dist, contact_mask)
        
        try:
            d_data = torch.load(DIST_DIR / f"{pdb_id}.pt", map_location='cpu', weights_only=False)
            if isinstance(d_data, dict):
                d_traj = d_data['pred_distance']
            else:
                d_traj = d_data
                
            if isinstance(d_traj, torch.Tensor):
                if d_traj.ndim == 4:
                    d_block25 = d_traj[25, 0].float().numpy()
                elif d_traj.ndim == 3:
                    d_block25 = d_traj[25].float().numpy()
                else:
                    d_block25 = d_traj.float().numpy()
            else:
                d_block25 = d_traj[25] # fallback
            
            eff_rank, neg_frac = get_geom_signatures(d_block25)
            
            results.append({
                "pdb_id": pdb_id,
                "aco": aco,
                "abs_cross": abs_cross,
                "eff_rank": eff_rank,
                "neg_frac": neg_frac,
                "L": int(p['L']),
                "fold_class": p['fold_class']
            })
        except: continue

    if not results: return

    correlations = {}
    for comp in ["aco", "abs_cross", "L"]:
        for geom in ["eff_rank", "neg_frac"]:
            r, p = pearsonr([res[comp] for res in results], [res[geom] for res in results])
            correlations[f"{comp}_vs_{geom}"] = {"r": float(r), "p": float(p)}
    
    output = {"summary": correlations, "results": results}
    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Final Geometric Signatures of Complexity:")
    for k, v in correlations.items():
        print(f"  {k:25}: r={v['r']:.4f}, p={v['p']:.4e}")

if __name__ == "__main__":
    main()
