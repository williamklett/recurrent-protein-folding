
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
OUT_PATH = ROOT / "runs/complexity_relaxation_correlation_refined.json"

def calculate_co_metrics(dist_matrix, threshold=8.0):
    L = dist_matrix.shape[0]
    mask = np.triu(dist_matrix < threshold, k=1)
    contacts = np.where(mask)
    n_contacts = len(contacts[0])
    if n_contacts == 0:
        return 0.0, 0.0
    
    seq_sep = np.abs(contacts[0] - contacts[1])
    # Absolute Contact Order (ACO)
    aco = np.mean(seq_sep)
    # Relative Contact Order (RCO)
    rco = aco / L
    return float(aco), float(rco)

def get_eigen_metrics(dist_matrix):
    L = dist_matrix.shape[0]
    D2 = dist_matrix.astype(np.float32) ** 2
    row_means = D2.mean(axis=1, keepdims=True)
    col_means = D2.mean(axis=0, keepdims=True)
    total_mean = D2.mean()
    G = -0.5 * (D2 - row_means - col_means + total_mean)
    
    eigvals = np.linalg.eigvalsh(G)
    
    pos_energy = np.sum(eigvals[eigvals > 0])
    neg_energy = np.sum(np.abs(eigvals[eigvals < 0]))
    total_energy = pos_energy + neg_energy
    
    if total_energy == 0:
        return 0.0, 0.0
    
    return float(neg_energy / total_energy), float(neg_energy)

def main():
    with open(MANIFEST, 'r') as f:
        reader = csv.DictReader(f)
        proteins = list(reader)

    results = []
    for p in proteins:
        pdb_id = p['pdb_id'].lower()
        phys_file = PHYS_DIR / f"{pdb_id}.npz"
        dist_file = DIST_DIR / f"{pdb_id}.pt"
        
        if not phys_file.exists() or not dist_file.exists():
            continue
            
        phys_data = np.load(phys_file)
        gt_dist = None
        for k in ['d_ca', 'distances']:
            if k in phys_data:
                gt_dist = phys_data[k]
                break
        if gt_dist is None:
            for k in phys_data.files:
                if phys_data[k].ndim == 2:
                    gt_dist = phys_data[k]
                    break
        
        if gt_dist is None: continue
            
        aco, rco = calculate_co_metrics(gt_dist)
        
        try:
            d_traj = torch.load(dist_file, map_location='cpu', weights_only=False)
            if isinstance(d_traj, dict):
                if 'pred_distance' in d_traj:
                    d_block25 = d_traj['pred_distance'][25].float().numpy()
                else:
                    # Fallback or skip
                    continue
            elif isinstance(d_traj, torch.Tensor):
                d_block25 = d_traj[25, 0].float().numpy() if d_traj.ndim == 4 else d_traj[25].float().numpy()
            else:
                d_block25 = d_traj[25].float().numpy()
                
            neg_frac, neg_abs = get_eigen_metrics(d_block25)
            
            results.append({
                "pdb_id": pdb_id,
                "aco": aco,
                "rco": rco,
                "neg_frac": neg_frac,
                "neg_abs": neg_abs,
                "L": int(p['L']),
                "fold_class": p['fold_class']
            })
        except Exception: continue

    if not results: return

    # Analyze correlations
    aco_vals = [r['aco'] for r in results]
    rco_vals = [r['rco'] for r in results]
    neg_frac_vals = [r['neg_frac'] for r in results]
    neg_abs_vals = [r['neg_abs'] for r in results]
    L_vals = [r['L'] for r in results]
    
    correlations = {
        "rco_vs_neg_frac": pearsonr(rco_vals, neg_frac_vals),
        "aco_vs_neg_frac": pearsonr(aco_vals, neg_frac_vals),
        "aco_vs_neg_abs": pearsonr(aco_vals, neg_abs_vals),
        "L_vs_neg_frac": pearsonr(L_vals, neg_frac_vals),
        "L_vs_neg_abs": pearsonr(L_vals, neg_abs_vals),
        "aco_vs_L": pearsonr(aco_vals, L_vals),
        "rco_vs_L": pearsonr(rco_vals, L_vals)
    }
    
    # Format correlations for JSON
    serializable_corr = {k: {"r": float(v[0]), "p": float(v[1])} for k, v in correlations.items()}
    
    output = {
        "summary": serializable_corr,
        "results": results
    }
    
    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Correlations:")
    for k, v in serializable_corr.items():
        print(f"  {k:20}: r={v['r']:.4f}, p={v['p']:.4e}")

if __name__ == "__main__":
    main()
