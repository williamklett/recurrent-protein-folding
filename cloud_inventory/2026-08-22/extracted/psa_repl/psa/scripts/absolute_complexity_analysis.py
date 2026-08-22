
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
OUT_PATH = ROOT / "runs/absolute_complexity_analysis.json"

def calculate_absolute_crossings(contact_mask):
    contacts = np.where(contact_mask)
    pairs = list(zip(contacts[0], contacts[1]))
    n = len(pairs)
    if n < 2: return 0
    crossings = 0
    for idx1 in range(n):
        i, j = pairs[idx1]
        for idx2 in range(idx1 + 1, n):
            k, l = pairs[idx2]
            if i > k: i, j, k, l = k, l, i, j
            if i < k < j < l: crossings += 1
    return crossings

def get_absolute_neg_energy(dist_matrix):
    D2 = dist_matrix.astype(np.float32) ** 2
    row_means = D2.mean(axis=1, keepdims=True)
    col_means = D2.mean(axis=0, keepdims=True)
    total_mean = D2.mean()
    G = -0.5 * (D2 - row_means - col_means + total_mean)
    eigvals = np.linalg.eigvalsh(G)
    neg_energy = np.sum(np.abs(eigvals[eigvals < 0]))
    return float(neg_energy)

def main():
    with open(MANIFEST, 'r') as f:
        reader = csv.DictReader(f)
        proteins = list(reader)

    results = []
    for p in proteins:
        pdb_id = p['pdb_id'].lower()
        phys_file = PHYS_DIR / f"{pdb_id}.npz"
        dist_file = DIST_DIR / f"{pdb_id}.pt"
        if not phys_file.exists() or not dist_file.exists(): continue
            
        phys_data = np.load(phys_file)
        gt_dist = phys_data['distances'] if 'distances' in phys_data else phys_data['d_ca']
        gt_contacts = phys_data['contacts'] if 'contacts' in phys_data else (gt_dist < 8.0)
        
        L = gt_dist.shape[0]
        contact_mask = np.triu(gt_contacts, k=3)
        n_contacts = int(np.sum(contact_mask))
        abs_crossings = calculate_absolute_crossings(contact_mask)
        
        try:
            data = torch.load(dist_file, map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                d_traj = data['pred_distance']
            else:
                d_traj = data
                
            if isinstance(d_traj, torch.Tensor):
                d_block25 = d_traj[25, 0].float().numpy() if d_traj.ndim == 4 else d_traj[25].float().numpy()
            else:
                d_block25 = d_traj[25].float().numpy()
            
            abs_neg_energy = get_absolute_neg_energy(d_block25)
            
            results.append({
                "pdb_id": pdb_id,
                "abs_crossings": abs_crossings,
                "n_contacts": n_contacts,
                "abs_neg_energy": abs_neg_energy,
                "L": int(p['L']),
                "fold_class": p['fold_class']
            })
        except: continue

    if not results: return

    metrics = ["abs_crossings", "n_contacts", "L"]
    summary = {}
    for m in metrics:
        r, p = pearsonr([res[m] for res in results], [res["abs_neg_energy"] for res in results])
        summary[f"{m}_vs_abs_neg_energy"] = {"r": float(r), "p": float(p)}
    
    # Partial correlation idea: crossings vs energy controlling for L
    # We can check r(abs_crossings, abs_neg_energy) vs r(L, abs_neg_energy)
    
    output = {"summary": summary, "results": results}
    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Correlations with Absolute Negative Energy:")
    for k, v in summary.items():
        print(f"  {k:30}: r={v['r']:.4f}, p={v['p']:.4e}")

if __name__ == "__main__":
    main()
