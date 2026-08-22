
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
OUT_PATH = ROOT / "runs/complexity_suite_analysis.json"

def calculate_circuit_topology(contact_mask):
    """
    Calculate the fraction of crossed contacts.
    A crossing occurs between contact (i,j) and (k,l) if i < k < j < l.
    """
    contacts = np.where(contact_mask)
    pairs = list(zip(contacts[0], contacts[1]))
    n = len(pairs)
    if n < 2:
        return 0.0
    
    crossings = 0
    total_non_nested_pairs = 0
    
    for idx1 in range(n):
        i, j = pairs[idx1]
        for idx2 in range(idx1 + 1, n):
            k, l = pairs[idx2]
            
            # Ensure order for comparison
            if i > k:
                i, j, k, l = k, l, i, j
            
            # Check for crossing: i < k < j < l
            if i < k < j < l:
                crossings += 1
                
    return float(crossings / (n * (n - 1) / 2))

def calculate_algebraic_connectivity(contact_mask):
    """
    Calculate the second smallest eigenvalue of the Laplacian matrix.
    """
    L_size = contact_mask.shape[0]
    adj = contact_mask.astype(float)
    # Ensure symmetry
    adj = np.maximum(adj, adj.T)
    
    degree = np.sum(adj, axis=1)
    D = np.diag(degree)
    Laplacian = D - adj
    
    try:
        eigvals = np.linalg.eigvalsh(Laplacian)
        # Smallest is 0, second smallest is algebraic connectivity
        return float(eigvals[1]) if len(eigvals) > 1 else 0.0
    except:
        return 0.0

def calculate_aco(dist_matrix, threshold=8.0):
    L = dist_matrix.shape[0]
    mask = np.triu(dist_matrix < threshold, k=1)
    contacts = np.where(mask)
    n_contacts = len(contacts[0])
    if n_contacts == 0:
        return 0.0
    seq_sep = np.abs(contacts[0] - contacts[1])
    return float(np.mean(seq_sep))

def get_neg_signature_fraction(dist_matrix):
    L = dist_matrix.shape[0]
    D2 = dist_matrix.astype(np.float32) ** 2
    row_means = D2.mean(axis=1, keepdims=True)
    col_means = D2.mean(axis=0, keepdims=True)
    total_mean = D2.mean()
    G = -0.5 * (D2 - row_means - col_means + total_mean)
    eigvals = np.linalg.eigvalsh(G)
    pos_energy = np.sum(eigvals[eigvals > 0])
    neg_energy = np.sum(np.abs(eigvals[eigvals < 0]))
    total = pos_energy + neg_energy
    return float(neg_energy / total) if total > 0 else 0.0

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
        gt_dist = phys_data['distances'] if 'distances' in phys_data else phys_data['d_ca']
        gt_contacts = phys_data['contacts'] if 'contacts' in phys_data else (gt_dist < 8.0)
        
        # Binary contact mask for complexity (off-diagonal only)
        L = gt_dist.shape[0]
        contact_mask = np.triu(gt_contacts, k=3) # Ignore local i, i+1, i+2
        
        aco = calculate_aco(gt_dist)
        cross_frac = calculate_circuit_topology(contact_mask)
        alg_conn = calculate_algebraic_connectivity(contact_mask)
        
        try:
            d_traj = torch.load(dist_file, map_location='cpu', weights_only=False)
            if isinstance(d_traj, dict):
                d_block25 = d_traj['pred_distance'][25].float().numpy()
            elif isinstance(d_traj, torch.Tensor):
                d_block25 = d_traj[25, 0].float().numpy() if d_traj.ndim == 4 else d_traj[25].float().numpy()
            else:
                d_block25 = d_traj[25].float().numpy()
                
            neg_frac = get_neg_signature_fraction(d_block25)
            
            results.append({
                "pdb_id": pdb_id,
                "aco": aco,
                "cross_frac": cross_frac,
                "algebraic_connectivity": alg_conn,
                "neg_frac": neg_frac,
                "L": int(p['L']),
                "fold_class": p['fold_class']
            })
        except: continue

    if not results: return

    metrics = ["aco", "cross_frac", "algebraic_connectivity", "L"]
    summary = {}
    for m in metrics:
        r, p = pearsonr([res[m] for res in results], [res["neg_frac"] for res in results])
        summary[f"{m}_vs_neg_frac"] = {"r": float(r), "p": float(p)}
        
    output = {"summary": summary, "results": results}
    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Correlations with Negative Signature Fraction:")
    for k, v in summary.items():
        print(f"  {k:25}: r={v['r']:.4f}, p={v['p']:.4e}")

if __name__ == "__main__":
    main()
