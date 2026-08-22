
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
OUT_PATH = ROOT / "runs" / "complexity_relaxation_correlation.json"

def calculate_contact_order(dist_matrix, threshold=8.0):
    L = dist_matrix.shape[0]
    # Create mask for off-diagonal contacts
    mask = np.triu(dist_matrix < threshold, k=1)
    contacts = np.where(mask)
    n_contacts = len(contacts[0])
    if n_contacts == 0:
        return 0.0
    
    seq_sep = np.abs(contacts[0] - contacts[1])
    # Relative Contact Order
    co = np.sum(seq_sep) / (L * n_contacts)
    return float(co)

def get_negative_signature_fraction(dist_matrix):
    L = dist_matrix.shape[0]
    # Double centering to get Gram matrix
    # G = -0.5 * J * D^2 * J
    D2 = dist_matrix ** 2
    row_means = D2.mean(axis=1, keepdims=True)
    col_means = D2.mean(axis=0, keepdims=True)
    total_mean = D2.mean()
    G = -0.5 * (D2 - row_means - col_means + total_mean)
    
    eigvals = np.linalg.eigvalsh(G)
    
    pos_energy = np.sum(eigvals[eigvals > 0])
    neg_energy = np.sum(np.abs(eigvals[eigvals < 0]))
    
    if (pos_energy + neg_energy) == 0:
        return 0.0
    
    return float(neg_energy / (pos_energy + neg_energy))

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
            
        # Load ground truth physics
        phys_data = np.load(phys_file)
        # Assuming distance matrix is stored as 'dist' or similar
        # Based on physics.py, it's likely 'd_ca'
        if 'd_ca' in phys_data:
            gt_dist = phys_data['d_ca']
        elif 'distances' in phys_data:
            gt_dist = phys_data['distances']
        else:
            # Try to find any 2D array
            gt_dist = None
            for k in phys_data.files:
                if phys_data[k].ndim == 2:
                    gt_dist = phys_data[k]
                    break
        
        if gt_dist is None:
            continue
            
        co = calculate_contact_order(gt_dist)
        
        # Load predicted distogram trajectory
        # Structure is (blocks, L, L) in pred_distance
        try:
            d_traj = torch.load(dist_file, map_location='cpu', weights_only=False)
            
            if isinstance(d_traj, dict) and 'pred_distance' in d_traj:
                d_block25 = d_traj['pred_distance'][25].numpy().astype(np.float32)
            elif isinstance(d_traj, torch.Tensor):
                # Shape (48, L, L) or (48, 1, L, L)
                if d_traj.ndim == 4:
                    d_block25 = d_traj[25, 0].numpy().astype(np.float32)
                else:
                    d_block25 = d_traj[25].numpy().astype(np.float32)
            else:
                # Fallback for other dict structures
                d_block25 = d_traj[25].numpy().astype(np.float32)
                
            neg_frac = get_negative_signature_fraction(d_block25)
            
            results.append({
                "pdb_id": pdb_id,
                "contact_order": co,
                "negative_signature_fraction": neg_frac,
                "L": int(p['L']),
                "fold_class": p['fold_class']
            })
            print(f"Processed {pdb_id}: CO={co:.4f}, NegFrac={neg_frac:.4f}")
            
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")
            continue

    if not results:
        print("No results collected.")
        return

    co_vals = [r['contact_order'] for r in results]
    neg_vals = [r['negative_signature_fraction'] for r in results]
    
    r_val, p_val = pearsonr(co_vals, neg_vals)
    
    summary = {
        "pearson_r": float(r_val),
        "p_value": float(p_val),
        "n_proteins": len(results),
        "results": results
    }
    
    with open(OUT_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nAnalysis complete. Pearson r = {r_val:.4f} (p = {p_val:.4e})")

if __name__ == "__main__":
    main()
