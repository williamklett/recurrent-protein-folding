
import os
import numpy as np
import torch
import json
from pathlib import Path

ROOT = Path("/Users/williamklett/Desktop/Natural Bio Encoder")
DIST_DIR = ROOT / "data/distograms"
OUT_PATH = ROOT / "runs/signature_shedding_analysis.json"

def get_spectral_components(dist_matrix):
    L = dist_matrix.shape[0]
    D2 = dist_matrix.astype(np.float32) ** 2
    row_means = D2.mean(axis=1, keepdims=True)
    col_means = D2.mean(axis=0, keepdims=True)
    total_mean = D2.mean()
    G = -0.5 * (D2 - row_means - col_means + total_mean)
    eigvals = np.linalg.eigvalsh(G)
    
    # Components
    neg_energy = np.sum(np.abs(eigvals[eigvals < 0]))
    
    pos_eigvals = eigvals[eigvals > 0]
    # Sort descending
    pos_eigvals = np.sort(pos_eigvals)[::-1]
    
    rank13_energy = np.sum(pos_eigvals[:3])
    high_rank_pos_energy = np.sum(pos_eigvals[3:])
    
    return float(neg_energy), float(high_rank_pos_energy), float(rank13_energy)

def main():
    # We'll sample a few representative proteins to see the trajectory
    # 1TIT (Mixed), 1LMB (Alpha), 1LYZ (Alpha/Beta)
    test_ids = ["1tit", "1lmb", "1lyz"]
    
    all_trajectories = {}
    
    for pdb_id in test_ids:
        dist_file = DIST_DIR / f"{pdb_id}.pt"
        if not dist_file.exists(): continue
        
        data = torch.load(dist_file, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            d_traj = data['pred_distance']
        else:
            d_traj = data
        
        neg_traj = []
        hr_pos_traj = []
        rank13_traj = []
        
        blocks = range(48) # All blocks for context
        for b in blocks:
            try:
                # Handle tensor
                if d_traj.ndim == 4:
                    d_block = d_traj[b, 0].float().numpy()
                else:
                    d_block = d_traj[b].float().numpy()
                
                n, hr, r13 = get_spectral_components(d_block)
                neg_traj.append(n)
                hr_pos_traj.append(hr)
                rank13_traj.append(r13)
            except Exception as e:
                print(f"Error processing {pdb_id} block {b}: {e}")
                break
                
        all_trajectories[pdb_id] = {
            "neg": neg_traj,
            "hr_pos": hr_pos_traj,
            "rank13": rank13_traj
        }
    
    # Calculate average rates in the Crystallization Phase (Blocks 40-47)
    # Rate = value[47] / value[40] (Closer to 0 means faster shedding)
    shed_analysis = {}
    for pdb_id, trajs in all_trajectories.items():
        if len(trajs["neg"]) < 48: continue
        
        def get_rate(arr):
            # Use Block 40 to 47
            v40 = arr[40] + 1e-9
            v47 = arr[47]
            return float(v47 / v40)
            
        shed_analysis[pdb_id] = {
            "neg_shed_rate": get_rate(trajs["neg"]),
            "hr_pos_shed_rate": get_rate(trajs["hr_pos"]),
            "rank13_shed_rate": get_rate(trajs["rank13"])
        }

    with open(OUT_PATH, 'w') as f:
        json.dump({"trajectories": all_trajectories, "shed_analysis": shed_analysis}, f, indent=2)
    
    print("Spectral Shedding Rates (Block 47 / Block 40):")
    for pdb_id, rates in shed_analysis.items():
        print(f"  {pdb_id}: Neg={rates['neg_shed_rate']:.4f}, HR_Pos={rates['hr_pos_shed_rate']:.4f}, Rank1-3={rates['rank13_shed_rate']:.4f}")

if __name__ == "__main__":
    main()
