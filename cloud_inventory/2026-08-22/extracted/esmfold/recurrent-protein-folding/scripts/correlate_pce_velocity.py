#!/usr/bin/env python3
import csv
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path

def kadane_max(arr):
    best = cur = 0.0
    for v in arr:
        cur = max(0.0, cur + v)
        best = max(best, cur)
    return best

def pce_raw(corr):
    d2 = np.diff(corr.astype(float), n=2)
    return kadane_max(np.maximum(d2, 0.0))

def main():
    # 1. Load my results
    proteins = {}
    with open('runs/contact_convergence.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row['protein']
            if p not in proteins:
                proteins[p] = {
                    'layers': [], 'corrs': [], 
                    'vels_resid': [], 'vels_total': [],
                    'mae': float(row['final_mae'])
                }
            proteins[p]['layers'].append(int(row['layer']))
            proteins[p]['corrs'].append(float(row['corr_vs_final']))
            proteins[p]['vels_resid'].append(float(row['velocity_resid']))
            proteins[p]['vels_total'].append(float(row['velocity_total']))
    
    results = []
    for p, data in proteins.items():
        idx = np.argsort(data['layers'])
        corrs = np.array(data['corrs'])[idx]
        vels_r = np.array(data['vels_resid'])[idx]
        vels_t = np.array(data['vels_total'])[idx]
        layers = np.array(data['layers'])[idx]
        
        # Candidate PCE 1: Curvature based
        pce_curv = pce_raw(corrs)
        
        # Candidate PCE 2: Min last 5
        pce_min5 = corrs[-5:].min()
        
        # Velocity metrics (early average L1-17)
        mask = layers <= 17
        early_vel_resid = vels_r[mask].mean()
        early_vel_total = vels_t[mask].mean()
        
        results.append({
            'protein': p,
            'pce_curv': pce_curv,
            'pce_min5': pce_min5,
            'vel_resid': early_vel_resid,
            'vel_total': early_vel_total,
            'mae': data['mae']
        })
    
    # Analysis
    keys = ['pce_curv', 'pce_min5', 'vel_resid', 'vel_total', 'mae']
    data_mat = np.array([[r[k] for k in keys] for r in results])
    
    print("Correlation Matrix:")
    print(f"{'':12s} | " + " | ".join([f"{k:9s}" for k in keys]))
    print("-" * 70)
    for i, k1 in enumerate(keys):
        row_str = f"{k1:12s} | "
        vals = []
        for j, k2 in enumerate(keys):
            r = pearsonr(data_mat[:, i], data_mat[:, j])[0]
            vals.append(f"{r:9.4f}")
        print(row_str + " | ".join(vals))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(data_mat[:, 2], data_mat[:, 0], alpha=0.7, c='#7c3aed')
    plt.xlabel('Early Residual Velocity')
    plt.ylabel('PCE (Curvature)')
    plt.title(f'Resid Velocity vs PCE Curv (r={pearsonr(data_mat[:,2], data_mat[:,0])[0]:.2f})')
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(data_mat[:, 2], data_mat[:, 1], alpha=0.7, c='#2563eb')
    plt.xlabel('Early Residual Velocity')
    plt.ylabel('PCE (Min Last 5)')
    plt.title(f'Resid Velocity vs PCE Min5 (r={pearsonr(data_mat[:,2], data_mat[:,1])[0]:.2f})')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('runs/pce_velocity_matrix.png', dpi=150)
    print("\nDetailed plots saved to runs/pce_velocity_matrix.png")

if __name__ == "__main__":
    main()
