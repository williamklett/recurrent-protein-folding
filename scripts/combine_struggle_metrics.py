#!/usr/bin/env python3
import csv
import numpy as np
from scipy.stats import pearsonr, zscore
from pathlib import Path

def kadane_max(arr):
    best = cur = 0.0
    for v in arr:
        cur = max(0.0, cur + v)
        best = max(best, cur)
    return best

def pce_raw_fn(corr):
    d2 = np.diff(corr.astype(float), n=2)
    return kadane_max(np.maximum(d2, 0.0))

def main():
    OOD_DIR = Path('runs/general_convergence_ood500_native_cw30_stream')
    
    # 1. Load full trajectories for PCE calculation
    pce_dict = {}
    for p_dir in OOD_DIR.glob('*_analysis'):
        p_name = p_dir.name.replace('_analysis', '')
        traj_path = p_dir / 'layerwise_distance_correlations.csv'
        if not traj_path.exists(): continue
        
        layers, corrs = [], []
        with open(traj_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                layers.append(int(row['layer']))
                corrs.append(float(row['corr_vs_47']))
        
        idx = np.argsort(layers)
        corrs = np.array(corrs)[idx]
        pce_dict[p_name] = pce_raw_fn(corrs)

    # 2. Load my Residual Velocity results
    my_data = {}
    with open('runs/contact_convergence.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row['protein']
            if p not in my_data: 
                my_data[p] = {'vels': [], 'mae': float(row['final_mae'])}
            my_data[p]['vels'].append(float(row['velocity_resid']))
    
    # 3. Combine on Matching Set
    combined = []
    for name, data in my_data.items():
        # Clean name for matching
        match_name = name.replace('miss_', '').replace('design_', '').replace('benchmark_', '')
        if match_name in pce_dict:
            combined.append({
                'protein': name,
                'pce': pce_dict[match_name],
                'vel': np.mean(data['vels'][:17]),
                'mae': data['mae']
            })

    if not combined:
        print("No matching proteins found.")
        return

    print(f"Combining metrics for N={len(combined)} proteins...")
    
    # Extract arrays
    pce_vals = np.array([c['pce'] for c in combined])
    vel_vals = np.array([c['vel'] for c in combined])
    mae_vals = np.array([c['mae'] for c in combined])
    
    # Compute Z-scores
    zp = zscore(pce_vals)
    zv = zscore(vel_vals)
    
    # The Improvement: Combined Struggle Metric
    # Both are higher for worse folds
    struggle_score = zp + zv
    
    # Correlations
    r_pce = pearsonr(pce_vals, mae_vals)[0]
    r_vel = pearsonr(vel_vals, mae_vals)[0]
    r_comb = pearsonr(struggle_score, mae_vals)[0]
    
    print(f"\nMetric Correlations with Final MAE:")
    print(f"  PCE (Structural):        r = {r_pce:.4f}")
    print(f"  Velocity (Biochemical):  r = {r_vel:.4f}")
    print(f"  COMBINED STRUGGLE:       r = {r_comb:.4f}")
    
    improvement = (r_comb - max(r_pce, r_vel)) / max(r_pce, r_vel) * 100
    print(f"\nImprovement over best single metric: {improvement:.1f}%")

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))
    
    # Color by MAE (log scale for better visibility of failure cases)
    sc = plt.scatter(struggle_score, mae_vals, c=np.log10(mae_vals + 1e-3), 
                     cmap='viridis', alpha=0.8, edgecolors='w', s=100)
    plt.colorbar(sc, label='Log10(MAE)')
    
    plt.xlabel('Combined Struggle Score [Z(PCE) + Z(Residual Velocity)]', fontsize=12)
    plt.ylabel('Final MAE (Å)', fontsize=12)
    plt.title(f'Improvement through Combination: r={r_comb:.3f}\nIsolating Non-Convexity Struggle', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, ls='--')
    
    # Label top outliers
    for i, c in enumerate(combined):
        if struggle_score[i] > 2 or mae_vals[i] > 15:
            plt.annotate(c['protein'], (struggle_score[i], mae_vals[i]), 
                         fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig('runs/combined_struggle_analysis.png', dpi=180)
    print("\nResult plot saved to runs/combined_struggle_analysis.png")

if __name__ == "__main__":
    main()
