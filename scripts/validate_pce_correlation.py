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

def pce_raw_fn(corr):
    d2 = np.diff(corr.astype(float), n=2)
    return kadane_max(np.maximum(d2, 0.0))

def pce_smooth_fn(corr, sigma=1.5):
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(corr.astype(float), sigma=sigma)
    d2 = np.diff(smoothed, n=2)
    return kadane_max(np.maximum(d2, 0.0))

def main():
    OOD_DIR = Path('runs/general_convergence_ood500_native_cw30_stream')
    
    # 1. Load Summary for MAE, min_last5, TM-score
    mae_dict = {}
    min5_dict = {}
    tm_dict = {}
    summary_path = OOD_DIR / 'general_convergence_summary.csv'
    with open(summary_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row['protein']
            mae_dict[p] = float(row['mae'])
            min5_dict[p] = float(row['min_last5'])
            tm_dict[p] = float(row['tm_score'])

    # 2. Calculate PCE for all 500
    results_500 = []
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
        
        if p_name in mae_dict:
            results_500.append({
                'protein': p_name,
                'pce_raw': pce_raw_fn(corrs),
                'pce_smooth': pce_smooth_fn(corrs, sigma=1.5),
                'mae': mae_dict[p_name],
                'tm': tm_dict[p_name],
                'min5': min5_dict[p_name]
            })

    # Global Stats
    maes_500 = np.array([r['mae'] for r in results_500])
    raws_500 = np.array([r['pce_raw'] for r in results_500])
    tms_500 = np.array([r['tm'] for r in results_500])
    m5_500 = np.array([r['min5'] for r in results_500])
    
    print(f"500 Run Stats (N={len(results_500)}):")
    print(f"  PCE Raw vs MAE:      r = {pearsonr(raws_500, maes_500)[0]:.4f}")
    print(f"  PCE Raw vs TM-score: r = {pearsonr(raws_500, tms_500)[0]:.4f}")
    print(f"  min_last5 vs MAE:    r = {pearsonr(m5_500, maes_500)[0]:.4f}")

    # 3. Match with my Residual Velocity results
    my_vels = {}
    with open('runs/contact_convergence.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row['protein']
            if p not in my_vels: 
                my_vels[p] = {'vels': [], 'mae': float(row['final_mae'])}
            my_vels[p]['vels'].append(float(row['velocity_resid']))
    
    combined = []
    idx_500 = {r['protein']: r for r in results_500}
    
    for name, data in my_vels.items():
        match_name = name.replace('miss_', '').replace('design_', '').replace('benchmark_', '')
        if match_name in idx_500:
            res = idx_500[match_name]
            combined.append({
                'pce_raw': res['pce_raw'],
                'min5': res['min5'],
                'vel': np.mean(data['vels'][:17]),
                'mae': data['mae']
            })

    if combined:
        v = np.array([c['vel'] for c in combined])
        p = np.array([c['pce_raw'] for c in combined])
        m = np.array([c['min5'] for c in combined])
        e = np.array([c['mae'] for c in combined])
        
        print(f"\nMatching Set (N={len(combined)}):")
        print(f"  Residual Velocity vs PCE Raw:    r = {pearsonr(v, p)[0]:.4f}")
        print(f"  Residual Velocity vs min_last5:  r = {pearsonr(v, m)[0]:.4f}")
        print(f"  Residual Velocity vs Final MAE:  r = {pearsonr(v, e)[0]:.4f}")
        print(f"  PCE Raw           vs Final MAE:  r = {pearsonr(p, e)[0]:.4f}")
        print(f"  min_last5         vs Final MAE:  r = {pearsonr(m, e)[0]:.4f}")

if __name__ == "__main__":
    main()
