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
    pce_dict = {}
    for p_dir in OOD_DIR.glob('*_analysis'):
        pn = p_dir.name.replace('_analysis', '')
        traj_path = p_dir / 'layerwise_distance_correlations.csv'
        if not traj_path.exists(): continue
        layers, corrs = [], []
        with open(traj_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                layers.append(int(row['layer']))
                corrs.append(float(row['corr_vs_47']))
        idx = np.argsort(layers)
        pce_dict[pn] = pce_raw_fn(np.array(corrs)[idx])

    summary_dict = {}
    summary_path = OOD_DIR / 'general_convergence_summary.csv'
    with open(summary_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary_dict[row['protein']] = {
                'tm_fail': 1.0 - float(row['tm_score']),
                'mae': float(row['mae']),
                'instability': 1.0 - float(row['min_last5'])
            }

    my_data = {}
    with open('runs/contact_convergence.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row['protein']
            if p not in my_data: my_data[p] = []
            my_data[p].append(float(row['velocity_resid']))

    combined = []
    for pk, vlist in my_data.items():
        m = pk.replace('miss_', '').replace('design_', '').replace('benchmark_', '')
        if m in pce_dict and m in summary_dict:
            combined.append({
                'name': pk,
                'pce': pce_dict[m],
                'vel': np.mean(vlist[:17]),
                'mae': summary_dict[m]['mae'],
                'tm_fail': summary_dict[m]['tm_fail'],
                'conv_fail': summary_dict[m]['instability']
            })

    if not combined: return

    p = np.array([c['pce'] for c in combined])
    v = np.array([c['vel'] for c in combined])
    
    def analyze_target(target_arr, target_name):
        rp = pearsonr(p, target_arr)[0]
        rv = pearsonr(v, target_arr)[0]
        
        # ── THE ALIGNMENT FIX ──
        # struggle = (sign of correlation) * Z-score
        # This guarantees both components move in the "Worse" direction.
        struggle = (np.sign(rp) * zscore(p)) + (np.sign(rv) * zscore(v))
        r_comb = pearsonr(struggle, target_arr)[0]
        
        print(f"\nTarget: {target_name}")
        print(f"  PCE Correlation:     {rp:8.4f}")
        print(f"  Velocity Correlation: {rv:8.4f}")
        print(f"  COMBINED Correlation: {r_comb:8.4f}")
        print(f"  Best Single Metric:  {max(abs(rp), abs(rv)):8.4f}")
        imp = (r_comb - max(abs(rp), abs(rv))) / max(abs(rp), abs(rv)) * 100
        print(f"  Improvement:         {imp:+.1f}%")

    analyze_target(np.array([c['tm_fail'] for c in combined]), "Structural Failure (1-TM)")
    analyze_target(np.array([c['mae'] for c in combined]), "Distance Error (MAE)")
    analyze_target(np.array([c['conv_fail'] for c in combined]), "Late-stage Instability (1-min5)")

if __name__ == "__main__":
    main()
