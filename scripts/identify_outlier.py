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

    my_vels = {}
    with open('runs/contact_convergence.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row['protein']
            if p not in my_vels: my_vels[p] = {'vels': [], 'mae': float(row['final_mae'])}
            my_vels[p]['vels'].append(float(row['velocity_resid']))

    combined = []
    for pk, vd in my_vels.items():
        m = pk.replace('miss_', '').replace('design_', '').replace('benchmark_', '')
        if m in pce_dict:
            combined.append({'name': pk, 'pce': pce_dict[m], 'vel': np.mean(vd['vels'][:17]), 'mae': vd['mae']})

    p_vals = np.array([c['pce'] for c in combined])
    v_vals = np.array([c['vel'] for c in combined])
    zp = zscore(p_vals)
    zv = zscore(v_vals)
    score = zp + zv

    print('Outlier Identification (Score ~ -2, MAE ~ 6):')
    for i, c in enumerate(combined):
        if score[i] < -0.5 and c['mae'] > 5:
            print(f"  {c['name']:20s} | Score: {score[i]:6.2f} | MAE: {c['mae']:5.2f} | Zp: {zp[i]:5.2f} | Zv: {zv[i]:5.2f}")

if __name__ == "__main__":
    main()
