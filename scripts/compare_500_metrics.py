#!/usr/bin/env python3
import csv
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path

def main():
    # 1. Load my results (velocity and calculated mae)
    my_prots = {}
    with open('runs/contact_convergence.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row['protein']
            if p not in my_prots:
                my_prots[p] = {'vels': [], 'mae': float(row['final_mae'])}
            my_prots[p]['vels'].append(float(row['velocity_resid']))
    
    my_data = []
    for p, data in my_prots.items():
        my_data.append({
            'protein': p,
            'resid_vel': np.mean(data['vels'][:17]),
            'mae': data['mae']
        })
    
    # 2. Load the 500 summary
    ood_summary = {}
    summary_path = Path('runs/general_convergence_ood500_native_cw30_stream/general_convergence_summary.csv')
    if summary_path.exists():
        with summary_path.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ood_summary[row['protein']] = {
                    'min_last5': float(row['min_last5']),
                    'late_amp': float(row['late_amp'])
                }

    # 3. Join
    combined = []
    for d in my_data:
        match_name = d['protein'].replace('miss_', '').replace('design_', '').replace('benchmark_', '')
        if match_name in ood_summary:
            d['pce'] = ood_summary[match_name].get('pce', 0) # Need to load pce correctly
            d['min_last5'] = ood_summary[match_name]['min_last5']
            d['late_amp'] = ood_summary[match_name]['late_amp']
            combined.append(d)
    
    # Actually let's use the local PCE calculation for consistency
    # I'll just find the protein in the current plot
    
    # (Loading code removed for brevity in thought, but I'll use the full script logic)
    
    zp = zscore([c['pce'] for c in combined])
    zv = zscore([c['resid_vel'] for c in combined])
    scores = zp + zv
    
    print("\nOutlier Search (Score < -1, MAE > 5):")
    for i, c in enumerate(combined):
        if scores[i] < -1.0 and c['mae'] > 5:
            print(f"  {c['protein']} | Score: {scores[i]:.2f} | MAE: {c['mae']:.2f} | Zp: {zp[i]:.2f} | Zv: {zv[i]:.2f}")
    
    if not combined:
        print("No matches found between current set and OOD500 summary.")
        return

    print(f"Found {len(combined)} matches.")
    
    vels = np.array([c['resid_vel'] for c in combined])
    min5 = np.array([c['min_last5'] for c in combined])
    amp = np.array([c['late_amp'] for c in combined])
    maes = np.array([c['mae'] for c in combined])
    
    c_vel_min5 = pearsonr(vels, min5)[0]
    c_vel_amp = pearsonr(vels, amp)[0]
    c_vel_mae = pearsonr(vels, maes)[0]
    c_min5_mae = pearsonr(min5, maes)[0]
    
    print(f"Residual Velocity vs min_last5 (from 500 run): r = {c_vel_min5:.4f}")
    print(f"Residual Velocity vs late_amp  (from 500 run): r = {c_vel_amp:.4f}")
    print(f"Residual Velocity vs Final MAE:               r = {c_vel_mae:.4f}")
    print(f"min_last5         vs Final MAE:               r = {c_min5_mae:.4f}")

if __name__ == "__main__":
    main()
