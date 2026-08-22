
import os
import numpy as np
import torch
import csv
import json
from pathlib import Path
from scipy.stats import pearsonr

ROOT = Path("/Users/williamklett/Desktop/Natural Bio Encoder")
MANIFEST = ROOT / "data" / "manifest.csv"
OUT_PATH = ROOT / "runs/complexity_suite_analysis.json"
REFINED_OUT_PATH = ROOT / "runs/complexity_foldclass_analysis.json"

def main():
    if not OUT_PATH.exists():
        print("Run complexity_suite.py first.")
        return

    with open(OUT_PATH, 'r') as f:
        data = json.load(f)
    
    results = data["results"]
    
    # Split by fold class
    by_class = {}
    for r in results:
        fc = r["fold_class"]
        if fc not in by_class:
            by_class[fc] = []
        by_class[fc].append(r)
        
    class_summary = {}
    for fc, res_list in by_class.items():
        if len(res_list) < 5: continue
        
        metrics = ["aco", "cross_frac", "L"]
        summary = {}
        for m in metrics:
            r_val, p_val = pearsonr([res[m] for res in res_list], [res["neg_frac"] for res in res_list])
            summary[f"{m}_vs_neg_frac"] = {"r": float(r_val), "p": float(p_val)}
        
        class_summary[fc] = {
            "summary": summary,
            "n": len(res_list),
            "mean_neg_frac": float(np.mean([res["neg_frac"] for res in res_list]))
        }

    with open(REFINED_OUT_PATH, 'w') as f:
        json.dump(class_summary, f, indent=2)
        
    print("Fold-Class Specific Correlations (r-values):")
    for fc, data in class_summary.items():
        s = data["summary"]
        print(f"  {fc:12} (n={data['n']}): ACO={s['aco_vs_neg_frac']['r']:.3f}, Cross={s['cross_frac_vs_neg_frac']['r']:.3f}, L={s['L_vs_neg_frac']['r']:.3f}, MeanNeg={data['mean_neg_frac']:.3f}")

if __name__ == "__main__":
    main()
