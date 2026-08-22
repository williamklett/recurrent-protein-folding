
import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

def partial_corr(x, y, z):
    """
    Compute partial correlation of x and y, controlling for z.
    r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2) * (1 - r_yz^2))
    """
    r_xy = pearsonr(x, y)[0]
    r_xz = pearsonr(x, z)[0]
    r_yz = pearsonr(y, z)[0]
    
    num = r_xy - (r_xz * r_yz)
    den = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    return num / den

def main():
    root = Path("/Users/williamklett/Desktop/Natural Bio Encoder")
    data_path = root / "runs/final_complexity_synthesis.json"
    
    with open(data_path) as f:
        data = json.load(f)
        
    results = data["results"]
    
    L = np.array([r["L"] for r in results])
    neg_frac = np.array([r["neg_frac"] for r in results])
    abs_cross = np.array([r["abs_cross"] for r in results])
    aco = np.array([r["aco"] for r in results])
    
    # 1. Basic correlations with L
    r_l_neg = pearsonr(L, neg_frac)[0]
    r_l_cross = pearsonr(L, abs_cross)[0]
    r_l_aco = pearsonr(L, aco)[0]
    
    # 2. Crossing Density (normalized by L^2 since total potential pairs scale with L^2)
    cross_density = abs_cross / (L ** 2)
    r_dense_neg = pearsonr(cross_density, neg_frac)[0]
    
    # 3. Partial Correlations
    # Does Crossings predict NegFrac above and beyond L?
    p_r_cross_neg = partial_corr(abs_cross, neg_frac, L)
    
    # Does ACO predict NegFrac above and beyond L?
    p_r_aco_neg = partial_corr(aco, neg_frac, L)

    output = {
        "correlations": {
            "L_vs_neg_frac": float(r_l_neg),
            "L_vs_abs_cross": float(r_l_cross),
            "L_vs_aco": float(r_l_aco),
            "crossing_density_vs_neg_frac": float(r_dense_neg)
        },
        "partial_correlations_controlled_for_L": {
            "abs_cross_vs_neg_frac_partial": float(p_r_cross_neg),
            "aco_vs_neg_frac_partial": float(p_r_aco_neg)
        }
    }
    
    with open(root / "runs/partial_correlation_analysis.json", 'w') as f:
        json.dump(output, f, indent=2)
        
    print("Length-Controlled Correlation Analysis:")
    print(f"  L vs NegFrac: {r_l_neg:.4f}")
    print(f"  L vs Absolute Crossings: {r_l_cross:.4f}")
    print(f"  Crossing Density vs NegFrac: {r_dense_neg:.4f}")
    print(f"\nPartial Correlations (Controlled for L):")
    print(f"  Absolute Crossings vs NegFrac: {p_r_cross_neg:.4f}")
    print(f"  Absolute Contact Order vs NegFrac: {p_r_aco_neg:.4f}")

if __name__ == "__main__":
    main()
