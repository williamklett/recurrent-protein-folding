import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import sys
import os
import csv

sys.path.insert(0, os.getcwd())
from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers

ROOT = Path(".")
PROBES_DIR = ROOT / "runs/general_belief_replay_benchmark20_cw30_partial"

def get_ed_and_p(z_layer, probe, centers, device):
    L = z_layer.shape[0]
    ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    ii_f, jj_f = ii.reshape(-1), jj.reshape(-1)
    pairs = z_layer[ii_f, jj_f].to(device)
    with torch.no_grad():
        logits = probe.encode(pairs, ii_f.to(device), jj_f.to(device), L)
        probs = F.softmax(logits, dim=-1)
        ed = (probs * centers).sum(-1)
    return ed.reshape(L, L).cpu().numpy(), probs.reshape(L, L, -1).cpu().numpy()

def lj_energy_expected(P, centers, sigma=6.8, seq_sep=3):
    L = P.shape[0]
    ii, jj = np.triu_indices(L, k=seq_sep + 1)
    p_pairs = P[ii, jj]
    d = centers.clip(min=0.5)
    r = sigma / d
    e_bins = 4 * (r**12 - r**6)
    e_expected = (p_pairs * e_bins).sum(axis=-1)
    return float(e_expected.sum()) / L

def main():
    device = torch.device("cpu")
    centers = distogram_bin_centers(device, torch.float32).reshape(-1)[:64]
    centers_np = centers.numpy()
    
    # Identify high TM proteins from the results file using csv module
    correct_folds = []
    with open('runs/lj_energy_results_full.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row['tm']) > 0.8:
                correct_folds.append(row['name'])
    
    print(f"Found {len(correct_folds)} high-TM proteins (>0.8)")
    
    all_expect_mono = []
    
    for name in correct_folds:
        clean = name.replace('benchmark_', '')
        # Search specifically in runs/benchmark_NAME/esmfold_activations.pt or similar
        act_paths = list(ROOT.glob(f"runs/**/{clean}*/esmfold_activations.pt"))
        if not act_paths:
            # Try smoke cache as backup
            if clean == "8fmi":
                act_paths = [ROOT / "runs/esmfold_probe_smoke_cache.pt"]
            else:
                continue
        
        act_p = act_paths[0]
        print(f"Analyzing {name} ({act_p})...")
        try:
            cache = torch.load(act_p, map_location=device, weights_only=False)
            # Handle smoke cache vs normal cache structure
            z_all = cache["z"].squeeze(1).float()
        except: continue
        
        prob_matrices = []
        for t in range(20, 47):
            p_path = PROBES_DIR / f"layer_{t:02d}" / "probe.pt"
            if p_path.exists():
                pr = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
                pr.load_state_dict(torch.load(p_path, map_location=device, weights_only=True))
                _, P = get_ed_and_p(z_all[t], pr, centers, device)
                prob_matrices.append(P)
        
        if len(prob_matrices) > 1:
            e_exps = [lj_energy_expected(P, centers_np) for P in prob_matrices]
            mono_exp = np.mean(np.diff(e_exps) <= 0)
            all_expect_mono.append(mono_exp)
            print(f"  Expected Energy Mono Frac: {mono_exp:.2f}")

    if all_expect_mono:
        print(f"\nAcross {len(all_expect_mono)} correctly folded proteins (layers 20-46):")
        print(f"Mean Monotone Fraction (Expected Energy): {np.mean(all_expect_mono):.3f}")
    else:
        print("No activations found for high-TM proteins on this machine.")

if __name__ == "__main__":
    main()
