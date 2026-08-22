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

def lj_energy_mean(D, sigma=6.8, seq_sep=3):
    L = D.shape[0]
    ii, jj = np.triu_indices(L, k=seq_sep + 1)
    d = D[ii, jj].clip(min=0.5)
    r = sigma / d
    e = 4 * (r**12 - r**6)
    return float(e.sum()) / L

def lj_energy_expected(P, centers, sigma=6.8, seq_sep=3):
    L = P.shape[0]
    ii, jj = np.triu_indices(L, k=seq_sep + 1)
    p_pairs = P[ii, jj]
    d = centers.clip(min=0.5)
    r = sigma / d[None, :]
    e_bins = 4 * (r**12 - r**6)
    e_expected = (p_pairs * e_bins).sum(axis=-1)
    return float(e_expected.sum()) / L

def main():
    device = torch.device("cpu")
    centers = distogram_bin_centers(device, torch.float32).reshape(-1)[:64]
    centers_np = centers.numpy()
    
    # Load the results mapping to find TM scores for whatever we find
    tm_map = {}
    with open('runs/lj_energy_results_full.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = row['name'].replace('benchmark_', '')
            tm_map[clean] = float(row['tm'])

    # Find ALL activations
    all_act = list(ROOT.glob("runs/**/esmfold_activations.pt"))
    all_act.append(ROOT / "runs/esmfold_probe_smoke_cache.pt")
    
    results = []
    
    for act_p in all_act:
        if not act_p.exists(): continue
        # Identify the protein
        prot = None
        for k in tm_map:
            if k in str(act_p):
                prot = k; break
        if not prot: 
            if "smoke_cache" in str(act_p): prot = "8fmi" # Special case
            else: continue
            
        print(f"Analyzing {prot} ({act_p}) [TM={tm_map.get(prot, 'N/A')}]...")
        try:
            cache = torch.load(act_p, map_location=device, weights_only=False)
            z_all = cache["z"].squeeze(1).float()
        except: continue
        
        prob_matrices = []
        dist_matrices = []
        for t in range(20, 47):
            p_path = PROBES_DIR / f"layer_{t:02d}" / "probe.pt"
            if p_path.exists():
                pr = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
                pr.load_state_dict(torch.load(p_path, map_location=device, weights_only=True))
                D, P = get_ed_and_p(z_all[t], pr, centers, device)
                dist_matrices.append(D)
                prob_matrices.append(P)
        
        if len(prob_matrices) > 1:
            e_means = [lj_energy_mean(D) for D in dist_matrices]
            e_exps = [lj_energy_expected(P, centers_np) for P in prob_matrices]
            
            mono_mean = np.mean(np.diff(e_means) <= 0)
            mono_exp = np.mean(np.diff(e_exps) <= 0)
            
            results.append({
                "protein": prot,
                "tm": tm_map.get(prot, 0),
                "mono_mean": mono_mean,
                "mono_exp": mono_exp
            })
            print(f"  Mono Mean: {mono_mean:.2f} | Mono Exp: {mono_exp:.2f}")

    if results:
        high_tm = [r for r in results if r['tm'] > 0.8]
        if high_tm:
            print(f"\nCorrectly Folded (TM > 0.8) n={len(high_tm)}:")
            print(f"  Mean Monotone Frac (Mean Energy): {np.mean([r['mono_mean'] for r in high_tm]):.3f}")
            print(f"  Mean Monotone Frac (Expect Energy): {np.mean([r['mono_exp'] for r in high_tm]):.3f}")
        
        low_tm = [r for r in results if r['tm'] < 0.5]
        if low_tm:
            print(f"\nFailed Folds (TM < 0.5) n={len(low_tm)}:")
            print(f"  Mean Monotone Frac (Mean Energy): {np.mean([r['mono_mean'] for r in low_tm]):.3f}")
            print(f"  Mean Monotone Frac (Expect Energy): {np.mean([r['mono_exp'] for r in low_tm]):.3f}")

if __name__ == "__main__":
    main()
