import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import sys
import os

sys.path.insert(0, os.getcwd())
from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers

ROOT = Path(".")
# Use the smoke cache and any other valid activation file
PATHS = [
    ROOT / "runs/esmfold_probe_smoke_cache.pt",
    ROOT / "runs/remote/esmfold_probe_real_smoke/esmfold_activations.pt"
]
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
    
    all_mean_mono = []
    all_expect_mono = []
    
    for path in PATHS:
        if not path.exists(): continue
        print(f"Loading {path}...")
        try:
            cache = torch.load(path, map_location=device, weights_only=False)
            z_all = cache["z"].squeeze(1).float()
        except: continue
        
        dist_matrices = []
        prob_matrices = []
        for t in range(20, 47): # partial range
            p_path = PROBES_DIR / f"layer_{t:02d}" / "probe.pt"
            if p_path.exists():
                pr = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
                pr.load_state_dict(torch.load(p_path, map_location=device, weights_only=True))
                D, P = get_ed_and_p(z_all[t], pr, centers, device)
                dist_matrices.append(D)
                prob_matrices.append(P)
        
        if len(dist_matrices) > 1:
            e_means = [lj_energy_mean(D) for D in dist_matrices]
            e_exps = [lj_energy_expected(P, centers_np) for P in prob_matrices]
            
            mono_mean = np.mean(np.diff(e_means) <= 0)
            mono_exp = np.mean(np.diff(e_exps) <= 0)
            
            all_mean_mono.append(mono_mean)
            all_expect_mono.append(mono_exp)
            print(f"  Mono: Mean={mono_mean:.2f}, Expect={mono_exp:.2f}")

    if all_mean_mono:
        print(f"\nAcross {len(all_mean_mono)} proteins (layers 20-46):")
        print(f"Average Mono Frac (Mean Energy): {np.mean(all_mean_mono):.3f}")
        print(f"Average Mono Frac (Expect Energy): {np.mean(all_expect_mono):.3f}")

if __name__ == "__main__":
    main()
