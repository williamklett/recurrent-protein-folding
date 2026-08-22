import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.getcwd())
from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers

ROOT = Path(".")
CACHE_PATH = ROOT / "runs/esmfold_probe_smoke_cache.pt"
PROBES_DIR = ROOT / "runs/general_belief_replay_benchmark20_cw30_partial"

def get_ed(z_layer, probe, centers, device):
    L = z_layer.shape[0]
    ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    ii_f, jj_f = ii.reshape(-1), jj.reshape(-1)
    pairs = z_layer[ii_f, jj_f].to(device)
    with torch.no_grad():
        logits = probe.encode(pairs, ii_f.to(device), jj_f.to(device), L)
        probs = F.softmax(logits, dim=-1)
        ed = (probs * centers).sum(-1)
    return ed.reshape(L, L).cpu().numpy(), probs.reshape(L, L, -1).cpu().numpy()

def lj_energy_mean(D, sigma, seq_sep=3):
    L = D.shape[0]
    ii, jj = np.triu_indices(L, k=seq_sep + 1)
    d = D[ii, jj].clip(min=0.5)
    r = sigma / d
    e = 4 * (r**12 - r**6)
    return float(e.sum()) / L

def lj_energy_expected(P, centers, sigma, seq_sep=3):
    L = P.shape[0]
    ii, jj = np.triu_indices(L, k=seq_sep + 1)
    # P[ii, jj] is (N_pairs, bins)
    p_pairs = P[ii, jj]
    # centers is (bins,)
    d = centers.clip(min=0.5)
    r = sigma / d[None, :] # (1, bins)
    e_bins = 4 * (r**12 - r**6) # (1, bins)
    e_expected = (p_pairs * e_bins).sum(axis=-1) # (N_pairs,)
    return float(e_expected.sum()) / L

def main():
    device = torch.device("cpu")
    centers = distogram_bin_centers(device, torch.float32).reshape(-1)[:64]
    cache = torch.load(CACHE_PATH, map_location=device, weights_only=False)
    z_all = cache["z"].squeeze(1).float()
    
    dist_matrices = []
    prob_matrices = []
    layer_indices = []
    for t in range(20, 47):
        path = PROBES_DIR / f"layer_{t:02d}" / "probe.pt"
        if path.exists():
            pr = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
            pr.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            D, P = get_ed(z_all[t], pr, centers, device)
            dist_matrices.append(D)
            prob_matrices.append(P)
            layer_indices.append(t)

    sigmas = [5.0, 6.0, 6.8]
    centers_np = centers.numpy()
    
    print("\nRobustness Check (Layer 20-46):")
    print("Sigma | Metric | Mono Frac | E_start | E_end")
    print("------|--------|-----------|---------|-------")
    for s in sigmas:
        e_mean = [lj_energy_mean(D, s) for D in dist_matrices]
        m_mean = np.mean(np.diff(e_mean) <= 0)
        print(f"{s:5.1f} | Mean   | {m_mean:9.2f} | {e_mean[0]:7.2f} | {e_mean[-1]:7.2f}")
        
        e_exp = [lj_energy_expected(P, centers_np, s) for P in prob_matrices]
        m_exp = np.mean(np.diff(e_exp) <= 0)
        print(f"{s:5.1f} | Expect | {m_exp:9.2f} | {e_exp[0]:7.2f} | {e_exp[-1]:7.2f}")

if __name__ == "__main__":
    main()
