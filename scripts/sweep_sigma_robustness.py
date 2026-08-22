#!/usr/bin/env python3
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers

ROOT = Path(__file__).resolve().parents[1]
PROBES_DIR = ROOT / "runs/general_belief_replay_benchmark20_cw30"
RUNS_DIR = ROOT / "runs"
FIRST, LAST = 1, 46

def load_probe(layer, device):
    path = PROBES_DIR / f"layer_{layer:02d}" / "probe.pt"
    if not path.exists(): return None
    pr = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
    pr.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return pr

def get_ed(z_layer, probe, centers, device):
    L = z_layer.shape[1] if z_layer.dim() == 4 else z_layer.shape[0]
    z_layer = z_layer.squeeze()
    ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    ii_f, jj_f = ii.reshape(-1), jj.reshape(-1)
    pairs = z_layer[ii_f, jj_f].to(device)
    with torch.no_grad():
        logits = probe.encode(pairs, ii_f.to(device), jj_f.to(device), L)
        ed = (F.softmax(logits, -1) * centers).sum(-1)
    return ed.reshape(L, L).cpu().numpy()

def lj_energy(D, sigma, seq_sep=3):
    L = D.shape[0]
    ii, jj = np.triu_indices(L, k=seq_sep + 1)
    d = D[ii, jj].clip(min=0.5)
    r = sigma / d
    e = 4 * (r**12 - r**6)
    return float(e.sum()) / L

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    centers = distogram_bin_centers(device, torch.float32).reshape(-1)[:64]
    
    # Using the one activation file we found
    act_p = ROOT / "runs/remote/esmfold_probe_8fmi_20layers/esmfold_activations.pt"
    if not act_p.exists():
        print(f"Could not find {act_p}")
        return

    name = "8fmi"
    print(f"Processing {name} from {act_p}...")
    act = torch.load(act_p, map_location="cpu", weights_only=False)
    z_all = act["z"].squeeze(1).float()
    
    # Cache distance matrices
    dist_matrices = []
    for t in range(FIRST, LAST + 1):
        probe = load_probe(t, device)
        if probe is None: 
            dist_matrices.append(None)
            continue
        dist_matrices.append(get_ed(z_all[t], probe, centers, device))
        
    sigmas = np.linspace(4.0, 10.0, 7)
    results = {}
    for s in sigmas:
        energies = []
        for D in dist_matrices:
            if D is None: continue
            energies.append(lj_energy(D, s))
        
        mono_frac = np.mean(np.diff(energies) <= 0)
        results[float(s)] = {
            "energies": energies,
            "mono_frac": float(mono_frac)
        }
        print(f"  sigma={s:.1f} mono={mono_frac:.2f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    for s in sigmas:
        plt.plot(range(FIRST, LAST+1), results[s]["energies"], label=f"sigma={s:.1f}")
    plt.title(f"{name} Energy Sweep")
    plt.xlabel("Layer")
    plt.ylabel("LJ Energy")
    plt.legend()
    plt.grid(True)
    plt.savefig("runs/sigma_sweep_robustness.png")
    print("Saved plot to runs/sigma_sweep_robustness.png")

    print("\nRobustness Table (Monotone Fraction):")
    row = f"{name:7s} | " + " | ".join([f"s={s:.1f}: {results[s]['mono_frac']:.2f}" for s in sigmas])
    print(row)

if __name__ == "__main__":
    main()
