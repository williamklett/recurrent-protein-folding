import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import sys
import os

sys.path.insert(0, os.getcwd())
from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers

def main():
    device = torch.device("cpu")
    centers = distogram_bin_centers(device, torch.float32).reshape(-1)[:64]
    cache = torch.load("runs/esmfold_probe_smoke_cache.pt", map_location=device, weights_only=False)
    z_all = cache["z"].squeeze(1).float()
    
    L = z_all.shape[1]
    probes_dir = Path("runs/general_belief_replay_benchmark20_cw30_partial")
    
    layers = [20, 30, 40, 46]
    
    print(f"Layer | Mean Entropy | Clash Mass (>3.5A) | Bimodal Frac (>0.1 peak separation)")
    print("------|--------------|--------------------|------------------------------------")
    
    for t in layers:
        p_path = probes_dir / f"layer_{t:02d}" / "probe.pt"
        if not p_path.exists(): continue
        
        pr = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
        pr.load_state_dict(torch.load(p_path, map_location=device, weights_only=True))
        
        ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
        ii_f, jj_f = ii.reshape(-1), jj.reshape(-1)
        pairs = z_all[t][ii_f, jj_f]
        
        with torch.no_grad():
            logits = pr.encode(pairs, ii_f, jj_f, L)
            probs = F.softmax(logits, dim=-1).numpy()
        
        # Entropy
        ent = -(probs * np.log(probs + 1e-9)).sum(axis=-1).mean()
        
        # Clash mass (sum of bins where distance < 3.5A)
        # Bins are usually 2.3 to 21.7 in 64 bins. 
        # (21.7-2.3)/64 = 0.3A per bin.
        # 3.5A is bin ~ (3.5 - 2.3)/0.3 = 4.
        clash_idx = centers < 3.5
        clash_mass = probs[:, clash_idx].sum(axis=-1).mean()
        
        # Bimodality check: simple peak counting
        # A pair is bimodal if it has two peaks separated by at least 3 bins and > 0.1 height
        bimodal_count = 0
        for p in probs:
            peaks = []
            for i in range(1, 63):
                if p[i] > p[i-1] and p[i] > p[i+1] and p[i] > 0.1:
                    peaks.append(i)
            if len(peaks) >= 2 and (peaks[-1] - peaks[0]) >= 5:
                bimodal_count += 1
        
        print(f" {t:2d}   | {ent:12.3f} | {clash_mass:18.4f} | {bimodal_count/len(probs):34.4f}")

if __name__ == "__main__":
    main()
