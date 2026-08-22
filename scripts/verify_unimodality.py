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

def main():
    device = torch.device("cpu")
    centers = distogram_bin_centers(device, torch.float32).reshape(-1)[:64]
    probes_dir = Path("runs/general_belief_replay_benchmark20_cw30_partial")
    
    # Load mapping to find TM scores
    tm_map = {}
    with open('runs/lj_energy_results_full.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = row['name'].replace('benchmark_', '')
            tm_map[clean] = float(row['tm'])

    # Find ALL activations
    all_act = list(Path("runs").glob("**/esmfold_activations.pt"))
    all_act.append(Path("runs/esmfold_probe_smoke_cache.pt"))
    
    print(f"Protein | TM | Avg Unimodal % (Layer 46) | Mean Entropy | Max Bimodal % in Trunk")
    print("-" * 80)
    
    for act_p in all_act:
        prot = None
        for k in tm_map:
            if k in str(act_p): prot = k; break
        if not prot: 
            if "smoke_cache" in str(act_p): prot = "8fmi"
            else: continue
            
        try:
            cache = torch.load(act_p, map_location=device, weights_only=False)
            z_all = cache["z"].squeeze(1).float()
        except: continue
        
        L = z_all.shape[1]
        
        # Test final layer 46
        p_path = probes_dir / "layer_46/probe.pt"
        if not p_path.exists(): continue
        
        pr = PairDistogramBottleneck(pair_dim=128, bins=64, hidden_dim=256).to(device).eval()
        pr.load_state_dict(torch.load(p_path, map_location=device, weights_only=True))
        
        ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
        ii_f, jj_f = ii.reshape(-1), jj.reshape(-1)
        pairs = z_all[46][ii_f, jj_f]
        
        with torch.no_grad():
            logits = pr.encode(pairs, ii_f, jj_f, L)
            probs = F.softmax(logits, dim=-1).numpy()
        
        unimodal_count = 0
        for p in probs:
            peaks = [i for i in range(1, 63) if p[i] > p[i-1] and p[i] > p[i+1] and p[i] > 0.1]
            if len(peaks) <= 1: unimodal_count += 1
            elif (peaks[-1] - peaks[0]) < 5: unimodal_count += 1 # peaks too close to count as bimodal basin
            
        unimodal_pct = (unimodal_count / len(probs)) * 100
        entropy = -(probs * np.log(probs + 1e-9)).sum(axis=-1).mean()
        
        print(f"{prot:7s} | {tm_map.get(prot, 0):.2f} | {unimodal_pct:23.1f}% | {entropy:12.2f}")

if __name__ == "__main__":
    main()
