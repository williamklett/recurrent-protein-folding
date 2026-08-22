import json, math, os, torch, torch.nn.functional as F
import numpy as np
from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
NB = ESMFOLD_DISTO_BINS
logNB = math.log(NB)

def pent(logits, L, sep=4):
    p = F.softmax(logits.float(), dim=-1)
    H = -(p * torch.clamp(p, min=1e-12).log()).sum(-1)
    iu = torch.triu_indices(L, L, offset=sep, device=logits.device)
    return float((H[iu[0], iu[1]] / logNB).mean())

runs_dir = 'runs'
out = {}

# Use the exact same set of proteins I found earlier
prefixes = ['benchmark_', 'synthetic_design_', 'hard_synthetic_miss_']
proteins = [d for d in os.listdir(runs_dir) if any(d.startswith(p) for p in prefixes) and os.path.isdir(os.path.join(runs_dir, d))]

for name in proteins:
    d = os.path.join(runs_dir, name)
    cap = os.path.join(d, 'esmfold_activations.pt')
    if not os.path.exists(cap): continue
    
    if not (os.path.exists(os.path.join(d, 'layer_01/probe.pt')) and os.path.exists(os.path.join(d, 'layer_02/probe.pt'))):
        continue

    try:
        c = torch.load(cap, map_location='cpu', weights_only=False)
        z = c['z'].squeeze(1)
        L = z.shape[1]
        del c
        
        mj = os.path.join(d, 'analysis/esmfold_direct_eval/metrics.json')
        if not os.path.exists(mj): mj = os.path.join(d, 'metrics.json')
        met = json.load(open(mj)) if os.path.exists(mj) else {}
        dcorr = met.get('distance_corr_all_pairs')
        if dcorr is None: continue

        ent = []
        for layer in [1, 2]:
            pp = os.path.join(d, f'layer_{layer:02d}/probe.pt')
            pr = PairDistogramBottleneck(pair_dim=128, bins=NB, hidden_dim=256).to(dev).eval()
            pr.load_state_dict(torch.load(pp, map_location=dev, weights_only=True))
            
            zt = z[layer-1].float().to(dev)
            i, j = torch.meshgrid(torch.arange(L, device=dev), torch.arange(L, device=dev), indexing='ij')
            with torch.no_grad():
                lg = pr.encode(zt.reshape(-1, 128), i.flatten(), j.flatten(), L).reshape(L, L, NB)
                ent.append(pent(lg, L))
            del zt, lg, pr; torch.cuda.empty_cache()
        
        out[name] = {'norm_entropy': ent, 'dcorr': dcorr}
        print(f"Processed {name}")
    except Exception as e:
        pass

with open('vm_60_consistent.json', 'w') as f:
    json.dump(out, f)
