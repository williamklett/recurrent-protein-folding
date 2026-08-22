from pathlib import Path
import csv, json
import numpy as np
import torch
import torch.nn.functional as F
from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers

ROOT=Path('runs')
old45=Path('runs/general_belief_replay_hard20/layer_45/probe.pt')
new46=Path('runs/general_belief_replay_hard20_cw30_l46/layer_46/probe.pt')
dirs=[Path(x.strip()) for x in Path('runs/hard_synthetic_general_train_dirs.txt').read_text().splitlines() if x.strip()]
out=Path('runs/general_belief_replay_hard20_cw30_l46/l45_old_vs_l46_new.csv')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
centers=distogram_bin_centers(device, torch.float32)

def load(p):
    m=PairDistogramBottleneck(pair_dim=128,bins=ESMFOLD_DISTO_BINS,hidden_dim=256).to(device).eval()
    m.load_state_dict(torch.load(p,map_location=device,weights_only=True))
    for pp in m.parameters(): pp.requires_grad_(False)
    return m
p45=load(old45); p46=load(new46)
rng=np.random.default_rng(123)
rows=[]
for d in dirs:
    c=torch.load(d/'esmfold_activations.pt',map_location='cpu',weights_only=False,mmap=True)
    z=c['z'].squeeze(1); L=int(z.shape[1])
    n=8192
    i=torch.from_numpy(rng.integers(0,L,n)).long(); j=torch.from_numpy(rng.integers(0,L-1,n)).long(); j=j+(j>=i)
    with torch.no_grad():
        lg45=p45.encode(z[45][i,j].to(device,torch.float32), i.to(device), j.to(device), L)
        lg46=p46.encode(z[46][i,j].to(device,torch.float32), i.to(device), j.to(device), L)
        prob46=F.softmax(lg46,-1)
        kl=F.kl_div(F.log_softmax(lg45,-1), prob46, reduction='batchmean').item()
        ed45=(F.softmax(lg45,-1)*centers).sum(-1).cpu().numpy()
        ed46=(prob46*centers).sum(-1).cpu().numpy()
    corr=float(np.corrcoef(ed45,ed46)[0,1]) if np.std(ed45)>1e-9 and np.std(ed46)>1e-9 else float('nan')
    rows.append({'protein':d.name,'length':L,'kl_old45_to_new46':kl,'ed_diff':float(np.abs(ed45-ed46).mean()),'corr':corr})
    print(rows[-1], flush=True)
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('wrote',out)
print('mean ed_diff',np.mean([r['ed_diff'] for r in rows]),'mean kl',np.mean([r['kl_old45_to_new46'] for r in rows]),'mean corr',np.nanmean([r['corr'] for r in rows]))
