import json, numpy as np
d = json.load(open('/home/williamklett/recurrent-protein-folding/runs/attractor_experiment_mae/results.json'))

print('PAIRS — symmetric or directional degradation?')
print(f"{'A->B':14s}  base_src  base_tgt  Δsrc   Δtgt   Δtgt<Δsrc?")
for r in d['pairs']:
    ds = r['steer_mae_vs_src'] - r['base_mae_vs_src']
    dt = r['delta_mae_tgt']
    label = r['src'] + '-' + r['tgt']
    flag = 'YES' if dt < ds else ''
    print(f"{label:14s}  {r['base_mae_vs_src']:6.2f}    {r['base_mae_vs_tgt']:6.2f}    {ds:+5.2f}  {dt:+5.2f}  {flag}")

ds_all = [r['steer_mae_vs_src'] - r['base_mae_vs_src'] for r in d['pairs']]
dt_all = [r['delta_mae_tgt'] for r in d['pairs']]
print(f"\nmean Δsrc={float(sum(ds_all)/len(ds_all)):+.3f}  mean Δtgt={float(sum(dt_all)/len(dt_all)):+.3f}")
print(f"Δtgt < Δsrc in {sum(t<s for t,s in zip(dt_all,ds_all))}/10 pairs")
