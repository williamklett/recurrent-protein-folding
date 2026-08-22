#!/usr/bin/env python3
"""Local Block Backprop Cosine Alignment Test.

Tests if block t's update Δz_t aligns with −∇_{z_t} LJ(distogram_{t+1}).
Uses general probes at each layer and backprops through a single block.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForProteinFolding

from recurrent_folding.esmfold_probes import PairDistogramBottleneck

# Constants for LJ energy
SIGMA = 6.8
EPS = 1.0
SEQ_SEP = 3

PROBE_DIR = Path("runs/general_belief_replay_benchmark20_cw30")

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}

def extract_sequence_and_ca(pdb_path: Path, chain_hint: str) -> tuple[str, np.ndarray, str]:
    from collections import Counter
    counts: Counter[str] = Counter()
    with pdb_path.open() as f:
        for ln in f:
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
                counts[ln[21]] += 1
    if not counts:
        raise ValueError(f"no CA atoms in {pdb_path}")
    hint = chain_hint.strip() or " "
    chain = hint if hint in counts and counts[hint] >= 20 else counts.most_common(1)[0][0]
    seq, coords, seen = [], [], set()
    with pdb_path.open() as f:
        for ln in f:
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[21] == chain:
                rs = (int(ln[22:26]), ln[26])
                if rs in seen:
                    continue
                seen.add(rs)
                seq.append(AA3.get(ln[17:20].strip(), "X"))
                coords.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return "".join(seq), np.asarray(coords, dtype=np.float32), chain

def get_distogram_bin_centers(device):
    start = 2.3125
    end = 21.6875
    bins = 64
    width = (end - start) / bins
    centers = torch.linspace(start + width/2, end - width/2, bins, device=device)
    return centers

def cosine_similarity(a, b):
    a = a.detach().flatten()
    b = b.detach().flatten()
    na, nb = a.norm(), b.norm()
    if float(na) < 1e-12 or float(nb) < 1e-12:
        return 0.0
    return float((a * b).sum() / (na * nb))

def load_probe(layer, device, pair_dim=128):
    path = PROBE_DIR / f"layer_{layer:02d}" / "probe.pt"
    if not path.exists():
        return None
    probe = PairDistogramBottleneck(pair_dim=pair_dim, bins=64, hidden_dim=256).to(device)
    probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    probe.eval()
    for p in probe.parameters():
        p.requires_grad = False
    return probe

def run_local_test(sequence, model, tokenizer, device, gt_coords=None):
    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False).to(device)
    centers = get_distogram_bin_centers(device)
    L = len(sequence)
    i_idx, j_idx = torch.meshgrid(torch.arange(L, device=device), torch.arange(L, device=device), indexing="ij")
    i_t, j_t = i_idx.flatten(), j_idx.flatten()
    sep = (j_idx - i_idx).abs()
    keep_mask = (sep > SEQ_SEP).float()

    if gt_coords is not None:
        gt_dist = torch.linalg.norm(gt_coords[:, None] - gt_coords[None, :], dim=-1).to(device)
        bin_width = (21.6875 - 2.3125) / 64
        gt_bins = ((gt_dist - 2.3125) / bin_width).long().clamp(0, 63)

    # Capture final belief (Layer 48)
    with torch.no_grad():
        final_out = model(**inputs, num_recycles=0)
        final_logits = final_out.distogram_logits
        final_probs = F.softmax(final_logits, dim=-1).detach()

    initial_state = {}
    def hook_simple(_m, args):
        if 's' not in initial_state:
            initial_state['s'] = args[0].detach()
            initial_state['z'] = args[1].detach()

    handle = model.trunk.blocks[0].register_forward_pre_hook(hook_simple)
    with torch.no_grad():
        model(**inputs, num_recycles=0)
    handle.remove()

    s_s = initial_state['s']
    s_z = initial_state['z']
    residx = torch.arange(L, device=device).unsqueeze(0)

    rows = []
    
    for t in range(48):
        probe = load_probe(t + 1, device, pair_dim=s_z.shape[-1])
        if probe is None:
            with torch.no_grad():
                s_s, s_z = model.trunk.blocks[t](s_s, s_z, mask=None, residue_index=residx)
            continue

        s_z_in = s_z.detach().clone().requires_grad_(True)
        s_s_in = s_s.detach().clone()
        
        s_s_out, s_z_out = model.trunk.blocks[t](s_s_in, s_z_in, mask=None, residue_index=residx)
        
        logits = probe.encode(s_z_out.reshape(-1, s_z.shape[-1]), i_t, j_t, L)
        logits = logits.reshape(1, L, L, 64)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        
        # Loss 1: LJ Energy
        probs = F.softmax(logits, dim=-1)
        ed = (probs * centers).sum(dim=-1)
        d = ed.clamp(min=0.5)
        r = SIGMA / d
        lj_map = 4 * EPS * (r**12 - r**6)
        energy = (lj_map * keep_mask).sum() / L
        
        # Loss 2: Self-Belief (CE against final layer)
        # This asks: "Which way to move z_t to reach the final answer?"
        log_probs = F.log_softmax(logits, dim=-1)
        loss_belief = -(final_probs * log_probs).sum(dim=-1).mean()

        # Loss 3: GT CE
        loss_gt = None
        if gt_coords is not None:
            loss_gt = F.cross_entropy(logits.reshape(-1, 64), gt_bins.reshape(-1))

        # Gradients
        grad_lj = torch.autograd.grad(energy, s_z_in, retain_graph=True)[0]
        grad_bel = torch.autograd.grad(loss_belief, s_z_in, retain_graph=(loss_gt is not None))[0]
        grad_gt = torch.autograd.grad(loss_gt, s_z_in)[0] if loss_gt is not None else None
        
        delta_z = (s_z_out - s_z_in).detach()
        
        row = {
            "layer": t,
            "energy_next": float(energy.detach()),
            "loss_bel_next": float(loss_belief.detach()),
            "cos_neg_lj": cosine_similarity(delta_z, -grad_lj),
            "cos_neg_bel": cosine_similarity(delta_z, -grad_bel),
            "cos_neg_gt": cosine_similarity(delta_z, -grad_gt) if grad_gt is not None else 0,
            "cos_random": cosine_similarity(delta_z, torch.randn_like(delta_z)),
            "norm_delta": float(delta_z.norm()),
            "norm_grad_lj": float(grad_lj.norm()),
        }
        rows.append(row)
        
        c_lj = f"{row['cos_neg_lj']:7.4f}"
        c_bel = f"{row['cos_neg_bel']:7.4f}"
        c_gt = f"{row['cos_neg_gt']:7.4f}" if grad_gt is not None else " N/A  "
        print(f"L{t:02d} | E {float(energy):6.2f} | CosLJ {c_lj} | CosBel {c_bel} | CosGT {c_gt}")

        s_s = s_s_out.detach()
        s_z = s_z_out.detach()

    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG") # Ubiquitin
    parser.add_argument("--pdb", type=str, default="data/1UBQ.pdb")
    parser.add_argument("--out-csv", type=str, default="runs/cosine_backprop_local.csv")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    
    gt_coords = None
    if os.path.exists(args.pdb):
        seq, coords, _ = extract_sequence_and_ca(Path(args.pdb), "")
        args.sequence = seq
        gt_coords = torch.from_numpy(coords).float()
    else:
        print(f"Warning: PDB {args.pdb} not found, skipping GT test.")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to(device)
    model.eval()
    for p in model.parameters(): p.requires_grad = True

    print(f"Running Local Backprop Test (L={len(args.sequence)})")
    rows = run_local_test(args.sequence, model, tokenizer, device, gt_coords)
    
    if not rows:
        print("No results generated.")
        return

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Results written to {args.out_csv}")
    
    m_lj = np.mean([r["cos_neg_lj"] for r in rows])
    m_gt = np.mean([r["cos_neg_gt"] for r in rows]) if gt_coords is not None else 0
    m_rd = np.mean([r["cos_random"] for r in rows])
    print(f"Global Mean Cosine LJ: {m_lj:.6f}")
    print(f"Global Mean Cosine GT: {m_gt:.6f}")
    print(f"Global Random Mean:    {m_rd:.6f}")

if __name__ == "__main__":
    main()
