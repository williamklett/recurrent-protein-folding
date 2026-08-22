#!/usr/bin/env python3
"""Standard Backprop Cosine Alignment Test.

Tests if the ESMFold trunk update Δz_t aligns with −∇_{z_t} LJ(distogram_final).
Uses full autograd backpropagation through the trunk.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForProteinFolding

# Constants for LJ energy
SIGMA = 6.8
EPS = 1.0
SEQ_SEP = 3

def get_distogram_bin_centers(device):
    # ESMFold uses 64 bins from 2.3125 to 21.6875
    start = 2.3125
    end = 21.6875
    bins = 64
    width = (end - start) / bins
    centers = torch.linspace(start + width/2, end - width/2, bins, device=device)
    return centers

class ESMFoldBackpropCapture:
    def __init__(self, model):
        self.model = model
        # activations[recycle_idx][block_idx] = (s, z)
        # block_idx 0 is input to first block, 1..48 are outputs of blocks
        self.activations = []
        self._current_recycle = -1
        self._handles = []

    def _make_input_hook(self):
        def hook(_module, args):
            # This is called at the start of the trunk in each recycle
            self._current_recycle += 1
            self.activations.append({})
            s, z = args[0], args[1]
            # Clone and retain grad to avoid in-place modification issues
            s_cap = s.clone().detach().requires_grad_(True)
            z_cap = z.clone().detach().requires_grad_(True)
            self.activations[self._current_recycle][0] = (s_cap, z_cap)
            # We actually need to return these or hook the output to use them.
            # But ESMFold trunk blocks are complex.
            # A better way is to just let the forward pass happen and use retain_grad on the REAL tensors.
        return hook

    def _make_output_hook(self, block_idx: int):
        def hook(_module, _args, output):
            s, z = output
            # We store the REAL tensors from the graph
            self.activations[self._current_recycle][block_idx + 1] = (s, z)
            s.retain_grad()
            z.retain_grad()
        return hook
    
    # Actually, hooking PRE-hook on the first block to get input
    def _make_block_pre_hook(self, block_idx: int):
        def hook(_module, args):
            s, z = args[0], args[1]
            if block_idx == 0:
                self._current_recycle += 1
                self.activations.append({})
            
            self.activations[self._current_recycle][block_idx] = (s, z)
        return hook

    def __enter__(self):
        for i, block in enumerate(self.model.trunk.blocks):
            self._handles.append(block.register_forward_pre_hook(self._make_block_pre_hook(i)))
        # Also hook the final output of the last block in the last recycle
        def final_hook(_module, _args, output):
            s, z = output
            self.activations[self._current_recycle][48] = (s, z)
        self._handles.append(self.model.trunk.blocks[-1].register_forward_hook(final_hook))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

def cosine_similarity(a, b):
    a = a.detach().flatten()
    b = b.detach().flatten()
    na, nb = a.norm(), b.norm()
    if float(na) < 1e-12 or float(nb) < 1e-12:
        return 0.0
    return float((a * b).sum() / (na * nb))

def run_test(sequence, model, tokenizer, device, gt_coords=None):
    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False).to(device)
    
    centers = get_distogram_bin_centers(device)
    L = len(sequence)
    i_idx, j_idx = torch.meshgrid(torch.arange(L, device=device), torch.arange(L, device=device), indexing="ij")
    sep = (j_idx - i_idx).abs()
    keep_mask = (sep > SEQ_SEP).float()

    capture = ESMFoldBackpropCapture(model)
    
    # Forward pass
    with capture:
        output = model(**inputs, num_recycles=0)
        
    logits = output.distogram_logits
    
    # LJ Energy
    probs = F.softmax(logits, dim=-1)
    ed = (probs * centers).sum(dim=-1)
    d = ed.clamp(min=0.5)
    r = SIGMA / d
    lj_map = 4 * EPS * (r**12 - r**6)
    energy = (lj_map * keep_mask).sum() / L
    print(f"Final LJ Energy: {float(energy):.4f}")

    # Compute grads wrt z_t using autograd.grad to save memory over retain_grad
    recycle_acts = capture.activations[0]
    z_list = [recycle_acts[t][1] for t in range(48) if t in recycle_acts]
    
    print("Computing gradients wrt activations...")
    grads_lj_list = torch.autograd.grad(energy, z_list, retain_graph=(gt_coords is not None))
    grads_lj = {t: g for t, g in zip(range(48), grads_lj_list)}
    
    # Optional: GT Loss
    grads_gt = {}
    if gt_coords is not None:
        gt_dist = torch.linalg.norm(gt_coords[:, None] - gt_coords[None, :], dim=-1).to(device)
        bin_width = (21.6875 - 2.3125) / 64
        gt_bins = ((gt_dist - 2.3125) / bin_width).long().clamp(0, 63)
        loss_gt = F.cross_entropy(logits.reshape(-1, 64), gt_bins.reshape(-1))
        print(f"Final GT Loss: {float(loss_gt):.4f}")
        grads_gt_list = torch.autograd.grad(loss_gt, z_list)
        grads_gt = {t: g for t, g in zip(range(48), grads_gt_list)}

    rows = []
    for t in range(48):
        if t not in recycle_acts or (t+1) not in recycle_acts:
            continue
        
        z_t, z_next = recycle_acts[t][1], recycle_acts[t+1][1]
        delta_z = z_next - z_t
        
        row = {"layer": t, "norm_delta": float(delta_z.norm().detach())}
        if t in grads_lj:
            row["cos_neg_lj"] = cosine_similarity(delta_z, -grads_lj[t])
            row["norm_grad_lj"] = float(grads_lj[t].norm().detach())
        if t in grads_gt:
            row["cos_neg_gt"] = cosine_similarity(delta_z, -grads_gt[t])
            row["norm_grad_gt"] = float(grads_gt[t].norm().detach())
            
        row["cos_random"] = cosine_similarity(delta_z, torch.randn_like(delta_z))
        rows.append(row)
        
    return rows

import sys
from collections import Counter

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}

def extract_sequence_and_ca(pdb_path: Path, chain_hint: str) -> tuple[str, np.ndarray, str]:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG") # Ubiquitin
    parser.add_argument("--pdb", type=str, help="Path to GT PDB for GT alignment test")
    parser.add_argument("--out-csv", type=str, default="runs/cosine_backprop_standard.csv")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    
    gt_coords = None
    if args.pdb:
        seq, coords, _ = extract_sequence_and_ca(Path(args.pdb), "")
        args.sequence = seq
        gt_coords = torch.from_numpy(coords).float()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to(device)
    model.eval()
    for p in model.parameters(): p.requires_grad = True

    print(f"Running backprop test for L={len(args.sequence)}")
    rows = run_test(args.sequence, model, tokenizer, device, gt_coords)
    
    if not rows:
        print("No results generated.")
        return

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Results written to {args.out_csv}")
    
    # Summary
    if "cos_neg_lj" in rows[0]:
        print(f"Mean Cosine LJ: {np.mean([r['cos_neg_lj'] for r in rows]):.6f}")
    if "cos_neg_gt" in rows[0]:
        print(f"Mean Cosine GT: {np.mean([r['cos_neg_gt'] for r in rows]):.6f}")
    print(f"Mean Random:    {np.mean([r['cos_random'] for r in rows]):.6f}")

    print("\nLayer | Cos(Δz, -∇LJ) | Cos(Δz, -∇GT) | ||Δz|| | ||∇LJ|| | ||∇GT||")
    print("------|---------------|---------------|--------|---------|---------")
    for r in rows[:5] + [{"layer": "..."}] + rows[-5:]:
        if isinstance(r["layer"], str):
            print("...")
            continue
        c_lj = f"{r.get('cos_neg_lj', 0):13.4f}"
        c_gt = f"{r.get('cos_neg_gt', 0):13.4f}"
        n_lj = f"{r.get('norm_grad_lj', 0):7.2f}"
        n_gt = f"{r.get('norm_grad_gt', 0):7.2f}"
        print(f"{r['layer']:5d} | {c_lj} | {c_gt} | {r['norm_delta']:6.2f} | {n_lj} | {n_gt}")

if __name__ == "__main__":
    main()
