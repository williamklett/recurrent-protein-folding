#!/usr/bin/env python3
"""Targeted contact steering experiment.

Identify pairs that the probe currently predicts as contacts (E[d] < contact_thresh).
Steer ONLY those pairs: push to 7.63 Å (LJ minimum) for low-energy, 3.0 Å (repulsive)
for high-energy. Leave all other pairs completely unchanged.

This is the cleanest possible test: we know exactly which pairs we touched,
and we can track whether each steer stuck at the injection point and whether
subsequent trunk blocks correct or accept it.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForProteinFolding

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers

ROOT       = Path(__file__).resolve().parents[1]
PROBES_DIR = ROOT / "runs/general_belief_replay_benchmark20_cw30"
RUNS_DIR   = ROOT / "runs"

SIGMA        = 6.8
EPS          = 1.0
SEQ_SEP      = 3
LJ_MIN_D     = 6.5                       # Å — energy minimum (Cα contact distance)
HIGH_E_D     = 3.0                       # Å — repulsive wall
CONTACT_MAX  = 10.0                      # Å — pairs below this are "contacts"
STEER_DELTA  = 1.5                       # Å — same shift magnitude in both directions

PROTEIN_ACTS = {
    "8fmi":    "benchmark_8fmi/esmfold_activations.pt",
    "T1137s4": "benchmark_T1137s4/esmfold_activations.pt",
}


def load_probe(layer: int, pd: int, device: torch.device) -> PairDistogramBottleneck:
    pr = PairDistogramBottleneck(pair_dim=pd, bins=ESMFOLD_DISTO_BINS,
                                 hidden_dim=256).to(device).eval()
    pr.load_state_dict(torch.load(
        PROBES_DIR / f"layer_{layer:02d}" / "probe.pt",
        map_location=device, weights_only=True))
    for p in pr.parameters():
        p.requires_grad_(False)
    return pr


def lj(Ed: torch.Tensor) -> torch.Tensor:
    r = SIGMA / Ed.clamp(min=0.5)
    return 4 * EPS * (r ** 12 - r ** 6)


class InjectHook:
    def __init__(self, z_steered: torch.Tensor):
        self.z = z_steered
    def __call__(self, _mod, _inp, out):
        s, z = out
        return (s, self.z[None].to(device=z.device, dtype=z.dtype))


class CaptureHook:
    def __init__(self):
        self.z: torch.Tensor | None = None
    def __call__(self, _mod, _inp, out):
        _s, z = out
        self.z = z.detach().squeeze(0).cpu()
        return out


def run_with_hooks(model, tokenizer, seq, device,
                   inject_block=None, z_inject=None,
                   capture_blocks=None):
    ids = tokenizer([seq], return_tensors="pt",
                    add_special_tokens=False)["input_ids"].to(device)
    captures = {b: CaptureHook() for b in (capture_blocks or [])}
    handles = []
    if inject_block is not None and z_inject is not None:
        handles.append(model.trunk.blocks[inject_block].register_forward_hook(
            InjectHook(z_inject)))
    for b, cap in captures.items():
        handles.append(model.trunk.blocks[b].register_forward_hook(cap))
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()
    return {b: cap.z for b, cap in captures.items() if cap.z is not None}


def probe_distances(z: torch.Tensor, layer: int, L: int,
                    i_t, j_t, ctr, device) -> torch.Tensor:
    """Returns E[d] for all L*L pairs."""
    pd = z.shape[-1]
    pr = load_probe(layer, pd, device)
    with torch.no_grad():
        logits = pr.encode(z.to(device).reshape(L * L, pd), i_t, j_t, L)
        probs  = F.softmax(logits, dim=-1)
    return (probs * ctr).sum(-1).reshape(L, L).cpu()   # (L, L)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject-layer", type=int, default=40)
    ap.add_argument("--contact-max", type=float, default=CONTACT_MAX)
    ap.add_argument("--steer-delta", type=float, default=STEER_DELTA,
                    help="Å shift applied equally in both directions")
    ap.add_argument("--proteins", nargs="+", default=list(PROTEIN_ACTS.keys()))
    args = ap.parse_args()

    INJECT_BLOCK  = args.inject_layer - 1
    TRACK_LAYERS  = sorted(set(
        list(range(args.inject_layer + 1, 47, max(1, (46 - args.inject_layer) // 5))) + [46]
    ))
    capture_blocks = [l - 1 for l in TRACK_LAYERS]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  inject_layer={args.inject_layer}  contact_max={args.contact_max} Å")
    print(f"track_layers: {TRACK_LAYERS}")

    print("Loading ESMFold …")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model     = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device).eval()
    model.trunk.set_chunk_size(64)

    all_results = {}

    for prot in args.proteins:
        print(f"\n{'='*55}\n{prot}")
        cache = torch.load(RUNS_DIR / PROTEIN_ACTS[prot],
                           map_location="cpu", weights_only=False)
        z_all = cache["z"].squeeze(1).float()
        seq   = str(cache.get("sequence", ""))
        L     = z_all.shape[1]
        pd    = z_all.shape[-1]

        i_idx, j_idx = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
        i_t   = i_idx.flatten().to(device)
        j_t   = j_idx.flatten().to(device)
        sep   = (i_t - j_t).abs()
        nonbonded = sep > SEQ_SEP
        ctr   = distogram_bin_centers(device, torch.float32).reshape(1, -1)

        # ── identify contacts at injection layer ─────────────────────────────
        probe_inj = load_probe(INJECT_BLOCK + 1, pd, device)
        z_inj     = z_all[INJECT_BLOCK + 1].clone().to(device)
        z_inj_flat = z_inj.reshape(L * L, pd)

        with torch.no_grad():
            logits_base = probe_inj.encode(z_inj_flat, i_t, j_t, L)
            z_recon     = probe_inj.decode(logits_base, i_t, j_t, L)
            residual    = z_inj_flat - z_recon
            Ed_base     = (F.softmax(logits_base, dim=-1) * ctr).sum(-1)  # (L*L,)

        # contact mask: non-bonded AND E[d] < contact_max
        contact_mask = nonbonded & (Ed_base < args.contact_max)
        n_contacts   = int(contact_mask.sum())
        print(f"  Contacts (|i-j|>{SEQ_SEP}, E[d]<{args.contact_max}Å): {n_contacts} / {int(nonbonded.sum())} non-bonded pairs")
        print(f"  Contact E[d]: mean={float(Ed_base[contact_mask].mean()):.2f} Å  "
              f"min={float(Ed_base[contact_mask].min()):.2f}  "
              f"max={float(Ed_base[contact_mask].max()):.2f}")

        # ── build steered z for each condition ──────────────────────────────
        Ed_contacts_mean = float(Ed_base[contact_mask].mean())

        def make_steered_z(target_d: float, label: str) -> torch.Tensor:
            """Hard-swap only contact pairs to target_d; leave all others unchanged."""
            with torch.no_grad():
                new_logits  = logits_base.clone()
                target_bin  = (ctr.squeeze(0) - target_d).abs().argmin()
                new_logits[contact_mask] = -10.0
                new_logits[contact_mask, target_bin] = 10.0

                z_geom = probe_inj.decode(new_logits, i_t, j_t, L)
                z_new  = (z_geom + residual).reshape(L, L, pd).cpu()

                re_logits = probe_inj.encode(z_new.to(device).reshape(L*L, pd), i_t, j_t, L)
                Ed_re     = (F.softmax(re_logits, dim=-1) * ctr).sum(-1)
                achieved  = float(Ed_re[contact_mask].mean())
                print(f"    {label} (target={target_d:.2f}Å): "
                      f"contacts {Ed_contacts_mean:.2f} → {achieved:.2f}Å  "
                      f"(Δ={achieved - Ed_contacts_mean:+.2f}Å)")
            return z_new

        print(f"  Low-energy steer (contacts → {LJ_MIN_D:.2f} Å):")
        z_low  = make_steered_z(LJ_MIN_D, "low ")
        print(f"  High-energy steer (contacts → {HIGH_E_D:.2f} Å):")
        z_high = make_steered_z(HIGH_E_D, "high")

        # ── baseline E[d] at contact pairs across track layers ───────────────
        print("  Baseline contact E[d] at track layers …")
        baseline_contact_Ed = {}
        baseline_lj = {}
        for layer in TRACK_LAYERS:
            Ed_mat = probe_distances(z_all[layer], layer, L, i_t, j_t, ctr, device)
            Ed_flat = Ed_mat.reshape(-1)
            baseline_contact_Ed[layer] = float(Ed_flat[contact_mask.cpu()].mean())
            lj_val = float(lj(Ed_flat[contact_mask.cpu()]).mean())
            baseline_lj[layer] = lj_val
            print(f"    layer {layer:2d}: mean contact E[d]={baseline_contact_Ed[layer]:.2f}Å  LJ={lj_val:+.4f}")

        # ── run steered forward passes ────────────────────────────────────────
        steered = {"low": {}, "high": {}}
        for steer_name, z_inj_s in [("low", z_low), ("high", z_high)]:
            print(f"  Running {steer_name}-energy steered pass …")
            captured = run_with_hooks(model, tokenizer, seq, device,
                                      inject_block=INJECT_BLOCK,
                                      z_inject=z_inj_s,
                                      capture_blocks=capture_blocks)
            for b, layer in zip(capture_blocks, TRACK_LAYERS):
                z_cap = captured.get(b)
                if z_cap is None:
                    continue
                Ed_mat  = probe_distances(z_cap, layer, L, i_t, j_t, ctr, device)
                Ed_flat = Ed_mat.reshape(-1)
                Ed_c    = float(Ed_flat[contact_mask.cpu()].mean())
                lj_c    = float(lj(Ed_flat[contact_mask.cpu()]).mean())
                steered[steer_name][layer] = {"Ed": Ed_c, "lj": lj_c}
                bl_Ed   = baseline_contact_Ed[layer]
                print(f"    {steer_name:4s} steer, layer {layer:2d}: "
                      f"contact E[d]={Ed_c:.2f}Å (baseline={bl_Ed:.2f}Å, Δ={Ed_c-bl_Ed:+.2f})  "
                      f"LJ={lj_c:+.4f}")

        all_results[prot] = {
            "baseline_Ed":  baseline_contact_Ed,
            "baseline_lj":  baseline_lj,
            "low":  steered["low"],
            "high": steered["high"],
            "n_contacts": n_contacts,
        }

    # ── save + plot ───────────────────────────────────────────────────────────
    tag = f"contact_inject{args.inject_layer}"
    out_json = RUNS_DIR / f"contact_steering_{tag}.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_json}")

    n = len(all_results)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 9), squeeze=False)
    fig.suptitle(
        f"Contact-pair steering — inject layer {args.inject_layer} "
        f"(only pairs with E[d]<{args.contact_max}Å steered)",
        fontsize=13)

    for col, (prot, res) in enumerate(all_results.items()):
        layers = TRACK_LAYERS
        # top: E[d] of contact pairs
        ax = axes[0][col]
        ax.plot(layers, [res["baseline_Ed"][l] for l in layers],
                "k-o", lw=2, label="baseline")
        ax.plot(layers, [res["low"][l]["Ed"] for l in layers if l in res["low"]],
                color="#16a34a", marker="o", lw=2, label=f"low steer (→{LJ_MIN_D:.1f}Å)")
        ax.plot(layers, [res["high"][l]["Ed"] for l in layers if l in res["high"]],
                color="#dc2626", marker="o", lw=2, label=f"high steer (→{HIGH_E_D:.1f}Å)")
        ax.axvline(args.inject_layer, color="#64748b", ls="--", lw=1)
        ax.axhline(LJ_MIN_D, color="#16a34a", ls=":", lw=1, alpha=0.5, label="LJ min")
        ax.axhline(HIGH_E_D, color="#dc2626", ls=":", lw=1, alpha=0.5, label="repulsive target")
        ax.set_ylabel("Mean contact E[d] (Å)", fontsize=11)
        ax.set_title(f"{prot}  (n_contacts={res['n_contacts']})", fontsize=12)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        # bottom: LJ of contact pairs
        ax = axes[1][col]
        ax.plot(layers, [res["baseline_lj"][l] for l in layers],
                "k-o", lw=2, label="baseline")
        ax.plot(layers, [res["low"][l]["lj"] for l in layers if l in res["low"]],
                color="#16a34a", marker="o", lw=2, label="low steer")
        ax.plot(layers, [res["high"][l]["lj"] for l in layers if l in res["high"]],
                color="#dc2626", marker="o", lw=2, label="high steer")
        ax.axvline(args.inject_layer, color="#64748b", ls="--", lw=1)
        ax.set_xlabel("Trunk layer", fontsize=11)
        ax.set_ylabel("Mean contact LJ energy", fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_png = RUNS_DIR / f"contact_steering_{tag}.png"
    fig.savefig(out_png, dpi=160)
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
