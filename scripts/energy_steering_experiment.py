#!/usr/bin/env python3
"""Energy attractor experiment.

At trunk block 25, steer z in the direction that LOWERS LJ energy (toward
the LJ minimum) vs. the direction that RAISES it (into the repulsive regime).
Then continue the ESMFold forward pass from block 25 onward.

Hypothesis:
  High-energy steering → trunk has learned to correct → LJ at layer 46 rebounds to baseline
  Low-energy  steering → trunk accepts / continues   → LJ at layer 46 stays low or improves

This would show the trunk's implicit energy landscape has genuine attractors.
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

from recurrent_folding.esmfold_probes import ESMFoldTrunkCapture, PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers

ROOT       = Path(__file__).resolve().parents[1]
PROBES_DIR = ROOT / "runs/general_belief_replay_benchmark20_cw30"
RUNS_DIR   = ROOT / "runs"

SIGMA      = 6.8
EPS        = 1.0
SEQ_SEP    = 3
INJECT_BLOCK_DEFAULT = 24  # overridden by --inject-layer CLI arg

PROTEIN_ACTS = {
    "8fmi":    "benchmark_8fmi/esmfold_activations.pt",
    "T1137s4": "benchmark_T1137s4/esmfold_activations.pt",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def lj_energy_from_probs(probs: torch.Tensor, centers: torch.Tensor,
                          L: int, keep_mask: torch.Tensor) -> float:
    E_d = (probs * centers).sum(-1).clamp(min=0.5)
    r   = SIGMA / E_d
    lj  = 4 * EPS * (r ** 12 - r ** 6)
    return float((lj * keep_mask.float()).sum() / L)


def load_probe(layer: int, pd: int, device: torch.device) -> PairDistogramBottleneck:
    pr = PairDistogramBottleneck(pair_dim=pd, bins=ESMFOLD_DISTO_BINS,
                                 hidden_dim=256).to(device).eval()
    pr.load_state_dict(torch.load(
        PROBES_DIR / f"layer_{layer:02d}" / "probe.pt",
        map_location=device, weights_only=True))
    for p in pr.parameters():
        p.requires_grad_(False)
    return pr


def probe_lj(z: torch.Tensor, layer: int, L: int, i_t: torch.Tensor,
             j_t: torch.Tensor, keep: torch.Tensor, centers: torch.Tensor,
             device: torch.device) -> float:
    pd = z.shape[-1]
    pr = load_probe(layer, pd, device)
    with torch.no_grad():
        logits = pr.encode(z.to(device).reshape(L * L, pd), i_t, j_t, L)
        probs  = F.softmax(logits, dim=-1)
    return lj_energy_from_probs(probs, centers, L, keep)


# ── hooks ─────────────────────────────────────────────────────────────────────

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
        return out   # pass through unmodified


# ── fold with optional injection + multi-layer capture ───────────────────────

def fold_with_hooks(model, tokenizer, seq: str, device: torch.device,
                    inject_block: int | None = None,
                    z_inject: torch.Tensor | None = None,
                    capture_blocks: list[int] | None = None
                    ) -> dict[int, torch.Tensor]:
    """Run ESMFold; optionally inject z at inject_block and capture z at
    each block in capture_blocks.  Returns {block_idx: z_tensor}."""
    ids = tokenizer([seq], return_tensors="pt",
                    add_special_tokens=False)["input_ids"].to(device)

    captures: dict[int, CaptureHook] = {}
    handles = []

    if inject_block is not None and z_inject is not None:
        hook = InjectHook(z_inject)
        handles.append(model.trunk.blocks[inject_block].register_forward_hook(hook))

    for b in (capture_blocks or []):
        cap = CaptureHook()
        captures[b] = cap
        handles.append(model.trunk.blocks[b].register_forward_hook(cap))

    with torch.no_grad():
        model(ids)

    for h in handles:
        h.remove()

    return {b: cap.z for b, cap in captures.items() if cap.z is not None}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject-layer", type=int, default=25,
                    help="1-indexed trunk layer to inject at (default 25)")
    ap.add_argument("--proteins", nargs="+",
                    default=list(PROTEIN_ACTS.keys()))
    args = ap.parse_args()

    INJECT_BLOCK = args.inject_layer - 1   # convert to 0-indexed block
    # Track layers from just after injection through 46
    post = list(range(args.inject_layer + 1, 47))
    # Pick ~6 evenly spaced checkpoints plus always include 46
    step = max(1, len(post) // 5)
    TRACK_LAYERS = sorted(set(post[::step] + [46]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("Loading ESMFold …")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device).eval()
    model.trunk.set_chunk_size(64)

    print(f"Inject block: {INJECT_BLOCK} (layer {args.inject_layer})")
    print(f"Track layers: {TRACK_LAYERS}")

    # blocks to capture in the steered pass (0-indexed) = probe layers - 1
    capture_blocks = [lay - 1 for lay in TRACK_LAYERS]   # e.g. layer 46 → block 45

    all_results: dict[str, dict] = {}

    for prot in args.proteins:
        print(f"\n{'='*50}\n{prot}")
        act_path = RUNS_DIR / PROTEIN_ACTS[prot]
        cache    = torch.load(act_path, map_location="cpu", weights_only=False)
        z_all    = cache["z"].squeeze(1).float()   # (48, L, L, pd)
        seq      = str(cache.get("sequence", ""))
        L        = z_all.shape[1]
        pd       = z_all.shape[-1]
        print(f"  L={L}  pd={pd}  seq={seq[:20]}…")

        # coordinate grids
        i_idx, j_idx = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
        i_t   = i_idx.flatten().to(device)
        j_t   = j_idx.flatten().to(device)
        keep  = (i_t - j_t).abs() > SEQ_SEP
        ctr   = distogram_bin_centers(device, torch.float32).reshape(1, -1)

        # ── baseline LJ at each track layer (from cached activations) ──────
        print("  Baseline LJ trajectory …")
        baseline: dict[int, float] = {}
        for layer in TRACK_LAYERS:
            baseline[layer] = probe_lj(z_all[layer], layer, L, i_t, j_t, keep, ctr, device)
            print(f"    layer {layer:2d}  LJ={baseline[layer]:+.4f}")

        # ── hard-swap steering, preserve residual ────────────────────────────
        # Hard-swap: replace every non-bonded pair's logits with a peaked
        # distribution at a target distance.  This is the strongest possible
        # perturbation while keeping the residual (non-geometric part of z)
        # completely intact — exactly the approach from run_attractor_experiment.
        #
        # LOW  energy target: σ * 2^(1/6) ≈ 7.63 Å  (LJ potential minimum)
        # HIGH energy target: 3.0 Å                  (steep repulsive regime)
        LJ_MIN_D  = SIGMA * (2 ** (1 / 6))   # ≈ 7.63 Å
        HIGH_E_D  = 3.0                       # Å — deep in the repulsive wall

        z25      = z_all[INJECT_BLOCK + 1].clone().to(device)
        z25_flat = z25.reshape(L * L, pd)

        probe25 = load_probe(INJECT_BLOCK + 1, pd, device)

        with torch.no_grad():
            logits_base = probe25.encode(z25_flat, i_t, j_t, L)
            z_recon     = probe25.decode(logits_base, i_t, j_t, L)
            residual    = z25_flat - z_recon      # non-geometric part of z

        lj25 = lj_energy_from_probs(F.softmax(logits_base, dim=-1), ctr, L, keep)
        print(f"  LJ at layer 25 (baseline): {lj25:+.4f}")
        print(f"  Low-energy target: {LJ_MIN_D:.2f} Å (LJ minimum)")
        print(f"  High-energy target: {HIGH_E_D:.2f} Å (repulsive wall)")

        def hard_swap_z(target_d_nonbonded: float, target_d_bonded: float) -> torch.Tensor:
            """Replace logits for non-bonded pairs with a hard peak at target_d,
            leave bonded pairs unchanged.  Add residual back."""
            with torch.no_grad():
                # target distance per pair
                target_per_pair = torch.where(
                    keep,
                    torch.full((L * L,), target_d_nonbonded, device=device),
                    # bonded / diagonal: keep original expected distance
                    (F.softmax(logits_base, dim=-1) * ctr).sum(-1),
                ).clamp(2.3125, 21.6875)

                # find closest bin for each pair
                target_bins = torch.argmin(
                    (ctr.squeeze(0) - target_per_pair.unsqueeze(1)).abs(), dim=1)

                # peaked logits: -10 everywhere, +10 at target bin
                new_logits = torch.full_like(logits_base, -10.0)
                new_logits.scatter_(1, target_bins.unsqueeze(1), 10.0)

                # decode → geometry; add residual back
                z_geom = probe25.decode(new_logits, i_t, j_t, L)
            return (z_geom + residual).reshape(L, L, pd).cpu()

        z_low  = hard_swap_z(LJ_MIN_D, LJ_MIN_D)    # push non-bonded → LJ minimum
        z_high = hard_swap_z(HIGH_E_D, HIGH_E_D)    # push non-bonded → repulsive wall

        # verify the perturbation is strong
        with torch.no_grad():
            for label, z_s in [("low (→7.63Å) ", z_low),
                                ("high (→3.0Å) ", z_high)]:
                lg = probe25.encode(z_s.to(device).reshape(L*L, pd), i_t, j_t, L)
                pr = F.softmax(lg, dim=-1)
                e  = lj_energy_from_probs(pr, ctr, L, keep)
                Ed = (pr * ctr).sum(-1)[keep].mean()
                print(f"    Steered {label}: LJ={e:+.4f}  mean_E[d]={float(Ed):.2f} Å")

        # ── continue forward pass from block 24 with steered z ──────────────
        steered: dict[str, dict[int, float]] = {"low": {}, "high": {}}

        for steer_name, z_inj in [("low", z_low), ("high", z_high)]:
            print(f"  Running {steer_name}-energy steered pass …")
            captured = fold_with_hooks(
                model, tokenizer, seq, device,
                inject_block=INJECT_BLOCK,
                z_inject=z_inj,
                capture_blocks=capture_blocks,
            )
            for b, layer in zip(capture_blocks, TRACK_LAYERS):
                z_cap = captured.get(b)
                if z_cap is None:
                    print(f"    !! no capture for block {b}")
                    continue
                e = probe_lj(z_cap, layer, L, i_t, j_t, keep, ctr, device)
                steered[steer_name][layer] = e
                print(f"    {steer_name} steer, layer {layer:2d}: LJ={e:+.4f}  "
                      f"(baseline={baseline[layer]:+.4f}  "
                      f"Δ={e - baseline[layer]:+.4f})")

        all_results[prot] = {
            "baseline": baseline,
            "low":      steered["low"],
            "high":     steered["high"],
        }

    # ── save JSON ─────────────────────────────────────────────────────────────
    tag = f"inject{args.inject_layer}"
    out_json = RUNS_DIR / f"energy_steering_{tag}.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_json}")

    # ── plot ──────────────────────────────────────────────────────────────────
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    fig.suptitle(
        f"Energy attractor experiment — injection at layer {args.inject_layer} "
        f"({46 - args.inject_layer} blocks of runway)",
        fontsize=13)

    for ax, (prot, res) in zip(axes[0], all_results.items()):
        layers = TRACK_LAYERS
        bl = [res["baseline"][l] for l in layers]
        lo = [res["low"].get(l, float("nan")) for l in layers]
        hi = [res["high"].get(l, float("nan")) for l in layers]

        ax.plot(layers, bl, "k-o", lw=2.2, label="baseline")
        ax.plot(layers, lo, color="#16a34a", marker="o", lw=2.2, label="low-energy steer")
        ax.plot(layers, hi, color="#dc2626", marker="o", lw=2.2, label="high-energy steer")
        ax.axvline(INJECT_BLOCK + 1, color="#64748b", ls="--", lw=1.2,
                   label=f"injection (block {INJECT_BLOCK+1})")
        ax.set_xlabel("Trunk layer", fontsize=11)
        ax.set_ylabel("LJ energy / residue", fontsize=11)
        ax.set_title(prot, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_png = RUNS_DIR / f"energy_steering_{tag}.png"
    fig.savefig(out_png, dpi=160)
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
