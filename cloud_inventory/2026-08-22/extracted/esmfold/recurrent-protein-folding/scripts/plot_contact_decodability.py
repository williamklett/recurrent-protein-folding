#!/usr/bin/env python3
"""Contact decodability vs layer, broken down by sequence separation.

For each layer t in [0..46]:
  - Run cw30 probe on captured z_t → distogram distribution per (i,j)
  - P(contact) = sum of bins where bin_center < CONTACT_THRESH
  - Compute AUROC vs GT contacts (GT Cα distance < CONTACT_THRESH)
  - Separate by sequence separation:
      short:  6  ≤ |i-j| ≤ 11
      medium: 12 ≤ |i-j| ≤ 23
      long:   |i-j| ≥ 24

Replicates and extends baulab.info (arXiv 2602.06020) Fig showing
spatial information emerges in late blocks, long-range last.

Outputs:  contact_decodability_by_layer.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, EsmForProteinFolding

from recurrent_folding.esmfold_probes import ESMFoldTrunkCapture, PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers


ROOT    = Path(__file__).resolve().parents[1]
OOD_DIR = ROOT / "runs/general_convergence_ood500_native_cw30_stream"
PROBES  = ROOT / "runs/general_belief_replay_benchmark20_cw30"

CONTACT_THRESH = 8.0   # Å, standard contact definition
N_PROTEINS     = 8     # how many proteins to average over
MIN_TM         = 0.80  # only well-folded proteins

# Sequence-separation ranges (skip |i-j| < 6, trivially close)
RANGES = [
    ("short  (6–11)",  6,  11),
    ("medium (12–23)", 12, 23),
    ("long   (≥24)",   24, 9999),
]
RANGE_COLORS = ["#2563eb", "#16a34a", "#dc2626"]


# ── helpers ───────────────────────────────────────────────────────────────────
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-scores)
    tpr, fpr = 0.0, 0.0
    auc = 0.0
    prev_fpr = 0.0
    for i in order:
        if labels[i]:
            tpr += 1 / n_pos
        else:
            fpr += 1 / n_neg
            auc += tpr * (fpr - prev_fpr)
            prev_fpr = fpr
    return float(auc)


def pairwise_dist(ca: np.ndarray) -> np.ndarray:
    d = ca[:, None] - ca[None, :]
    return np.sqrt((d * d).sum(-1))


def load_probe(layer: int, device: torch.device) -> PairDistogramBottleneck:
    path = PROBES / f"layer_{layer:02d}" / "probe.pt"
    pr = PairDistogramBottleneck(
        pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256
    ).to(device).eval()
    pr.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    for p in pr.parameters():
        p.requires_grad_(False)
    return pr


def load_gt_ca(name: str) -> np.ndarray:
    p = OOD_DIR / f"{name}_analysis" / "run_artifacts/gt_ca_distances.npz"
    return np.load(p)["ca_coords"].astype(float)


def get_summary(ood_dir: Path) -> dict:
    import csv
    return {r["protein"]: r for r in csv.DictReader(
        (ood_dir / "general_convergence_summary.csv").open())}


# ── pick proteins ─────────────────────────────────────────────────────────────
def get_seq(name: str) -> str:
    """Read sequence from .npz (preferred) or source.json fallback."""
    gt_p  = OOD_DIR / f"{name}_analysis" / "run_artifacts/gt_ca_distances.npz"
    src_p = OOD_DIR / f"{name}_analysis" / "source.json"
    if gt_p.exists():
        data = np.load(gt_p, allow_pickle=True)
        seq_raw = data.get("sequence", None)
        if seq_raw is not None:
            seq = str(seq_raw)
            if seq and seq != "None":
                return seq
    if src_p.exists():
        return json.loads(src_p.read_text()).get("sequence", "")
    return ""


def select_proteins(n: int, min_tm: float, min_L: int = 60, max_L: int = 200) -> list[dict]:
    summary = get_summary(OOD_DIR)
    candidates = []
    for name, row in summary.items():
        try:
            tm = float(row["tm_score"])
        except (ValueError, KeyError):
            continue
        if tm < min_tm:
            continue
        gt_p = OOD_DIR / f"{name}_analysis" / "run_artifacts/gt_ca_distances.npz"
        if not gt_p.exists():
            continue
        seq = get_seq(name)
        if not seq:
            continue
        L = len(seq)
        if not (min_L <= L <= max_L):
            continue
        candidates.append({"name": name, "tm": tm, "seq": seq, "L": L})
    candidates.sort(key=lambda x: -x["tm"])
    return candidates[:n]


# ── main computation ──────────────────────────────────────────────────────────
def compute_decodability(proteins: list[dict], device: torch.device) -> dict:
    """Returns {range_label: array of shape (47,)} with mean AUROC per layer."""
    print(f"Loading ESMFold...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", low_cpu_mem_usage=True
    ).to(device).eval()
    if device.type == "cuda":
        model.esm = model.esm.half()
    model.trunk.set_chunk_size(64)
    capture = ESMFoldTrunkCapture(model)
    centers = distogram_bin_centers(device, torch.float32)  # (n_bins,)

    # accumulate per-layer, per-range AUROC across proteins
    # shape: (47, n_ranges, n_proteins)
    n_layers = 47
    all_aucs = np.full((n_layers, len(RANGES), len(proteins)), np.nan)

    for pi, prot in enumerate(proteins):
        name, seq = prot["name"], prot["seq"]
        print(f"  [{pi+1}/{len(proteins)}] {name}  L={prot['L']}  TM={prot['tm']:.3f}")

        # fold and capture all z
        ids = tokenizer([seq], return_tensors="pt",
                        add_special_tokens=False)["input_ids"].to(device)
        with capture.capturing(), torch.no_grad():
            model(ids)
        z_all = capture.stacked_z().squeeze(1).float()  # (48, L, L, 128)

        # GT contacts
        gt_ca = load_gt_ca(name)
        L = min(len(seq), len(gt_ca))
        D_gt = pairwise_dist(gt_ca[:L])  # (L, L)

        # upper-triangle indices, excluding diagonal and |i-j| < 1
        ii, jj = np.triu_indices(L, k=1)
        sep = jj - ii  # sequence separation for each pair

        gt_contact = (D_gt[ii, jj] < CONTACT_THRESH).astype(bool)

        # build separation masks for each range
        range_masks = []
        for _, lo, hi in RANGES:
            range_masks.append((sep >= lo) & (sep <= hi))

        # process each layer
        for t in range(n_layers):
            z_t = z_all[t]  # (L_model, L_model, 128); may be larger than L

            # run probe in chunks to save memory
            ii_t = torch.from_numpy(ii).long().to(device)
            jj_t = torch.from_numpy(jj).long().to(device)
            pairs_flat = z_t[ii_t, jj_t].to(device)

            probe = load_probe(t, device)
            with torch.no_grad():
                logits = probe.encode(pairs_flat, ii_t, jj_t, L)
                probs = torch.softmax(logits, dim=-1)  # (N_pairs, n_bins)
                # P(contact) = sum of bins where center < threshold
                contact_mask_bins = (centers < CONTACT_THRESH).float()  # (n_bins,)
                p_contact = (probs * contact_mask_bins).sum(-1).cpu().numpy()
            del probe  # free GPU memory

            for ri, (_, _, _) in enumerate(RANGES):
                m = range_masks[ri]
                if m.sum() > 0:
                    all_aucs[t, ri, pi] = auroc(p_contact[m], gt_contact[m])

        del z_all
        torch.cuda.empty_cache()

    # mean over proteins (nanmean)
    mean_aucs = np.nanmean(all_aucs, axis=2)  # (47, n_ranges)
    return {RANGES[ri][0]: mean_aucs[:, ri] for ri in range(len(RANGES))}


# ── plot ──────────────────────────────────────────────────────────────────────
def plot(aucs_by_range: dict, n_proteins: int, out: Path) -> None:
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })

    layers = np.arange(47)
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.12)

    fig.suptitle(
        "Contact decodability from ESMFold pair representations — by sequence separation",
        x=0.08, y=0.96, ha="left", fontsize=16, fontweight="bold", color="#172033")
    fig.text(
        0.08, 0.895,
        f"Contact = GT Cα distance < {CONTACT_THRESH}Å.  "
        f"P(contact) from cw30 distogram probe at each layer.  "
        f"Averaged over {n_proteins} well-folded proteins (TM ≥ {MIN_TM}).  "
        "Replicates baulab.info (arXiv 2602.06020) layer-wise geometry emergence.",
        ha="left", fontsize=10, color="#475569")

    for (label, _, _), color in zip(RANGES, RANGE_COLORS):
        auc = aucs_by_range[label]
        ax.plot(layers, auc, color=color, lw=2.4, label=label)
        ax.scatter(layers, auc, color=color, s=28, zorder=3)

    ax.axhline(0.5, color="#94a3b8", lw=1.2, ls="--", alpha=0.7, label="random (0.50)")
    ax.axvline(7,  color="#e2e8f0", lw=1.0, ls=":", alpha=0.8)
    ax.axvline(20, color="#e2e8f0", lw=1.0, ls=":", alpha=0.8)
    ax.text(3.5,  0.51, "early\n(biochem)", fontsize=8, color="#94a3b8", ha="center")
    ax.text(13.5, 0.51, "mid",               fontsize=8, color="#94a3b8", ha="center")
    ax.text(33,   0.51, "late\n(geometric)",  fontsize=8, color="#94a3b8", ha="center")

    ax.set_xlabel("ESMFold trunk layer", fontsize=12)
    ax.set_ylabel("AUROC  (contact vs non-contact)", fontsize=12)
    ax.set_title("Long-range contacts emerge latest — short-range decodable from early layers",
                 fontsize=12)
    ax.set_xlim(-0.5, 46.5)
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, lw=0.5)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-dir",     type=Path, default=OOD_DIR)
    ap.add_argument("--probes-root", type=Path, default=PROBES)
    ap.add_argument("--out",         type=Path,
                    default=OOD_DIR / "contact_decodability_by_layer.png")
    ap.add_argument("--n-proteins",  type=int, default=N_PROTEINS)
    ap.add_argument("--min-tm",      type=float, default=MIN_TM)
    ap.add_argument("--device",      default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    proteins = select_proteins(args.n_proteins, args.min_tm)
    print(f"Selected {len(proteins)} proteins:")
    for p in proteins:
        print(f"  {p['name']}  L={p['L']}  TM={p['tm']:.3f}")

    aucs = compute_decodability(proteins, device)

    print("\n=== AUROC by layer (mean across proteins) ===")
    print(f"{'layer':>6}", end="")
    for label, *_ in RANGES:
        print(f"  {label:>18}", end="")
    print()
    for t in range(47):
        print(f"{t:>6}", end="")
        for label, *_ in RANGES:
            print(f"  {aucs[label][t]:>18.4f}", end="")
        print()

    plot(aucs, len(proteins), args.out)


if __name__ == "__main__":
    main()
