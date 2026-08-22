#!/usr/bin/env python3
"""Measure whether ubq distogram momentum is correlated between blocks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers


def load_probe(path: Path, device: torch.device) -> PairDistogramBottleneck:
    probe = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
    probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation-cache", type=Path, default=Path("runs/benchmark_ubq/esmfold_activations.pt"))
    ap.add_argument("--probes-root", type=Path, default=Path("runs/general_belief_replay_benchmark20_cw30"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/momentum_steer_ubq_allblocks_fine_v1"))
    ap.add_argument("--seq-sep", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    cache = torch.load(args.activation_cache, map_location="cpu", weights_only=False)
    z_all = cache["z"].squeeze(1).float()
    length = int(z_all.shape[1])
    centers = distogram_bin_centers(device, torch.float32)
    ii, jj = torch.meshgrid(torch.arange(length, device=device), torch.arange(length, device=device), indexing="ij")
    ii_f, jj_f = ii.reshape(-1), jj.reshape(-1)
    mask = ((ii_f - jj_f).abs() > args.seq_sep).detach().cpu().numpy()

    eds: dict[int, np.ndarray] = {}
    for layer in range(1, 48):
        probe_path = args.probes_root / f"layer_{layer:02d}" / "probe.pt"
        if not probe_path.exists():
            continue
        probe = load_probe(probe_path, device)
        with torch.no_grad():
            logits = probe.encode(z_all[layer].to(device).reshape(length * length, 128), ii_f, jj_f, length)
            ed = (F.softmax(logits, -1) * centers).sum(-1).detach().cpu().numpy()
        eds[layer] = ed
        del probe, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    momenta: dict[int, np.ndarray] = {}
    for layer in sorted(eds):
        prev = layer - 1
        if prev in eds:
            momenta[layer] = eds[layer] - eds[prev]

    layers = sorted(momenta)
    rows = []
    for a, b in zip(layers[:-1], layers[1:]):
        x = momenta[a][mask]
        y = momenta[b][mask]
        pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-9 and np.std(y) > 1e-9 else float("nan")
        cosine = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        rows.append(
            {
                "momentum_a": a,
                "momentum_b": b,
                "pearson": pearson,
                "cosine": cosine,
                "same_sign_frac": float(np.mean(np.sign(x) == np.sign(y))),
                "mean_abs_a": float(np.mean(np.abs(x))),
                "mean_abs_b": float(np.mean(np.abs(y))),
            }
        )

    matrix = np.stack([momenta[layer][mask] for layer in layers])
    corr = np.corrcoef(matrix)
    np.save(args.out_dir / "ubq_momentum_correlation_matrix.npy", corr)
    with (args.out_dir / "ubq_momentum_adjacent_correlations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "momentum_layers": layers,
        "num_momentum_fields": len(layers),
        "adjacent_pearson_mean": float(np.nanmean([r["pearson"] for r in rows])),
        "adjacent_pearson_median": float(np.nanmedian([r["pearson"] for r in rows])),
        "adjacent_pearson_min": float(np.nanmin([r["pearson"] for r in rows])),
        "adjacent_pearson_max": float(np.nanmax([r["pearson"] for r in rows])),
        "adjacent_cosine_mean": float(np.nanmean([r["cosine"] for r in rows])),
        "same_sign_mean": float(np.nanmean([r["same_sign_frac"] for r in rows])),
    }
    (args.out_dir / "ubq_momentum_correlation_summary.json").write_text(json.dumps(summary, indent=2))

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    xs = [r["momentum_a"] for r in rows]
    axes[0].plot(xs, [r["pearson"] for r in rows], marker="o", lw=2.2, label="Pearson")
    axes[0].plot(xs, [r["cosine"] for r in rows], marker="s", lw=1.8, label="Cosine")
    axes[0].axhline(0, color="#777", lw=1)
    axes[0].set_title("Adjacent Momentum Alignment")
    axes[0].set_xlabel("momentum field starting layer")
    axes[0].set_ylabel("correlation with next momentum field")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.18)

    im = axes[1].imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    axes[1].set_title("Momentum Correlation Matrix")
    axes[1].set_xlabel("momentum layer")
    axes[1].set_ylabel("momentum layer")
    step = max(1, len(layers) // 8)
    tick_idx = list(range(0, len(layers), step))
    axes[1].set_xticks(tick_idx, [str(layers[i]) for i in tick_idx])
    axes[1].set_yticks(tick_idx, [str(layers[i]) for i in tick_idx])
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.savefig(args.out_dir / "ubq_momentum_correlations.png", dpi=220, bbox_inches="tight")

    print(json.dumps(summary, indent=2))
    print(args.out_dir / "ubq_momentum_correlations.png")


if __name__ == "__main__":
    main()
