"""Per-fold-class residual SVD at the dip (block 9).

Tests whether the high-rank diffuse non-physical content at the dip is
fold-conditional reasoning — if so, the top residual PCs for α-only vs β-only
proteins should be different (low cosine similarity).

We use ALL 53 proteins partitioned by fold class (this is a characterization,
not a generalization test).
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch

from psa.models import PSA
from psa.targets_k2 import K_PAIR_K2, build_pair_targets_k2

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("foldclass_pca")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
ACT_DIR = ROOT / "data" / "activations"
PHYS_DIR = ROOT / "data" / "physics"
RUNS_DIR = ROOT / "runs"

DIP_BLOCK = 9


def load_psa(block_idx: int) -> PSA:
    block_dir = RUNS_DIR / f"k2_block_{block_idx:02d}"
    model = PSA(d_pair=128, d_hidden=128, K=K_PAIR_K2)
    model.load_state_dict(torch.load(block_dir / "model.pt",
                                       map_location="cpu", weights_only=True))
    return model.eval()


def compute_residuals(psa: PSA, pdb_ids: list[str]) -> np.ndarray:
    """Concatenate per-pair residuals h - PSA(h) across the given proteins."""
    res = []
    for pid in pdb_ids:
        act = torch.load(ACT_DIR / f"{pid.lower()}.pt",
                          map_location="cpu", weights_only=False)
        z = act["z"][DIP_BLOCK, 0].float()    # (L, L, 128)
        L = z.shape[0]
        _, keep = build_pair_targets_k2(PHYS_DIR / f"{pid.lower()}.npz")
        if int(keep.numel() ** 0.5) != L:
            continue
        x = z.reshape(L * L, 128)[keep]
        with torch.no_grad():
            _, h_hat = psa(x)
        res.append((x - h_hat).numpy())
    return np.concatenate(res)


def pca(R: np.ndarray, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Return (top-k PCs (k, 128), variance fraction (k,))."""
    Rc = R - R.mean(0, keepdims=True)
    n = Rc.shape[0]
    if n > 60_000:
        idx = np.random.choice(n, 60_000, replace=False)
        Rc = Rc[idx]
    u, s, vt = np.linalg.svd(Rc, full_matrices=False)
    return vt[:k], (s ** 2 / (s ** 2).sum())[:k]


def main() -> None:
    np.random.seed(0)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    by_class: dict[str, list[str]] = {}
    for r in rows:
        by_class.setdefault(r["fold_class"], []).append(r["pdb_id"])
    log.info("Class sizes: %s", {k: len(v) for k, v in by_class.items()})

    psa = load_psa(DIP_BLOCK)
    log.info("Loaded K=2 PSA at block %d", DIP_BLOCK)

    pcs: dict[str, np.ndarray] = {}
    per_class: dict[str, dict] = {}
    for fc, ids in by_class.items():
        if len(ids) < 3:
            log.warning("Skipping %s (only %d proteins)", fc, len(ids))
            continue
        R = compute_residuals(psa, ids)
        vt, var = pca(R, k=20)
        pcs[fc] = vt
        per_class[fc] = {
            "n_proteins": len(ids),
            "n_pairs": int(R.shape[0]),
            "var_top1": float(var[0]),
            "var_top5": float(var[:5].sum()),
            "var_top20": float(var[:20].sum()),
        }
        log.info("  %-12s  n_prot=%2d  n_pairs=%6d  var top1=%.3f top5=%.3f top20=%.3f",
                  fc, per_class[fc]["n_proteins"], per_class[fc]["n_pairs"],
                  per_class[fc]["var_top1"], per_class[fc]["var_top5"],
                  per_class[fc]["var_top20"])

    classes = list(pcs.keys())
    sim_top1 = np.zeros((len(classes), len(classes)))
    sim_subspace5 = np.zeros((len(classes), len(classes)))
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            v1 = pcs[c1][0]
            v2 = pcs[c2][0]
            sim_top1[i, j] = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
            # Subspace similarity: mean abs cosine of all pairs among top-5
            cs = np.array([[np.abs(np.dot(pcs[c1][a], pcs[c2][b]) /
                                    (np.linalg.norm(pcs[c1][a]) * np.linalg.norm(pcs[c2][b]) + 1e-9))
                             for b in range(5)] for a in range(5)])
            # Average max along rows = "how aligned is each top PC of c1 with the best match in c2"
            sim_subspace5[i, j] = float(cs.max(axis=1).mean())

    log.info("\nClasses: %s", classes)
    log.info("Top-1 PC cosine similarity matrix:")
    log.info("            " + "  ".join(f"{c:>10}" for c in classes))
    for i, c in enumerate(classes):
        log.info("  %-10s" % c + "  ".join(f"{sim_top1[i, j]:>+10.3f}" for j in range(len(classes))))
    log.info("\nTop-5 subspace alignment (mean best-match cos):")
    log.info("            " + "  ".join(f"{c:>10}" for c in classes))
    for i, c in enumerate(classes):
        log.info("  %-10s" % c + "  ".join(f"{sim_subspace5[i, j]:>+10.3f}" for j in range(len(classes))))

    out = {
        "dip_block": DIP_BLOCK,
        "classes": classes,
        "per_class": per_class,
        "top1_cosine": sim_top1.tolist(),
        "top5_subspace_alignment": sim_subspace5.tolist(),
        "top_pcs": {c: pcs[c].tolist() for c in classes},
    }
    out_path = RUNS_DIR / "foldclass_residuals.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
