"""Download the starter-set mmCIFs, run the physics pipeline on each,
and write a manifest CSV that the rest of the pipeline reads.

Outputs:
  data/pdb/<pdb_id>.cif         downloaded mmCIF files
  data/physics/<pdb_id>.npz     computed physical quantities
  data/manifest.csv             one row per successfully processed protein

A protein is skipped (with a logged reason) if:
  * download fails
  * mdtraj fails to parse
  * length is outside [10, 200]
  * physics computation raises

This script is idempotent: cached files are reused, so re-runs are cheap.
"""

from __future__ import annotations

import csv
import logging
import urllib.request
from pathlib import Path

from psa.dataset import STARTER_SET, pdb_url
from psa.physics import compute_physics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("build_starter_set")

ROOT = Path(__file__).resolve().parents[1]
PDB_DIR = ROOT / "data" / "pdb"
PHYS_DIR = ROOT / "data" / "physics"
MANIFEST = ROOT / "data" / "manifest.csv"

MIN_LEN = 10
MAX_LEN = 200


def fetch_cif(pdb_id: str) -> Path:
    path = PDB_DIR / f"{pdb_id.lower()}.cif"
    if path.exists() and path.stat().st_size > 0:
        return path
    log.info("  downloading %s", pdb_id)
    with urllib.request.urlopen(pdb_url(pdb_id)) as r:
        path.write_bytes(r.read())
    return path


def main() -> None:
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    PHYS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n_total = len(STARTER_SET)
    log.info("Processing %d target proteins", n_total)

    for i, target in enumerate(STARTER_SET):
        pdb_id = target.pdb_id
        prefix = f"[{i + 1:>2}/{n_total}] {pdb_id}"
        try:
            cif_path = fetch_cif(pdb_id)
        except Exception as e:
            log.warning("%s  download failed: %s", prefix, e)
            continue

        phys_path = PHYS_DIR / f"{pdb_id.lower()}.npz"
        try:
            phys = compute_physics(cif_path)
        except Exception as e:
            log.warning("%s  physics failed: %s", prefix, e)
            continue

        if not (MIN_LEN <= phys.L <= MAX_LEN):
            log.warning("%s  skip: L=%d outside [%d, %d]", prefix, phys.L, MIN_LEN, MAX_LEN)
            continue

        phys.to_npz(phys_path)
        rows.append(
            {
                "pdb_id": pdb_id,
                "chain": target.chain_id or "",
                "fold_class": target.fold_class,
                "description": target.description,
                "L": phys.L,
                "sequence": phys.sequence,
                "phys_path": str(phys_path.relative_to(ROOT)),
            }
        )
        log.info("%s  OK  L=%-3d  %s", prefix, phys.L, target.description)

    log.info("Writing manifest: %d / %d proteins kept", len(rows), n_total)
    with MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pdb_id", "chain", "fold_class", "description", "L", "sequence", "phys_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Quick fold-class breakdown.
    from collections import Counter

    breakdown = Counter(r["fold_class"] for r in rows)
    log.info("Fold-class breakdown: %s", dict(breakdown))
    log.info("Length range: min=%d  max=%d  mean=%.0f", min(r["L"] for r in rows),
             max(r["L"] for r in rows), sum(r["L"] for r in rows) / len(rows))


if __name__ == "__main__":
    main()
