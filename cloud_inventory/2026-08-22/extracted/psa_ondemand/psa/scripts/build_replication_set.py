"""Build a larger fixed replication set from RCSB.

This creates an independent manifest so the original 53-protein starter set
and results remain untouched:

  data/replication/pdb/<pdb>.cif
  data/replication/physics/<pdb>.npz
  data/replication/manifest.csv

Selection is intentionally conservative:
  - experimental protein structures
  - one protein entity per entry
  - X-ray/electron/neutron structures with resolution <= 2.5 A
  - first protein chain length in [30, 220]

The current physics loader takes the first protein chain in the downloaded
mmCIF. We therefore query monomeric entries to keep this simple and consistent
with the existing pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from psa.dataset import pdb_url
from psa.physics import compute_physics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("build_replication_set")

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "data" / "replication"
PDB_DIR = OUT_ROOT / "pdb"
PHYS_DIR = OUT_ROOT / "physics"
MANIFEST = OUT_ROOT / "manifest.csv"


def rcsb_query(return_limit: int = 2000) -> list[str]:
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "equals",
                        "value": 1,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 2.5,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "greater_or_equal",
                        "value": 30,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "less_or_equal",
                        "value": 220,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": return_limit},
            "sort": [
                {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc",
                }
            ],
            "results_content_type": ["experimental"],
        },
    }
    req = urllib.request.Request(
        "https://search.rcsb.org/rcsbsearch/v2/query",
        data=json.dumps(query).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = json.loads(r.read())
    return [x["identifier"] for x in payload.get("result_set", [])]


def fetch_cif(pdb_id: str) -> Path:
    path = PDB_DIR / f"{pdb_id.lower()}.cif"
    if path.exists() and path.stat().st_size > 0:
        return path
    with urllib.request.urlopen(pdb_url(pdb_id), timeout=60) as r:
        path.write_bytes(r.read())
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-n", type=int, default=300)
    ap.add_argument("--candidate-limit", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=220)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--pdb-id", type=str)
    ap.add_argument("--cif-path", type=str)
    ap.add_argument("--phys-path", type=str)
    args = ap.parse_args()

    if args.worker:
        if not args.cif_path or not args.phys_path or not args.pdb_id:
            raise SystemExit("--worker requires --pdb-id, --cif-path, --phys-path")
        phys = compute_physics(Path(args.cif_path))
        phys.to_npz(Path(args.phys_path))
        print(json.dumps({"pdb_id": args.pdb_id, "L": phys.L, "sequence": phys.sequence}))
        return

    random.seed(args.seed)
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    PHYS_DIR.mkdir(parents=True, exist_ok=True)

    ids = rcsb_query(return_limit=args.candidate_limit)
    random.Random(args.seed).shuffle(ids)
    log.info("RCSB returned %d candidate entries", len(ids))

    rows: list[dict] = []
    seen_seq: set[str] = set()
    failures = 0
    for i, pdb_id in enumerate(ids, 1):
        if len(rows) >= args.target_n:
            break
        prefix = f"[{i:04d}/{len(ids)}] {pdb_id}"
        try:
            cif_path = fetch_cif(pdb_id)
            phys_path = PHYS_DIR / f"{pdb_id.lower()}.npz"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--pdb-id", pdb_id,
                "--cif-path", str(cif_path),
                "--phys-path", str(phys_path),
            ]
            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=90,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            )
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr or proc.stdout)[-1000:])
            meta = json.loads(proc.stdout.strip().splitlines()[-1])
        except (urllib.error.URLError, TimeoutError, subprocess.TimeoutExpired, Exception) as e:
            failures += 1
            log.warning("%s failed: %s", prefix, e)
            continue

        L = int(meta["L"])
        sequence = str(meta["sequence"])
        if not (args.min_len <= L <= args.max_len):
            continue
        if sequence in seen_seq:
            continue
        seen_seq.add(sequence)
        rows.append({
            "pdb_id": pdb_id,
            "chain": "",
            "fold_class": "unknown",
            "description": "rcsb_replication",
            "L": L,
            "sequence": sequence,
            "phys_path": str(phys_path.relative_to(ROOT)),
        })
        if len(rows) % 25 == 0:
            mean_l = sum(int(r["L"]) for r in rows) / len(rows)
            log.info("kept %d/%d target proteins; mean L=%.1f failures=%d",
                     len(rows), args.target_n, mean_l, failures)
        time.sleep(0.03)

    if not rows:
        raise RuntimeError("No replication proteins were built")
    with MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pdb_id", "chain", "fold_class", "description", "L", "sequence", "phys_path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %s with %d proteins", MANIFEST, len(rows))
    log.info("Length range: min=%d max=%d mean=%.1f",
             min(int(r["L"]) for r in rows),
             max(int(r["L"]) for r in rows),
             sum(int(r["L"]) for r in rows) / len(rows))


if __name__ == "__main__":
    main()
