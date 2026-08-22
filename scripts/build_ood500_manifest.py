#!/usr/bin/env python3
"""Build an error-enriched OOD manifest from RCSB.

The target mix is deliberately synthetic-heavy. The downstream cache step runs
ESMFold direct and supplies the actual error labels, so this script only builds
a broad candidate set with experimental structures.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC_KEYWORDS = [
    "coiled-coil protein origami",
    "protein origami triangle",
    "de novo designed coiled-coil",
    "designed coiled-coil protein",
    "de novo designed repeat protein",
    "computationally designed repeat protein",
    "designed beta barrel protein",
    "de novo designed beta barrel",
    "designed miniprotein synthetic construct",
    "de novo designed protein synthetic construct",
    "computationally designed protein synthetic construct",
    "synthetic construct de novo protein design",
    "designed protein synthetic construct",
    "de novo designed binder",
    "de novo designed scaffold",
    "protein design",
]

NATURAL_KEYWORDS = [
    "CASP target protein",
    "hypothetical protein crystal structure",
    "uncharacterized protein crystal structure",
    "unknown function protein crystal structure",
    "novel fold protein crystal structure",
    "metagenomic protein crystal structure",
    "bacterial protein unknown function",
]

SEEDS = {
    "8P4Y": "literature_reported_af2_failure_ccpo_triangle",
}

DESIGN_RE = re.compile(
    r"(de novo|design|designed|synthetic|miniprotein|origami|coiled-coil|repeat|barrel|scaffold|binder)",
    re.I,
)


def url_json(url: str, body: bytes | None = None) -> dict:
    headers = {"Accept": "application/json", "User-Agent": "recurrent-protein-folding/0.1"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.load(resp)


def rcsb_search(keyword: str, rows: int) -> list[str]:
    query = {
        "query": {"type": "terminal", "service": "full_text", "parameters": {"value": keyword}},
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }
    url = "https://search.rcsb.org/rcsbsearch/v2/query?json=" + urllib.parse.quote(json.dumps(query))
    try:
        data = url_json(url)
    except Exception as exc:
        print(f"[warn] search failed for {keyword!r}: {exc}")
        return []
    return [r["identifier"] for r in data.get("result_set", [])]


def fetch_metadata(ids: list[str], batch_size: int = 80) -> dict[str, dict]:
    gql = """
    query entries($ids:[String!]!) {
      entries(entry_ids:$ids) {
        rcsb_id
        struct { title }
        exptl { method }
        rcsb_accession_info { initial_release_date }
        polymer_entities {
          rcsb_polymer_entity_container_identifiers { auth_asym_ids asym_ids }
          entity_poly { pdbx_seq_one_letter_code_can type }
          rcsb_entity_source_organism { scientific_name }
        }
      }
    }
    """
    out: dict[str, dict] = {}
    for i in range(0, len(ids), batch_size):
        chunk = ids[i : i + batch_size]
        body = json.dumps({"query": gql, "variables": {"ids": chunk}}).encode()
        data = url_json("https://data.rcsb.org/graphql", body=body)
        if data.get("errors"):
            print(f"[warn] metadata errors for batch {i}: {data['errors']}")
        for entry in data.get("data", {}).get("entries", []) or []:
            if entry:
                out[entry["rcsb_id"]] = entry
        time.sleep(0.1)
    return out


def download_pdb(pdb_id: str, out_path: Path) -> bool:
    if out_path.exists() and out_path.stat().st_size > 1000:
        return True
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "recurrent-protein-folding/0.1"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            text = resp.read()
        if len(text) < 1000:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(text)
        return True
    except Exception as exc:
        print(f"[warn] pdb download failed for {pdb_id}: {exc}")
        return False


def seq_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return sum(x == y for x, y in zip(a[:n], b[:n])) / max(len(a), len(b))


def entry_rows(meta: dict[str, dict], min_len: int, max_len: int, synthetic_ids: set[str]) -> list[dict]:
    rows = []
    for pdb_id, entry in meta.items():
        title = (entry.get("struct") or {}).get("title") or ""
        release = ((entry.get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:10]
        methods = ";".join(m.get("method", "") for m in entry.get("exptl") or [])
        for ent in entry.get("polymer_entities") or []:
            poly = ent.get("entity_poly") or {}
            if poly.get("type") != "polypeptide(L)":
                continue
            seq = re.sub(r"[^A-Z]", "X", (poly.get("pdbx_seq_one_letter_code_can") or "").replace("\n", ""))
            if not (min_len <= len(seq) <= max_len):
                continue
            ids = ent.get("rcsb_polymer_entity_container_identifiers") or {}
            chains = ids.get("auth_asym_ids") or ids.get("asym_ids") or []
            if not chains:
                continue
            organisms = [(org or {}).get("scientific_name") or "" for org in (ent.get("rcsb_entity_source_organism") or [])]
            source = ";".join(organisms)
            is_synthetic = pdb_id in synthetic_ids or bool(DESIGN_RE.search(title) or DESIGN_RE.search(source))
            cls = "ood_synthetic_candidate" if is_synthetic else "ood_natural_candidate"
            evidence = SEEDS.get(pdb_id, "rcsb_design_search" if is_synthetic else "rcsb_natural_challenging_search")
            year = int(release[:4]) if release[:4].isdigit() else 0
            score = 0.0
            score += 100 if pdb_id in SEEDS else 0
            score += 12 if is_synthetic else 0
            score += 6 if re.search(r"(origami|coiled-coil|barrel|repeat|knot)", title, re.I) else 0
            score += 4 if re.search(r"(de novo|computational)", title, re.I) else 0
            score += 2 if year >= 2020 else 0
            score -= abs(len(seq) - 180) / 400
            rows.append(
                {
                    "name": pdb_id.lower(),
                    "pdb_id": pdb_id,
                    "chain": chains[0],
                    "class": cls,
                    "length": len(seq),
                    "release_date": release,
                    "method": methods,
                    "title": title,
                    "source": source,
                    "evidence": evidence,
                    "sequence": seq,
                    "score": score,
                }
            )
    return rows


def select(rows: list[dict], target_n: int, synthetic_min: int, out_dir: Path) -> list[dict]:
    out_dir = out_dir.resolve()
    rows.sort(key=lambda r: (r["score"], r["release_date"], r["length"]), reverse=True)
    selected: list[dict] = []
    selected_seqs: list[str] = []
    seen_pdb: set[str] = set()
    failed_pdb: set[str] = set()

    def try_add(row: dict) -> bool:
        if row["pdb_id"] in seen_pdb or row["pdb_id"] in failed_pdb:
            return False
        if any(seq_identity(row["sequence"], prev) >= 0.85 for prev in selected_seqs):
            return False
        pdb_path = out_dir / f"{row['pdb_id'].lower()}.pdb"
        if not download_pdb(row["pdb_id"], pdb_path):
            failed_pdb.add(row["pdb_id"])
            return False
        row = dict(row)
        row["gt_pdb"] = str(pdb_path.relative_to(ROOT.resolve()))
        selected.append(row)
        selected_seqs.append(row["sequence"])
        seen_pdb.add(row["pdb_id"])
        return True

    synthetic = [r for r in rows if r["class"] == "ood_synthetic_candidate"]
    natural = [r for r in rows if r["class"] != "ood_synthetic_candidate"]
    for row in synthetic:
        if sum(r["class"] == "ood_synthetic_candidate" for r in selected) >= synthetic_min:
            break
        try_add(row)
    for row in rows:
        if len(selected) >= target_n:
            break
        try_add(row)
    for row in natural:
        if len(selected) >= target_n:
            break
        try_add(row)
    return selected[:target_n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--synthetic-min", type=int, default=300)
    ap.add_argument("--search-rows", type=int, default=800)
    ap.add_argument("--min-len", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=700)
    ap.add_argument("--out-manifest", type=Path, default=ROOT / "data/ood500_manifest.tsv")
    ap.add_argument("--out-jsonl", type=Path, default=ROOT / "data/ood500_manifest.jsonl")
    ap.add_argument("--out-pdb-dir", type=Path, default=ROOT / "data/ood500_pdbs")
    args = ap.parse_args()

    synthetic_ids = set(SEEDS)
    natural_ids: set[str] = set()
    for kw in SYNTHETIC_KEYWORDS:
        got = rcsb_search(kw, args.search_rows)
        print(f"synthetic {kw}: {len(got)}")
        synthetic_ids.update(got)
    for kw in NATURAL_KEYWORDS:
        got = rcsb_search(kw, args.search_rows)
        print(f"natural {kw}: {len(got)}")
        natural_ids.update(got)

    ids = sorted(synthetic_ids | natural_ids)
    print(f"unique RCSB entries: {len(ids)} synthetic_search={len(synthetic_ids)} natural_search={len(natural_ids)}")
    meta = fetch_metadata(ids)
    rows = entry_rows(meta, args.min_len, args.max_len, synthetic_ids)
    selected = select(rows, args.n, args.synthetic_min, args.out_pdb_dir)

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "gt_pdb", "chain", "class", "pdb_id", "length", "release_date", "method", "title", "source", "evidence"]
    with args.out_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({k: row.get(k, "") for k in fields})
    with args.out_jsonl.open("w") as f:
        for row in selected:
            f.write(json.dumps(row) + "\n")

    n_syn = sum(r["class"] == "ood_synthetic_candidate" for r in selected)
    print(f"wrote {args.out_manifest}")
    print(f"N={len(selected)} synthetic={n_syn} natural={len(selected)-n_syn}")
    for i, row in enumerate(selected[:30], 1):
        print(f"{i:03d} {row['pdb_id']} {row['class']} L={row['length']} score={row['score']:.1f} :: {row['title'][:90]}")


if __name__ == "__main__":
    main()
