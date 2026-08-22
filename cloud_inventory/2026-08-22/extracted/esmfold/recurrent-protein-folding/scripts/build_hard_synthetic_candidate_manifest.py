#!/usr/bin/env python3
"""Build a candidate manifest for hard synthetic/de-novo protein folds.

This deliberately casts a wider net than build_synthetic_design_manifest.py.
The expensive 47-probe benchmark should run only after an ESMFold-direct
pre-screen identifies actual misses.
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
OUT_DIR = ROOT / "data/hard_synthetic_candidates"
MANIFEST = ROOT / "data/hard_synthetic_candidates_manifest.tsv"
JSONL = ROOT / "data/hard_synthetic_candidates_manifest.jsonl"

KEYWORDS = [
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
]

# Literature/control seeds. 8P4Y is explicitly reported as a designed
# coiled-coil origami fold where AlphaFold2 fails to recover the correct fold.
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
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def rcsb_search(keyword: str, rows: int = 400) -> list[str]:
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
            raise RuntimeError(data["errors"])
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
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read()
        if len(text) < 1000:
            return False
        out_path.write_bytes(text)
        return True
    except Exception as exc:
        print(f"[warn] pdb download failed for {pdb_id}: {exc}")
        return False


def seq_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    matches = sum(x == y for x, y in zip(a[:n], b[:n]))
    return matches / max(len(a), len(b))


def score_row(row: dict, evidence: str, min_release_year: int) -> float:
    title = row["title"]
    source = row["source"]
    year = int(row["release_date"][:4]) if row["release_date"][:4].isdigit() else 0
    score = 0.0
    score += 100 if evidence.startswith("literature_reported") else 0
    score += 8 if re.search(r"(origami|coiled-coil|knot|triangle)", title, re.I) else 0
    score += 5 if re.search(r"(de novo|computational)", title, re.I) else 0
    score += 3 if "synthetic" in source.lower() else 0
    score += 2 if year >= min_release_year else 0
    score += 1 if re.search(r"(repeat|barrel|binder|scaffold)", title, re.I) else 0
    score -= abs(row["length"] - 180) / 300
    return score


def candidate_rows(meta: dict[str, dict], n: int, min_len: int, max_len: int, min_release_year: int) -> list[dict]:
    rows = []
    for pdb_id, entry in meta.items():
        title = (entry.get("struct") or {}).get("title") or ""
        release = ((entry.get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:10]
        methods = ";".join(m.get("method", "") for m in entry.get("exptl") or [])
        entities = entry.get("polymer_entities") or []
        evidence = SEEDS.get(pdb_id.upper(), "candidate_prescreen_required")

        for ent in entities:
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
            designish = bool(DESIGN_RE.search(title) or DESIGN_RE.search(source))
            if not designish and pdb_id.upper() not in SEEDS:
                continue
            row = {
                "name": pdb_id.lower(),
                "pdb_id": pdb_id.upper(),
                "chain": chains[0],
                "length": len(seq),
                "release_date": release,
                "method": methods,
                "title": title,
                "source": source,
                "sequence": seq,
                "evidence": evidence,
            }
            row["score"] = score_row(row, evidence, min_release_year)
            rows.append(row)

    rows.sort(key=lambda r: (r["score"], r["release_date"], r["length"]), reverse=True)
    selected = []
    selected_seqs: list[str] = []
    seen_pdb = set()
    for row in rows:
        if row["pdb_id"] in seen_pdb:
            continue
        if any(seq_identity(row["sequence"], prev) >= 0.80 for prev in selected_seqs):
            continue
        pdb_path = OUT_DIR / f"{row['pdb_id'].lower()}.pdb"
        if not download_pdb(row["pdb_id"], pdb_path):
            continue
        row["gt_pdb"] = str(pdb_path.relative_to(ROOT))
        row["class"] = "hard_synthetic_candidate"
        selected.append(row)
        selected_seqs.append(row["sequence"])
        seen_pdb.add(row["pdb_id"])
        if len(selected) >= n:
            break
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--min-len", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=700)
    ap.add_argument("--min-release-year", type=int, default=2020)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = set(SEEDS)
    for kw in KEYWORDS:
        got = rcsb_search(kw)
        print(f"{kw}: {len(got)} hits")
        ids.update(got)
    ids = sorted(ids)
    print(f"unique RCSB entries: {len(ids)}")
    meta = fetch_metadata(ids)
    rows = candidate_rows(meta, args.n, args.min_len, args.max_len, args.min_release_year)

    with MANIFEST.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["name", "gt_pdb", "chain", "class", "pdb_id", "length", "release_date", "method", "title", "source", "evidence"])
        for r in rows:
            writer.writerow([r["name"], r["gt_pdb"], r["chain"], r["class"], r["pdb_id"], r["length"], r["release_date"], r["method"], r["title"], r["source"], r["evidence"]])

    with JSONL.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {MANIFEST}")
    print(f"wrote {JSONL}")
    for i, r in enumerate(rows, 1):
        print(f"{i:02d} {r['pdb_id']} chain {r['chain']} L={r['length']} score={r['score']:.1f} {r['evidence']} :: {r['title'][:90]}")


if __name__ == "__main__":
    main()
