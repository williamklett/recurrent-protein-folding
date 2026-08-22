#!/usr/bin/env python3
"""Build a ChimeraX cartoon scaffold from a decoded C-alpha trajectory.

The decoded MDS trajectory only determines C-alpha positions.  This script
adds a simple N/CA/C/O backbone scaffold around those C-alpha positions and
maps HELIX/SHEET annotations from a reference PDB so ChimeraX can render a
moving cartoon.  The added atoms are visualization scaffolding, not inferred
chemistry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


AA3 = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


def unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-7:
        return fallback.copy()
    return v / n


def parse_reference(path: Path, chain: str) -> tuple[list[str], list[int], list[str], list[str]]:
    residues: list[tuple[int, str]] = []
    names: list[str] = []
    seen = set()
    records: list[str] = []
    text = path.read_text(errors="ignore").splitlines()

    for line in text:
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == chain:
            key = (int(line[22:26]), line[26])
            if key in seen:
                continue
            seen.add(key)
            residues.append(key)
            resname = line[17:20].strip()
            names.append(resname if resname in AA3 else "GLY")

    for line in text:
        if line.startswith("HELIX") and line[19] == chain:
            records.append(line + "\n")
        elif line.startswith("SHEET") and line[21] == chain:
            records.append(line + "\n")

    return names, [r[0] for r in residues], [r[1] for r in residues], records


def parse_ca_models(path: Path) -> dict[int, np.ndarray]:
    models: dict[int, np.ndarray] = {}
    layer = None
    coords: list[tuple[float, float, float]] = []
    for line in path.open():
        if line.startswith("MODEL"):
            layer = int(line.split()[-1])
            coords = []
        elif line.startswith("ATOM") and layer is not None:
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        elif line.startswith("ENDMDL") and layer is not None:
            models[layer] = np.asarray(coords, dtype=np.float64)
            layer = None
    if not models:
        raise ValueError(f"no MODEL blocks in {path}")
    return models


def kabsch_transform(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(mobile), len(target))
    m = mobile[:n]
    t = target[:n]
    mc = m.mean(axis=0, keepdims=True)
    tc = t.mean(axis=0, keepdims=True)
    u, _s, vt = np.linalg.svd((m - mc).T @ (t - tc))
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return rot, mc, tc


def apply_transform(coords: np.ndarray, rot: np.ndarray, mobile_center: np.ndarray, target_center: np.ndarray) -> np.ndarray:
    return (coords - mobile_center) @ rot + target_center


def parse_reference_ca(path: Path, chain: str) -> np.ndarray:
    coords = []
    seen = set()
    for line in path.open(errors="ignore"):
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == chain:
            key = (int(line[22:26]), line[26])
            if key in seen:
                continue
            seen.add(key)
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(coords, dtype=np.float64)


def scaffold_backbone(ca: np.ndarray) -> dict[str, np.ndarray]:
    n = len(ca)
    tangents = np.zeros_like(ca)
    for i in range(n):
        if i == 0:
            tangents[i] = unit(ca[1] - ca[0], np.array([1.0, 0.0, 0.0]))
        elif i == n - 1:
            tangents[i] = unit(ca[-1] - ca[-2], tangents[i - 1])
        else:
            tangents[i] = unit(ca[i + 1] - ca[i - 1], tangents[i - 1])

    normals = np.zeros_like(ca)
    normals[0] = unit(np.cross(tangents[0], np.array([0.0, 0.0, 1.0])), np.array([0.0, 1.0, 0.0]))
    for i in range(1, n):
        projected = normals[i - 1] - tangents[i] * np.dot(normals[i - 1], tangents[i])
        normals[i] = unit(projected, normals[i - 1])
    binormals = np.asarray([unit(np.cross(t, nrm), np.array([0.0, 0.0, 1.0])) for t, nrm in zip(tangents, normals)])

    return {
        "N": ca - 1.15 * tangents + 0.34 * normals,
        "CA": ca,
        "C": ca + 1.15 * tangents + 0.34 * normals,
        "O": ca + 1.42 * tangents + 0.72 * normals + 0.24 * binormals,
    }


def write_pdb(
    path: Path,
    models: dict[int, np.ndarray],
    resnames: list[str],
    resnums: list[int],
    insertions: list[str],
    sec_records: list[str],
) -> None:
    n = min(len(next(iter(models.values()))), len(resnames))
    lines: list[str] = []
    lines.extend(sec_records)

    for model_id in sorted(models):
        bb = scaffold_backbone(models[model_id][:n])
        lines.append(f"MODEL     {model_id}\n")
        atom_id = 1
        for i in range(n):
            for atom in ("N", "CA", "C", "O"):
                x, y, z = bb[atom][i]
                atom_field = f"{atom:<4s}" if len(atom) == 1 else f"{atom:>4s}"
                element = atom[0]
                lines.append(
                    f"ATOM  {atom_id:5d} {atom_field} {resnames[i]:>3s} A{resnums[i]:4d}{insertions[i]:1s}   "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
                )
                atom_id += 1
        lines.append("ENDMDL\n")
    lines.append("END\n")
    path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ca-stack", type=Path, required=True)
    parser.add_argument("--reference-pdb", type=Path, required=True)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--align-to-reference", action="store_true")
    args = parser.parse_args()

    resnames, resnums, insertions, sec_records = parse_reference(args.reference_pdb, args.chain)
    models = parse_ca_models(args.ca_stack)
    if args.align_to_reference:
        ref_ca = parse_reference_ca(args.reference_pdb, args.chain)
        final = models[max(models)]
        rot, mc, tc = kabsch_transform(final, ref_ca)
        models = {layer: apply_transform(coords, rot, mc, tc) for layer, coords in models.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_pdb(args.out, models, resnames, resnums, insertions, sec_records)
    print(args.out)


if __name__ == "__main__":
    main()
