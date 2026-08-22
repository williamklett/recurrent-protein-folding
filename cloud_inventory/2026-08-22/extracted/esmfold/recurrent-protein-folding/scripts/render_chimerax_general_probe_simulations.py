#!/usr/bin/env python3
"""Render ChimeraX cartoon movies for general-probe decoded trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHIMERAX = Path("/Applications/ChimeraX-1.11.1.app/Contents/MacOS/ChimeraX")

REFS = {
    "ubq": Path("data/1UBQ.pdb"),
    "8fmi": Path("data/8FMI.pdb"),
    "6wxo": Path("data/hard_synthetic_candidates/6wxo.pdb"),
    "8qad": Path("data/hard_synthetic_candidates/8qad.pdb"),
    "9r2b": Path("data/hard_synthetic_candidates/9r2b.pdb"),
    "T1137s4": Path("data/casp15/T1137/T1137s4.pdb"),
    "T1137s6": Path("data/casp15/T1137/T1137s6.pdb"),
}


def q(path: Path) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def model_count(path: Path) -> int:
    return sum(1 for line in path.open() if line.startswith("MODEL"))


def ca_count_in_stack(path: Path) -> int:
    count = 0
    in_first = False
    for line in path.open():
        if line.startswith("MODEL"):
            in_first = True
        elif line.startswith("ENDMDL") and in_first:
            return count
        elif in_first and line.startswith("ATOM"):
            count += 1
    return count


def chain_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    seen = set()
    for line in path.open(errors="ignore"):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            chain = line[21]
            key = (chain, int(line[22:26]), line[26])
            if key in seen:
                continue
            seen.add(key)
            counts[chain] = counts.get(chain, 0) + 1
    return counts


def pick_chain(ref_pdb: Path, stack_ca_count: int) -> str:
    counts = chain_counts(ref_pdb)
    if not counts:
        raise ValueError(f"no CA atoms in {ref_pdb}")
    return min(counts, key=lambda chain: abs(counts[chain] - stack_ca_count))


def load_metrics(name: str) -> tuple[str, str]:
    analysis = ROOT / "runs/general_convergence_n40_native" / f"{name}_analysis"
    metrics_path = analysis / "esmfold_direct_eval/metrics.json"
    corr_path = analysis / "layerwise_distance_correlations.csv"
    left = ""
    right = ""
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        left = f"MAE {m['distance_mae_all_pairs']:.2f} A   pLDDT {m['mean_plddt']:.3f}"
    if corr_path.exists():
        rows = {int(r["layer"]): r for r in csv.DictReader(corr_path.open())}
        late = [float(rows[layer]["corr_vs_47"]) for layer in range(35, 47) if layer in rows]
        if late:
            right = f"min late convergence {min(late):.4f}"
    return left, right


def secondary_color_commands(scaffold: Path) -> list[str]:
    commands: list[str] = []
    for line in scaffold.read_text(errors="ignore").splitlines():
        if line.startswith("HELIX"):
            start_chain = line[19].strip() or "A"
            end_chain = line[31].strip() or start_chain
            if start_chain != end_chain:
                continue
            start = int(line[21:25])
            end = int(line[33:37])
            commands.append(f"color #1/{start_chain}:{start}-{end} #2563eb cartoons")
        elif line.startswith("SHEET"):
            start_chain = line[21].strip() or "A"
            end_chain = line[32].strip() or start_chain
            if start_chain != end_chain:
                continue
            start = int(line[22:26])
            end = int(line[33:37])
            commands.append(f"color #1/{start_chain}:{start}-{end} #38bdf8 cartoons")
    return commands


def write_cxc(path: Path, scaffold: Path, color_commands: list[str], out_mp4: Path, title: str, label_left: str, label_right: str, n_models: int, width: int, height: int) -> None:
    frames = n_models * 2
    path.write_text(
        "\n".join(
            [
                f"open {q(scaffold)} coordsets true",
                "coordset #1 1",
                "cartoon #1",
                "cartoon style helix width 1.45 thickness 0.18 xsection rectangle",
                "cartoon style strand width 1.55 thickness 0.18 xsection rectangle",
                "cartoon style coil width 0.42 thickness 0.42 xsection oval",
                "color #1 #8c939c cartoons",
                *color_commands,
                "lighting soft",
                "set bgColor white",
                "camera ortho",
                "view #1",
                "zoom 0.82",
                "turn x -8",
                "turn y 18",
                f"windowsize {width} {height}",
                f"2dlabels create title text '{title}' xpos .035 ypos .955 size 32 color #17202a",
                "2dlabels create subtitle text 'general-probe decoded distogram states rendered as a cartoon scaffold' xpos .035 ypos .915 size 17 color #596575",
                f"2dlabels create metrics text '{label_left}     {label_right}' xpos .035 ypos .055 size 18 color #17202a",
                "2dlabels create legend text 'blue helix   cyan beta strand   gray loop / other' xpos .035 ypos .025 size 16 color #596575",
                "wait 5",
                f"movie record size {width},{height} supersample 1",
                f"coordset #1 1,{n_models} pauseFrames 2",
                f"wait {frames}",
                f"movie encode {q(out_mp4)} quality high framerate 16",
                "exit",
                "",
            ]
        )
    )


def render_one(name: str, force: bool, width: int, height: int, source_root: Path, stack_subdir: str, output_tag: str) -> Path:
    root = source_root / name
    if not root.exists() and source_root.name == "cw30_partial_simulations":
        root = source_root / f"{name}_mds"
    stack = root / stack_subdir / "distogram_mds_models.pdb" if stack_subdir else root / "distogram_mds_models.pdb"
    ref = ROOT / REFS[name]
    out_dir = root / "chimerax"
    scaffold = out_dir / f"{name}_{output_tag}_cartoon_scaffold.pdb"
    cxc = out_dir / f"{name}_{output_tag}_cartoon.cxc"
    out_mp4 = out_dir / f"{name}_{output_tag}_chimerax_blue_ribbon.mp4"
    if out_mp4.exists() and not force:
        print(f"{name}: exists")
        return out_mp4

    if not stack.exists():
        raise FileNotFoundError(stack)
    if not ref.exists():
        raise FileNotFoundError(ref)
    out_dir.mkdir(parents=True, exist_ok=True)
    chain = pick_chain(ref, ca_count_in_stack(stack))
    subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/build_chimerax_cartoon_scaffold.py"),
            "--ca-stack",
            str(stack),
            "--reference-pdb",
            str(ref),
            "--chain",
            chain,
            "--out",
            str(scaffold),
            "--align-to-reference",
        ],
        cwd=ROOT,
        check=True,
    )
    left, right = load_metrics(name)
    colors = secondary_color_commands(scaffold)
    write_cxc(cxc, scaffold, colors, out_mp4, f"{name}: {output_tag} decoded trajectory", left, right, model_count(stack), width, height)
    tmp = Path(tempfile.gettempdir()) / f"chimerax_{output_tag}_{name}.cxc"
    tmp.write_text(cxc.read_text())
    subprocess.run([str(CHIMERAX), "--script", str(tmp)], cwd=ROOT, check=True)
    print(out_mp4)
    return out_mp4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=list(REFS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--source-root", type=Path, default=ROOT / "runs/general_probe_simulations")
    parser.add_argument("--stack-subdir", default="mds_general")
    parser.add_argument("--output-tag", default="general_probe")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()
    if not CHIMERAX.exists():
        raise SystemExit(f"ChimeraX not found at {CHIMERAX}")
    for name in args.only:
        if name not in REFS:
            raise SystemExit(f"no reference mapping for {name}")
        render_one(name, args.force, args.width, args.height, args.source_root, args.stack_subdir, args.output_tag)


if __name__ == "__main__":
    main()
