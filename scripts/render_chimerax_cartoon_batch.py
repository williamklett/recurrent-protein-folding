#!/usr/bin/env python3
"""Render ChimeraX cartoon-scaffold movies for benchmark decoded trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHIMERAX = Path("/Applications/ChimeraX-1.11.1.app/Contents/MacOS/ChimeraX")


MANUAL_REFS = {
    "1AO6": Path("data/1AO6.pdb"),
    "8fmi": Path("data/8FMI.pdb"),
    "ubq": Path("data/1UBQ.pdb"),
    "mbn": Path("data/1MBN.pdb"),
    "7ACN": Path("data/pdb_targets/7acn.pdb"),
    "T1137s4": Path("data/casp15/T1137/T1137s4.pdb"),
    "T1137s6": Path("data/casp15/T1137/T1137s6.pdb"),
}


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


def discover_jobs() -> list[dict[str, Path | str]]:
    jobs: list[dict[str, Path | str]] = []
    for stack in sorted((ROOT / "runs/partial_benchmark").glob("*_analysis/mds_structure_video_fixed/distogram_mds_models.pdb")):
        name = stack.parents[1].name.replace("_analysis", "")
        if name in MANUAL_REFS:
            ref = ROOT / MANUAL_REFS[name]
        elif name.startswith("T"):
            ref = ROOT / f"data/casp15_rcsb/{name}.pdb"
        else:
            raise ValueError(f"no reference mapping for {name}")
        jobs.append({"name": name, "stack": stack, "ref": ref, "analysis": stack.parents[1]})

    jobs.append(
        {
            "name": "7ACN",
            "stack": ROOT / "runs/remote/esmfold_probe_7acn_47layers_consistency1/analysis/mds_structure_video_fixed_chunked/distogram_mds_models.pdb",
            "ref": ROOT / MANUAL_REFS["7ACN"],
            "analysis": ROOT / "runs/remote/esmfold_probe_7acn_47layers_consistency1/analysis",
        }
    )
    return jobs


def load_metrics(analysis_dir: Path) -> tuple[str, str]:
    metrics_path = analysis_dir / "esmfold_direct_eval/metrics.json"
    corr_path = analysis_dir / "layerwise_distance_correlations.csv"
    left = ""
    right = ""
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        left = f"MAE {m['distance_mae_all_pairs']:.2f} A   pLDDT {m['mean_plddt']:.3f}"
    if corr_path.exists():
        rows = {int(r["layer"]): r for r in csv.DictReader(corr_path.open())}
        late = [float(rows[layer]["corr_vs_47"]) for layer in range(36, 48) if layer in rows]
        if late:
            right = f"min late convergence {min(late):.4f}"
    return left, right


def write_cxc(path: Path, scaffold: Path, out_mp4: Path, title: str, label_left: str, label_right: str, width: int, height: int) -> None:
    def q(p: Path) -> str:
        return "'" + str(p).replace("'", "'\\''") + "'"

    path.write_text(
        "\n".join(
            [
                f"open {q(scaffold)} coordsets true",
                "coordset #1 1",
                "cartoon #1",
                "cartoon style helix width 2.0 thickness 0.5",
                "cartoon style strand width 2.1 thickness 0.45",
                "cartoon style coil width 0.55 thickness 0.55",
                "color helix #1b77b4 cartoons",
                "color strand #1b77b4 cartoons",
                "color coil #8c939c cartoons",
                "lighting soft",
                "set bgColor white",
                "camera ortho",
                "view #1",
                "zoom 0.8",
                "turn x -8",
                "turn y 18",
                f"windowsize {width} {height}",
                f"2dlabels create title text '{title}' xpos .035 ypos .955 size 34 color #17202a",
                "2dlabels create subtitle text 'decoded distogram intermediate states as cartoon scaffold' xpos .035 ypos .915 size 18 color #596575",
                f"2dlabels create metrics text '{label_left}     {label_right}' xpos .035 ypos .055 size 18 color #17202a",
                "2dlabels create legend text 'blue helix + beta strand   gray loop / other' xpos .035 ypos .025 size 16 color #596575",
                "wait 5",
                f"movie record size {width},{height} supersample 1",
                "coordset #1 1,47 pauseFrames 2",
                "wait 94",
                f"movie encode {q(out_mp4)} quality high framerate 16",
                "exit",
                "",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()

    if not CHIMERAX.exists():
        raise SystemExit(f"ChimeraX not found at {CHIMERAX}")

    jobs = discover_jobs()
    if args.only:
        wanted = set(args.only)
        jobs = [job for job in jobs if str(job["name"]) in wanted]

    for idx, job in enumerate(jobs, 1):
        name = str(job["name"])
        stack = Path(job["stack"])
        ref = Path(job["ref"])
        analysis = Path(job["analysis"])
        out_dir = analysis / "chimerax_cartoon_scaffold"
        out_dir.mkdir(parents=True, exist_ok=True)
        scaffold = out_dir / f"{name}_cartoon_scaffold_coordsets.pdb"
        script = out_dir / f"{name}_cartoon_movie.cxc"
        out_mp4 = out_dir / f"{name}_cartoon_scaffold_movie.mp4"

        if out_mp4.exists() and not args.force:
            print(f"[{idx:02d}/{len(jobs):02d}] {name}: exists, skipping")
            continue

        stack_ca = ca_count_in_stack(stack)
        chain = pick_chain(ref, stack_ca)
        print(f"[{idx:02d}/{len(jobs):02d}] {name}: stack L={stack_ca}, ref={ref.relative_to(ROOT)}, chain={chain!r}")

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
        left, right = load_metrics(analysis)
        title = f"{name}: decoded folding trajectory"
        write_cxc(script, scaffold, out_mp4, title, left, right, args.width, args.height)
        tmp_script = Path(tempfile.gettempdir()) / f"chimerax_{name}_cartoon_movie.cxc"
        tmp_script.write_text(script.read_text())
        subprocess.run([str(CHIMERAX), "--script", str(tmp_script)], cwd=ROOT, check=True)
        print(f"  wrote {out_mp4.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
