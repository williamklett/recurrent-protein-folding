#!/usr/bin/env python3
"""Specificity-capacity curve for coordinated ubq distogram steering.

This takes the successful all-pair momentum steering field and compresses it
into low-rank coordinated modes across layers and residue pairs.  For each
mode count and steering strength, it applies the same residual distogram
injection to ubq and to replay proteins, then measures target movement versus
replay movement.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForProteinFolding

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers


def parse_layers(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_floats(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def parse_mode_counts(spec: str, max_modes: int) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        out.append(max_modes if part in {"all", "full"} else int(part))
    return sorted(set([m for m in out if 0 <= m <= max_modes]))


def clean_name(path: Path) -> str:
    name = path.name
    return name.removeprefix("benchmark_")


def load_sequence(run_dir: Path) -> str:
    cache_path = run_dir / "esmfold_activations.pt"
    if cache_path.exists():
        return str(torch.load(cache_path, map_location="cpu", weights_only=False)["sequence"])
    seq_p = run_dir / "sequence.txt"
    if seq_p.exists():
        return seq_p.read_text().strip()
    gt_p = run_dir / "gt_ca_distances.npz"
    if gt_p.exists():
        return str(np.load(gt_p)["sequence"])
    raise FileNotFoundError(f"Could not load sequence from {run_dir}")


def load_probe(path: Path, device: torch.device) -> PairDistogramBottleneck:
    probe = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
    probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


def pair_indices(length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    i, j = torch.meshgrid(torch.arange(length, device=device), torch.arange(length, device=device), indexing="ij")
    return i.reshape(-1), j.reshape(-1)


def expected_distance(logits: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (F.softmax(logits, -1) * centers).sum(-1)


def encode_layer(probe: PairDistogramBottleneck, z: torch.Tensor, i: torch.Tensor, j: torch.Tensor, length: int) -> torch.Tensor:
    return probe.encode(z.reshape(length * length, -1), i, j, length).reshape(length, length, -1)


def decode_layer(probe: PairDistogramBottleneck, logits: torch.Tensor, i: torch.Tensor, j: torch.Tensor, length: int) -> torch.Tensor:
    return probe.decode(logits.reshape(length * length, -1), i, j, length).reshape(length, length, -1)


def final_expected_from_output(out, centers: torch.Tensor) -> torch.Tensor:
    logits = out["distogram_logits"] if isinstance(out, dict) else out.distogram_logits
    logits = 0.5 * (logits + logits.transpose(1, 2))
    return expected_distance(logits[0], centers)


def kabsch_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pc = pred - pred.mean(0)
    tc = target - target.mean(0)
    h = pc.T @ tc
    u, _, vt = np.linalg.svd(h)
    det = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, det]) @ u.T
    return pc @ rot.T + target.mean(0)


def ca_tm_and_rmsd(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    n = min(len(a), len(b))
    aa = kabsch_align(a[:n], b[:n])
    d = np.sqrt(((aa - b[:n]) ** 2).sum(-1))
    d0 = max(0.5, 1.24 * np.cbrt(max(n - 15, 1)) - 1.8)
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2))), float(np.sqrt((d * d).mean()))


def transfer_field(field: torch.Tensor, target_length: int) -> torch.Tensor:
    source_length = int(field.shape[0])
    if source_length == target_length:
        return field
    pos = torch.arange(target_length, dtype=torch.float32)
    mapped = torch.round(pos * float(source_length - 1) / float(max(target_length - 1, 1))).long()
    return field[mapped[:, None], mapped[None, :]]


def transfer_field_to_length(field: torch.Tensor, out_length: int) -> torch.Tensor:
    source_length = int(field.shape[0])
    if source_length == out_length:
        return field
    pos = torch.arange(out_length, dtype=torch.float32)
    mapped = torch.round(pos * float(source_length - 1) / float(max(out_length - 1, 1))).long()
    return field[mapped[:, None], mapped[None, :]]


@dataclass
class SteeringState:
    probe: PairDistogramBottleneck
    field: torch.Tensor
    norm_scale: torch.Tensor


class CoordinatedHook:
    def __init__(
        self,
        layer: int,
        state: SteeringState,
        length: int,
        centers: torch.Tensor,
        bin_axis: torch.Tensor,
        strength: float,
        max_logit_tilt: float,
        stats: list[dict],
    ) -> None:
        self.layer = layer
        self.state = state
        self.length = length
        self.centers = centers
        self.bin_axis = bin_axis
        self.strength = strength
        self.max_logit_tilt = max_logit_tilt
        self.stats = stats

    def __call__(self, _module, _args, output):
        s, z = output
        z0 = z.squeeze(0).float()
        device = z0.device
        i, j = pair_indices(self.length, device)
        probe = self.state.probe
        field = transfer_field(self.state.field, self.length).to(device)
        scale = self.state.norm_scale.to(device).clamp_min(1e-6)
        bin_axis = self.bin_axis.to(device)

        logits = encode_layer(probe, z0, i, j, self.length)
        current_ed = expected_distance(logits, self.centers.to(device))
        signed = (field / scale).clamp(-1.0, 1.0)
        tilt = (self.strength * signed[..., None] * bin_axis).clamp(-self.max_logit_tilt, self.max_logit_tilt)
        steered_logits = logits + tilt
        base_recon = decode_layer(probe, logits, i, j, self.length)
        steer_recon = decode_layer(probe, steered_logits, i, j, self.length)
        z_new = z0 + (steer_recon - base_recon)
        diag = torch.arange(self.length, device=device)
        z_new[diag, diag] = z0[diag, diag]

        steered_ed = expected_distance(steered_logits, self.centers.to(device))
        self.stats.append(
            {
                "layer": self.layer,
                "mean_abs_field_A": float(field.abs().mean().detach().cpu()),
                "mean_abs_expected_shift_A": float((steered_ed - current_ed).abs().mean().detach().cpu()),
                "mean_z_delta_norm": float((z_new - z0).norm(dim=-1).mean().detach().cpu()),
            }
        )
        return s, z_new[None].to(dtype=z.dtype)


def build_momentum_matrix(
    target_cache: Path,
    probes_root: Path,
    layers: list[int],
    seq_sep: int,
    device: torch.device,
) -> tuple[str, int, list[int], torch.Tensor, torch.Tensor, dict[int, PairDistogramBottleneck]]:
    cache = torch.load(target_cache, map_location="cpu", weights_only=False)
    z_all = cache["z"].squeeze(1).float()
    sequence = str(cache["sequence"])
    length = int(z_all.shape[1])
    centers = distogram_bin_centers(device, torch.float32)
    i, j = pair_indices(length, device)
    ii, jj = torch.meshgrid(torch.arange(length), torch.arange(length), indexing="ij")
    valid = (jj > ii) & ((jj - ii).abs() > seq_sep)

    needed = sorted(set(layers + [max(0, l - 1) for l in layers]))
    eds: dict[int, torch.Tensor] = {}
    probes: dict[int, PairDistogramBottleneck] = {}
    for layer in needed:
        p = probes_root / f"layer_{layer:02d}" / "probe.pt"
        if not p.exists():
            continue
        probe = load_probe(p, device)
        probes[layer] = probe
        with torch.no_grad():
            logits = encode_layer(probe, z_all[layer].to(device), i, j, length)
            eds[layer] = expected_distance(logits, centers).detach().cpu()

    usable_layers = [layer for layer in layers if layer in eds and (layer - 1) in eds and layer in probes]
    rows = []
    for layer in usable_layers:
        momentum = 0.5 * ((eds[layer] - eds[layer - 1]) + (eds[layer] - eds[layer - 1]).T)
        rows.append(momentum[valid])
    matrix = torch.stack(rows, dim=0)
    return sequence, length, usable_layers, valid, matrix, probes


def build_replay_momentum_matrix(
    run_dir: Path,
    probes_root: Path,
    layers: list[int],
    target_length: int,
    valid: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    cache_path = run_dir / "esmfold_activations.pt"
    if not cache_path.exists():
        print(f"skip replay null cache missing: {run_dir}", flush=True)
        return None
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    z_all = cache["z"].squeeze(1).float()
    length = int(z_all.shape[1])
    centers = distogram_bin_centers(device, torch.float32)
    i, j = pair_indices(length, device)
    needed = sorted(set(layers + [max(0, l - 1) for l in layers]))
    eds: dict[int, torch.Tensor] = {}
    for layer in needed:
        p = probes_root / f"layer_{layer:02d}" / "probe.pt"
        if not p.exists():
            continue
        probe = load_probe(p, device)
        with torch.no_grad():
            logits = encode_layer(probe, z_all[layer].to(device), i, j, length)
            eds[layer] = expected_distance(logits, centers).detach().cpu()
        del probe, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
    rows = []
    for layer in layers:
        if layer not in eds or (layer - 1) not in eds:
            return None
        momentum = 0.5 * ((eds[layer] - eds[layer - 1]) + (eds[layer] - eds[layer - 1]).T)
        transferred = transfer_field_to_length(momentum, target_length)
        rows.append(transferred[valid])
    return torch.stack(rows, dim=0)


def replay_null_matrix(
    target_matrix: torch.Tensor,
    replay_matrices: list[torch.Tensor],
    ridge: float,
) -> torch.Tensor:
    if not replay_matrices:
        return target_matrix
    out = []
    for layer_idx in range(target_matrix.shape[0]):
        t = target_matrix[layer_idx]
        r = torch.stack([m[layer_idx] for m in replay_matrices], dim=0)
        r = r - r.mean(dim=1, keepdim=True)
        t0 = t - t.mean()
        gram = r @ r.T + ridge * torch.eye(r.shape[0])
        coeff = torch.linalg.solve(gram, r @ t0)
        proj = coeff @ r
        out.append(t0 - proj)
    return torch.stack(out, dim=0)


def fields_from_modes(
    matrix: torch.Tensor,
    valid: torch.Tensor,
    length: int,
    layers: list[int],
    mode_count: int,
    svd: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    quantile: float,
    probes: dict[int, PairDistogramBottleneck],
) -> dict[int, SteeringState]:
    u, s, vh = svd
    if mode_count <= 0:
        recon = torch.zeros_like(matrix)
    else:
        recon = (u[:, :mode_count] * s[:mode_count]) @ vh[:mode_count]
    states: dict[int, SteeringState] = {}
    for row_idx, layer in enumerate(layers):
        dense = torch.zeros((length, length), dtype=torch.float32)
        dense[valid] = recon[row_idx].float()
        dense = dense + dense.T
        active = dense[valid].abs()
        scale = torch.quantile(active, quantile).clamp_min(1e-6)
        states[layer] = SteeringState(probe=probes[layer], field=dense.cpu(), norm_scale=scale.cpu())
    return states


def fold_once(
    model,
    tokenizer,
    sequence: str,
    device: torch.device,
    centers: torch.Tensor,
    states: dict[int, SteeringState] | None,
    strength: float,
    max_logit_tilt: float,
    num_recycles: int,
) -> tuple[torch.Tensor, np.ndarray, list[dict]]:
    ids = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    stats: list[dict] = []
    handles = []
    if states and strength != 0.0:
        bin_axis = (centers - centers.mean()) / centers.std()
        for layer, state in states.items():
            handles.append(
                model.trunk.blocks[layer].register_forward_hook(
                    CoordinatedHook(layer, state, len(sequence), centers, bin_axis, strength, max_logit_tilt, stats)
                )
            )
    with torch.no_grad():
        out = model(ids, num_recycles=num_recycles)
    for h in handles:
        h.remove()
    final_ed = final_expected_from_output(out, centers).detach().cpu()
    ca = out["positions"][-1, 0, :, 1, :].detach().cpu().float().numpy()
    return final_ed, ca, stats


def plot_capacity(rows: list[dict], out: Path) -> None:
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    modes = sorted(set(int(r["mode_count"]) for r in rows if float(r["strength"]) > 0))
    cmap = plt.get_cmap("viridis", max(len(modes), 1))
    colors = {m: cmap(i) for i, m in enumerate(modes)}
    for m in modes:
        rs = sorted([r for r in rows if int(r["mode_count"]) == m], key=lambda r: float(r["strength"]))
        x = [float(r["strength"]) for r in rs]
        axes[0].plot(x, [float(r["target_tm_to_baseline"]) for r in rs], marker="o", lw=2, color=colors[m], label=f"k={m}")
        axes[1].plot(x, [float(r["max_replay_final_mae_A"]) for r in rs], marker="o", lw=2, color=colors[m])
        axes[2].plot(
            [float(r["max_replay_final_mae_A"]) for r in rs],
            [float(r["target_final_mae_A"]) for r in rs],
            marker="o",
            lw=2,
            color=colors[m],
        )
    axes[0].axhline(0.5, color="#777", ls="--", lw=1)
    axes[0].set_title("Target deformation")
    axes[0].set_xlabel("strength")
    axes[0].set_ylabel("ubq TM to baseline")
    axes[1].set_title("Replay movement")
    axes[1].set_xlabel("strength")
    axes[1].set_ylabel("max replay final MAE (A)")
    axes[2].set_title("Specificity-capacity curve")
    axes[2].set_xlabel("max replay final MAE (A)")
    axes[2].set_ylabel("ubq final MAE (A)")
    for ax in axes:
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    fig.suptitle("Coordinated low-rank momentum steering capacity", fontsize=15, fontweight="bold")
    fig.savefig(out, dpi=220, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-run-dir", type=Path, default=Path("runs/benchmark_ubq"))
    ap.add_argument("--replay-run-dirs", default="runs/benchmark_T1132,runs/benchmark_T1155,runs/benchmark_mbn,runs/benchmark_T1159")
    ap.add_argument("--probes-root", type=Path, default=Path("runs/general_belief_replay_benchmark20_cw30"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/coordinated_specificity_capacity_ubq"))
    ap.add_argument("--layers", default="2:46")
    ap.add_argument("--mode-counts", default="1,2,4,8,16,32,all")
    ap.add_argument("--basis", choices=["target", "replay_null"], default="target")
    ap.add_argument("--null-ridge", type=float, default=1e-3)
    ap.add_argument("--strengths", default="0,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.75")
    ap.add_argument("--seq-sep", type=int, default=3)
    ap.add_argument("--mode-quantile", type=float, default=0.90)
    ap.add_argument("--max-logit-tilt", type=float, default=2.0)
    ap.add_argument("--num-recycles", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    layers = parse_layers(args.layers)
    strengths = parse_floats(args.strengths)
    centers = distogram_bin_centers(device, torch.float32)

    sequence, length, usable_layers, valid, matrix, probes = build_momentum_matrix(
        args.target_run_dir / "esmfold_activations.pt",
        args.probes_root,
        layers,
        args.seq_sep,
        device,
    )
    replay_dirs = [Path(x.strip()) for x in args.replay_run_dirs.split(",") if x.strip()]
    if args.basis == "replay_null":
        replay_matrices = []
        for p in replay_dirs:
            m = build_replay_momentum_matrix(p, args.probes_root, usable_layers, length, valid, device)
            if m is not None:
                replay_matrices.append(m)
                print(f"loaded replay-null matrix {clean_name(p)}", flush=True)
        matrix = replay_null_matrix(matrix, replay_matrices, args.null_ridge)
        print(f"built replay-null matrix from {len(replay_matrices)} replay proteins", flush=True)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    max_modes = int(min(matrix.shape))
    mode_counts = parse_mode_counts(args.mode_counts, max_modes)
    explained = (s * s) / (s * s).sum()
    (args.out_dir / "svd_summary.json").write_text(
        json.dumps(
            {
                "usable_layers": usable_layers,
                "matrix_shape": list(matrix.shape),
                "singular_values": [float(x) for x in s[: min(20, len(s))]],
                "cumulative_explained": [float(x) for x in torch.cumsum(explained, dim=0)[: min(20, len(s))]],
            },
            indent=2,
        )
    )
    print(f"ubq L={length}; layers={len(usable_layers)}; pair dims={matrix.shape[1]}; max_modes={max_modes}", flush=True)

    print("loading ESMFold", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to(device).eval()
    if device.type == "cuda":
        model.esm = model.esm.half()
    model.trunk.set_chunk_size(args.chunk_size)
    for p in model.parameters():
        p.requires_grad_(False)

    proteins = [(clean_name(args.target_run_dir), sequence)]
    for p in replay_dirs:
        proteins.append((clean_name(p), load_sequence(p)))

    baselines = {}
    t0 = time.time()
    for name, seq in proteins:
        ed, ca, _ = fold_once(model, tokenizer, seq, device, centers, None, 0.0, args.max_logit_tilt, args.num_recycles)
        baselines[name] = {"ed": ed, "ca": ca}
        print(f"baseline {name} L={len(seq)}", flush=True)

    rows = []
    for mode_count in mode_counts:
        states = fields_from_modes(matrix, valid, length, usable_layers, mode_count, (u, s, vh), args.mode_quantile, probes)
        for strength in strengths:
            condition = {"mode_count": mode_count, "strength": strength}
            per_protein = []
            for name, seq in proteins:
                if strength == 0.0:
                    ed, ca, stats = baselines[name]["ed"], baselines[name]["ca"], []
                else:
                    ed, ca, stats = fold_once(
                        model, tokenizer, seq, device, centers, states, strength, args.max_logit_tilt, args.num_recycles
                    )
                base = baselines[name]
                mask = ~torch.eye(len(seq), dtype=torch.bool)
                mae = float((ed[mask] - base["ed"][mask]).abs().mean())
                tm, rmsd = ca_tm_and_rmsd(ca, base["ca"])
                per_protein.append({"name": name, "final_mae_A": mae, "tm": tm, "rmsd_A": rmsd})
            target = per_protein[0]
            replay = per_protein[1:]
            row = {
                **condition,
                "target_final_mae_A": target["final_mae_A"],
                "target_tm_to_baseline": target["tm"],
                "target_rmsd_A": target["rmsd_A"],
                "max_replay_final_mae_A": max((x["final_mae_A"] for x in replay), default=0.0),
                "min_replay_tm_to_baseline": min((x["tm"] for x in replay), default=1.0),
                "elapsed_s": round(time.time() - t0, 1),
                "per_protein": json.dumps(per_protein),
            }
            rows.append(row)
            print(
                f"k={mode_count:02d} strength={strength:g} target_MAE={row['target_final_mae_A']:.3f} "
                f"target_TM={row['target_tm_to_baseline']:.3f} replay_max={row['max_replay_final_mae_A']:.3f} "
                f"replay_minTM={row['min_replay_tm_to_baseline']:.3f}",
                flush=True,
            )

    with (args.out_dir / "capacity_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    plot_capacity(rows, args.out_dir / "specificity_capacity.png")
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
