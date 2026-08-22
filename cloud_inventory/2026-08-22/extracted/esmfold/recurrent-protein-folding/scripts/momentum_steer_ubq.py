#!/usr/bin/env python3
"""Momentum-aligned multi-block ESMFold distogram steering.

This is intentionally not a specificity experiment.  It tests whether the
folding trajectory can be pushed off-path by repeatedly nudging each block's
pair representation in the direction that the probe readout was already moving.

For cached trunk block/probe id l:
  1. encode current z_l through the trained layer-l distogram probe
  2. estimate baseline physical momentum: E[d]_l - E[d]_{l-1}
  3. tilt the current distogram logits in that signed distance direction
  4. decode the tilted distogram
  5. inject only the decoded residual: z + (decode(tilted) - decode(current))

The residual injection means non-probe information in z is not bottlenecked.
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


def parse_strengths(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


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
    logits = probe.encode(z.reshape(length * length, -1), i, j, length)
    return logits.reshape(length, length, -1)


def decode_layer(probe: PairDistogramBottleneck, logits: torch.Tensor, i: torch.Tensor, j: torch.Tensor, length: int) -> torch.Tensor:
    z = probe.decode(logits.reshape(length * length, -1), i, j, length)
    return z.reshape(length, length, -1)


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
    if n < 3:
        return float("nan"), float("nan")
    aa = kabsch_align(a[:n], b[:n])
    d = np.sqrt(((aa - b[:n]) ** 2).sum(-1))
    d0 = max(0.5, 1.24 * np.cbrt(max(n - 15, 1)) - 1.8)
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2))), float(np.sqrt((d * d).mean()))


@dataclass
class LayerState:
    probe: PairDistogramBottleneck
    momentum: torch.Tensor
    norm_scale: torch.Tensor
    pair_mask: torch.Tensor


class MomentumSteeringHook:
    def __init__(
        self,
        layer: int,
        state: LayerState,
        length: int,
        centers: torch.Tensor,
        bin_axis: torch.Tensor,
        strength: float,
        max_logit_tilt: float,
        keep_diagonal: bool,
        stats: list[dict],
    ) -> None:
        self.layer = layer
        self.state = state
        self.length = length
        self.centers = centers
        self.bin_axis = bin_axis
        self.strength = strength
        self.max_logit_tilt = max_logit_tilt
        self.keep_diagonal = keep_diagonal
        self.stats = stats

    def __call__(self, _module, _args, output):
        s, z = output
        z0 = z.squeeze(0).float()
        device = z0.device
        i, j = pair_indices(self.length, device)
        probe = self.state.probe
        momentum = self.state.momentum.to(device)
        pair_mask = self.state.pair_mask.to(device)
        scale = self.state.norm_scale.to(device).clamp_min(1e-6)
        bin_axis = self.bin_axis.to(device)

        logits = encode_layer(probe, z0, i, j, self.length)
        current_ed = expected_distance(logits, self.centers.to(device))
        signed = (momentum / scale).clamp(-1.0, 1.0) * pair_mask
        tilt = (self.strength * signed[..., None] * bin_axis).clamp(-self.max_logit_tilt, self.max_logit_tilt)
        steered_logits = logits + tilt
        base_recon = decode_layer(probe, logits, i, j, self.length)
        steer_recon = decode_layer(probe, steered_logits, i, j, self.length)
        z_new = z0 + (steer_recon - base_recon) * pair_mask[..., None]
        if self.keep_diagonal:
            diag = torch.arange(self.length, device=device)
            z_new[diag, diag] = z0[diag, diag]

        steered_ed = expected_distance(steered_logits, self.centers.to(device))
        active = pair_mask.bool()
        mean_shift = (steered_ed[active] - current_ed[active]).abs().mean() if active.any() else torch.tensor(0.0, device=device)
        mean_momentum = momentum[active].abs().mean() if active.any() else torch.tensor(0.0, device=device)
        mean_z_delta = (z_new - z0).norm(dim=-1)[active].mean() if active.any() else torch.tensor(0.0, device=device)
        self.stats.append(
            {
                "layer": self.layer,
                "active_pairs": int(active.sum().detach().cpu()),
                "mean_abs_momentum_A": float(mean_momentum.detach().cpu()),
                "mean_abs_expected_shift_A": float(mean_shift.detach().cpu()),
                "mean_z_delta_norm": float(mean_z_delta.detach().cpu()),
                "max_abs_tilt": float(tilt.abs().max().detach().cpu()),
            }
        )
        return s, z_new[None].to(dtype=z.dtype)


def lj_energy_map(expected_dist: torch.Tensor, sigma: float) -> torch.Tensor:
    d = expected_dist.clamp(min=0.5)
    r = sigma / d
    return 4.0 * (r**12 - r**6)


def select_top_energy_pair_mask(
    eds: dict[int, torch.Tensor],
    length: int,
    top_k: int,
    seq_sep: int,
    sigma: float,
    device: torch.device,
    out_dir: Path | None = None,
) -> torch.Tensor:
    if top_k <= 0:
        return torch.ones((length, length), dtype=torch.float32)
    energy_stack = torch.stack([lj_energy_map(ed, sigma) for _, ed in sorted(eds.items())], dim=0)
    # Positive repulsive energy is the "high energy pair" signal. If a pair is
    # mostly attractive/negative, clamp it to zero so it does not outrank a
    # true steric-conflict pair.
    score = energy_stack.clamp_min(0.0).mean(dim=0)
    ii, jj = torch.meshgrid(torch.arange(length), torch.arange(length), indexing="ij")
    valid = (jj > ii) & ((jj - ii).abs() > seq_sep)
    valid_scores = score.detach().cpu().clone()
    valid_scores[~valid] = -float("inf")
    flat = valid_scores.reshape(-1)
    k = min(top_k, int(valid.sum()))
    idx = torch.topk(flat, k=k).indices
    pairs = []
    mask = torch.zeros((length, length), dtype=torch.float32)
    for flat_idx in idx.tolist():
        i = flat_idx // length
        j = flat_idx % length
        mask[i, j] = 1.0
        mask[j, i] = 1.0
        pairs.append(
            {
                "i": i,
                "j": j,
                "one_indexed": [i + 1, j + 1],
                "mean_positive_lj": float(valid_scores[i, j]),
            }
        )
    if out_dir is not None:
        (out_dir / "top_energy_pairs.json").write_text(json.dumps({"top_k": top_k, "pairs": pairs}, indent=2))
    print("top energy pairs:", ", ".join(f"{p['one_indexed']} E={p['mean_positive_lj']:.3g}" for p in pairs), flush=True)
    return mask


def edge_betweenness_unweighted(length: int, edges: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
    adj = [[] for _ in range(length)]
    edge_set = set()
    for a, b in edges:
        a, b = min(a, b), max(a, b)
        adj[a].append(b)
        adj[b].append(a)
        edge_set.add((a, b))
    scores = {e: 0.0 for e in edge_set}
    for source in range(length):
        stack: list[int] = []
        pred = [[] for _ in range(length)]
        sigma = [0.0] * length
        sigma[source] = 1.0
        dist = [-1] * length
        dist[source] = 0
        queue = [source]
        for v in queue:
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = [0.0] * length
        while stack:
            w = stack.pop()
            for v in pred[w]:
                coeff = (sigma[v] / sigma[w]) * (1.0 + delta[w]) if sigma[w] else 0.0
                e = (min(v, w), max(v, w))
                scores[e] += coeff
                delta[v] += coeff
    # Undirected graph: each shortest path was counted from both endpoints.
    return {e: v * 0.5 for e, v in scores.items()}


def select_structural_low_energy_pair_mask(
    eds: dict[int, torch.Tensor],
    gt_npz: Path,
    length: int,
    top_k: int,
    seq_sep: int,
    sigma: float,
    contact_min: float,
    contact_max: float,
    out_dir: Path | None = None,
) -> torch.Tensor:
    if top_k <= 0:
        return torch.ones((length, length), dtype=torch.float32)
    gt = np.load(gt_npz)
    ca = gt["ca_coords"].astype(float)
    d = np.sqrt(((ca[:, None] - ca[None, :]) ** 2).sum(-1))
    edges: list[tuple[int, int]] = []
    for i in range(length):
        for j in range(i + 1, length):
            if j - i <= seq_sep:
                continue
            if contact_min <= d[i, j] <= contact_max:
                edges.append((i, j))
    if not edges:
        raise RuntimeError(f"No structural contact edges found in {gt_npz}")

    betweenness = edge_betweenness_unweighted(length, edges)
    energy_stack = torch.stack([lj_energy_map(ed, sigma) for _, ed in sorted(eds.items())], dim=0)
    positive_lj = energy_stack.clamp_min(0.0).mean(dim=0).detach().cpu().numpy()
    central = np.array([betweenness[e] for e in edges], dtype=float)
    high_e = np.array([positive_lj[e[0], e[1]] for e in edges], dtype=float)
    central_norm = central / (central.max() + 1e-12)
    energy_norm = high_e / (high_e.max() + 1e-12)
    combo = central_norm - energy_norm
    ranked = sorted(
        zip(edges, central, high_e, combo),
        key=lambda x: (x[3], x[1], -x[2]),
        reverse=True,
    )[:top_k]

    mask = torch.zeros((length, length), dtype=torch.float32)
    pairs = []
    for (i, j), centrality, energy, score in ranked:
        mask[i, j] = 1.0
        mask[j, i] = 1.0
        pairs.append(
            {
                "i": i,
                "j": j,
                "one_indexed": [i + 1, j + 1],
                "ca_distance_A": float(d[i, j]),
                "edge_betweenness": float(centrality),
                "mean_positive_lj": float(energy),
                "selection_score": float(score),
            }
        )
    if out_dir is not None:
        (out_dir / "structural_low_energy_pairs.json").write_text(
            json.dumps(
                {
                    "top_k": top_k,
                    "contact_min_A": contact_min,
                    "contact_max_A": contact_max,
                    "seq_sep": seq_sep,
                    "pairs": pairs,
                },
                indent=2,
            )
        )
    print(
        "structural low-energy pairs:",
        ", ".join(
            f"{p['one_indexed']} betw={p['edge_betweenness']:.2f} posLJ={p['mean_positive_lj']:.3g}"
            for p in pairs
        ),
        flush=True,
    )
    return mask


def build_layer_states(
    activation_cache: Path,
    probes_root: Path,
    layers: list[int],
    device: torch.device,
    momentum_quantile: float,
    top_energy_pairs: int,
    pair_selection: str,
    gt_npz: Path,
    energy_seq_sep: int,
    energy_sigma: float,
    contact_min: float,
    contact_max: float,
    out_dir: Path,
) -> tuple[str, int, dict[int, LayerState]]:
    cache = torch.load(activation_cache, map_location="cpu", weights_only=False)
    z_all = cache["z"].squeeze(1).float()
    sequence = str(cache["sequence"])
    length = int(z_all.shape[1])
    centers = distogram_bin_centers(device, torch.float32)
    i, j = pair_indices(length, device)

    needed = sorted(set(layers + [max(0, l - 1) for l in layers]))
    logits_by_layer: dict[int, torch.Tensor] = {}
    probes_by_layer: dict[int, PairDistogramBottleneck] = {}
    for layer in needed:
        probe_path = probes_root / f"layer_{layer:02d}" / "probe.pt"
        if not probe_path.exists():
            continue
        probe = load_probe(probe_path, device)
        probes_by_layer[layer] = probe
        with torch.no_grad():
            logits_by_layer[layer] = encode_layer(probe, z_all[layer].to(device), i, j, length).detach()

    eds_by_layer = {layer: expected_distance(logits, centers).detach() for layer, logits in logits_by_layer.items()}
    if pair_selection == "top_energy":
        pair_mask = select_top_energy_pair_mask(
            eds_by_layer,
            length,
            top_energy_pairs,
            energy_seq_sep,
            energy_sigma,
            device,
            out_dir,
        )
    elif pair_selection == "structural_low_energy":
        pair_mask = select_structural_low_energy_pair_mask(
            eds_by_layer,
            gt_npz,
            length,
            top_energy_pairs,
            energy_seq_sep,
            energy_sigma,
            contact_min,
            contact_max,
            out_dir,
        )
    else:
        raise ValueError(f"Unknown pair selection: {pair_selection}")

    states: dict[int, LayerState] = {}
    for layer in layers:
        if layer not in probes_by_layer or layer not in logits_by_layer:
            print(f"skip layer {layer}: missing probe/logits", flush=True)
            continue
        prev = layer - 1 if (layer - 1) in logits_by_layer else None
        if prev is None:
            print(f"skip layer {layer}: no previous layer momentum", flush=True)
            continue
        ed = eds_by_layer[layer]
        ed_prev = eds_by_layer[prev]
        momentum = 0.5 * ((ed - ed_prev) + (ed - ed_prev).T)
        active = pair_mask.bool().to(momentum.device)
        nz = momentum[active].abs() if active.any() else momentum.abs().reshape(-1)
        norm_scale = torch.quantile(nz, momentum_quantile).detach()
        states[layer] = LayerState(
            probe=probes_by_layer[layer],
            momentum=momentum.detach().cpu(),
            norm_scale=norm_scale.detach().cpu(),
            pair_mask=pair_mask.detach().cpu(),
        )
    return sequence, length, states


def fold_once(
    model,
    tokenizer,
    sequence: str,
    device: torch.device,
    centers: torch.Tensor,
    layer_states: dict[int, LayerState] | None = None,
    strength: float = 0.0,
    max_logit_tilt: float = 2.0,
    num_recycles: int = 0,
) -> tuple[torch.Tensor, np.ndarray, list[dict]]:
    ids = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    stats: list[dict] = []
    handles = []
    if layer_states and strength != 0.0:
        length = len(sequence)
        bin_axis = (centers - centers.mean()) / centers.std()
        for layer, state in layer_states.items():
            hook = MomentumSteeringHook(
                layer=layer,
                state=state,
                length=length,
                centers=centers,
                bin_axis=bin_axis,
                strength=strength,
                max_logit_tilt=max_logit_tilt,
                keep_diagonal=True,
                stats=stats,
            )
            handles.append(model.trunk.blocks[layer].register_forward_hook(hook))
    with torch.no_grad():
        out = model(ids, num_recycles=num_recycles)
    for h in handles:
        h.remove()
    final_ed = final_expected_from_output(out, centers).detach().cpu()
    ca = out["positions"][-1, 0, :, 1, :].detach().cpu().float().numpy()
    return final_ed, ca, stats


def plot_metrics(rows: list[dict], out_path: Path) -> None:
    strengths = [r["strength"] for r in rows]
    final_mae = [r["final_dist_mae_A"] for r in rows]
    tm = [r["tm_to_baseline"] for r in rows]
    rmsd = [r["ca_rmsd_to_baseline_A"] for r in rows]
    mean_shift = [r["mean_hook_expected_shift_A"] for r in rows]

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle("ubq momentum-aligned residual distogram steering", fontsize=15, fontweight="bold")
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.set_xlabel("steering strength")
    axes[0].plot(strengths, final_mae, marker="o", lw=2.5, color="#2166ac")
    axes[0].set_ylabel("final distogram MAE to baseline (A)")
    axes[0].set_title("Output readout movement")
    axes[1].plot(strengths, tm, marker="o", lw=2.5, color="#762a83")
    axes[1].axhline(0.5, color="#777", ls="--", lw=1)
    axes[1].set_ylabel("TM to baseline")
    axes[1].set_title("Coordinate deformation")
    axes[2].plot(strengths, rmsd, marker="o", lw=2.5, color="#d95f02", label="CA RMSD")
    ax2 = axes[2].twinx()
    ax2.plot(strengths, mean_shift, marker="s", lw=2.0, color="#1b9e77", label="hook shift")
    axes[2].set_ylabel("CA RMSD to baseline (A)")
    ax2.set_ylabel("mean hook expected shift (A)")
    axes[2].set_title("Injected movement")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-run-dir", type=Path, default=Path("runs/benchmark_ubq"))
    ap.add_argument("--probes-root", type=Path, default=Path("runs/general_belief_replay_benchmark20_cw30"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/momentum_steer_ubq"))
    ap.add_argument("--layers", default="2:47", help="Cached trunk block/probe ids to steer, e.g. 2:47 or 10:47")
    ap.add_argument("--strengths", default="0,0.25,0.5,1,1.5,2,3,4")
    ap.add_argument("--max-logit-tilt", type=float, default=2.0)
    ap.add_argument("--momentum-quantile", type=float, default=0.90)
    ap.add_argument("--top-energy-pairs", type=int, default=0, help="If >0, steer only this many highest-energy nonlocal residue pairs.")
    ap.add_argument("--pair-selection", choices=["top_energy", "structural_low_energy"], default="top_energy")
    ap.add_argument("--energy-seq-sep", type=int, default=3, help="Exclude pairs with |i-j| <= this when selecting energy pairs.")
    ap.add_argument("--energy-sigma", type=float, default=6.8, help="Sigma for LJ-style energy pair ranking.")
    ap.add_argument("--contact-min", type=float, default=5.0, help="Minimum CA distance for structural contact-pair selection.")
    ap.add_argument("--contact-max", type=float, default=10.0, help="Maximum CA distance for structural contact-pair selection.")
    ap.add_argument("--num-recycles", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    layers = parse_layers(args.layers)
    strengths = parse_strengths(args.strengths)
    centers = distogram_bin_centers(device, torch.float32)

    sequence, length, layer_states = build_layer_states(
        args.target_run_dir / "esmfold_activations.pt",
        args.probes_root,
        layers,
        device,
        args.momentum_quantile,
        args.top_energy_pairs,
        args.pair_selection,
        args.target_run_dir / "gt_ca_distances.npz",
        args.energy_seq_sep,
        args.energy_sigma,
        args.contact_min,
        args.contact_max,
        args.out_dir,
    )
    print(f"sequence length={length}; steerable layers={min(layer_states)}..{max(layer_states)} ({len(layer_states)})", flush=True)

    print("loading ESMFold", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to(device).eval()
    if device.type == "cuda":
        model.esm = model.esm.half()
    model.trunk.set_chunk_size(args.chunk_size)
    for p in model.parameters():
        p.requires_grad_(False)

    t0 = time.time()
    base_ed, base_ca, _ = fold_once(
        model, tokenizer, sequence, device, centers, None, 0.0, args.max_logit_tilt, args.num_recycles
    )
    rows: list[dict] = []
    np.save(args.out_dir / "ca_baseline.npy", base_ca)

    for strength in strengths:
        if strength == 0:
            final_ed, ca, stats = base_ed, base_ca, []
        else:
            final_ed, ca, stats = fold_once(
                model,
                tokenizer,
                sequence,
                device,
                centers,
                layer_states,
                strength,
                args.max_logit_tilt,
                args.num_recycles,
            )
        tm, rmsd = ca_tm_and_rmsd(ca, base_ca)
        mask = ~torch.eye(length, dtype=torch.bool)
        final_mae = float((final_ed[mask] - base_ed[mask]).abs().mean())
        hook_shift = float(np.mean([s["mean_abs_expected_shift_A"] for s in stats])) if stats else 0.0
        hook_z = float(np.mean([s["mean_z_delta_norm"] for s in stats])) if stats else 0.0
        row = {
            "strength": strength,
            "final_dist_mae_A": final_mae,
            "tm_to_baseline": tm,
            "ca_rmsd_to_baseline_A": rmsd,
            "mean_hook_expected_shift_A": hook_shift,
            "mean_hook_z_delta_norm": hook_z,
            "hooked_layers": len(stats),
            "elapsed_s": round(time.time() - t0, 1),
        }
        rows.append(row)
        np.save(args.out_dir / f"ca_strength_{strength:g}.npy", ca)
        (args.out_dir / f"hook_stats_strength_{strength:g}.json").write_text(json.dumps(stats, indent=2))
        print(
            f"strength={strength:g} final_MAE={final_mae:.3f} TM={tm:.3f} "
            f"RMSD={rmsd:.3f} hook_shift={hook_shift:.3f}",
            flush=True,
        )

    with (args.out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    plot_metrics(rows, args.out_dir / "momentum_steering_summary.png")
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
