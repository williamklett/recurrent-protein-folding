from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm import trange

from recurrent_folding.esmfold_probes import ESMFoldTrunkCapture, PairDistogramBottleneck
from recurrent_folding.geometry import pairwise_distances
from recurrent_folding.mdcath import parse_ca_atoms, read_scalar_text
from recurrent_folding.train import choose_device


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

ESMFOLD_DISTO_MIN = 2.3125
ESMFOLD_DISTO_MAX = 21.6875
ESMFOLD_DISTO_BINS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdcath-path", type=Path)
    parser.add_argument("--sequence")
    parser.add_argument("--target-distance-npz", type=Path)
    parser.add_argument("--domain")
    parser.add_argument("--temperature", default="320")
    parser.add_argument("--replica", default="0")
    parser.add_argument("--target-frame", type=int, default=-1)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/esmfold_distogram_probes"))
    parser.add_argument("--activation-cache", type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--start-layer", type=int, default=47)
    parser.add_argument("--num-layers", type=int, default=20)
    parser.add_argument("--steps-per-layer", type=int, default=1000)
    parser.add_argument("--pairs-per-step", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--probe-hidden-dim", type=int, default=256)
    parser.add_argument("--distogram-weight", type=float, default=5.0)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--fold-chunk-size", type=int, default=64)
    parser.add_argument("--reuse-activations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.activation_cache or (args.run_dir / "esmfold_activations.pt")

    sequence, target_distances = load_training_target(args)
    if cache_path.exists() and args.reuse_activations:
        activation_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        activation_cache = cache_esmfold_activations(sequence, cache_path, device, args.fold_chunk_size)

    z_all = activation_cache["z"].squeeze(1).contiguous()
    num_blocks, length, _, pair_dim = z_all.shape
    if length != target_distances.shape[0]:
        raise ValueError(f"Activation length {length} does not match target length {target_distances.shape[0]}")

    layers = list(range(args.start_layer, max(args.start_layer - args.num_layers, -1), -1))
    if not layers or layers[0] >= num_blocks or layers[-1] < 0:
        raise ValueError(f"Requested layers {layers} are outside ESMFold block range 0..{num_blocks - 1}")

    target_labels = distogram_labels_from_distances(target_distances).cpu()
    target_centers = distogram_bin_centers("cpu", torch.float32)
    config = vars(args).copy()
    config["mdcath_path"] = str(args.mdcath_path) if args.mdcath_path is not None else None
    config["target_distance_npz"] = str(args.target_distance_npz) if args.target_distance_npz is not None else None
    config["run_dir"] = str(args.run_dir)
    config["activation_cache"] = str(cache_path)
    config["device"] = str(device)
    config["sequence"] = sequence
    config["length"] = length
    config["pair_dim"] = pair_dim
    config["num_blocks"] = num_blocks
    config["layers"] = layers
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2))

    next_probe_path: Path | None = None
    next_layer: int | None = None
    for layer in layers:
        layer_dir = args.run_dir / f"layer_{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        next_probe = None
        if next_probe_path is not None:
            next_probe = PairDistogramBottleneck(
                pair_dim=pair_dim,
                bins=ESMFOLD_DISTO_BINS,
                hidden_dim=args.probe_hidden_dim,
            ).to(device)
            next_probe.load_state_dict(torch.load(next_probe_path, map_location=device))
            next_probe.eval()
            for param in next_probe.parameters():
                param.requires_grad_(False)

        train_one_layer(
            layer=layer,
            z_all=z_all,
            target_labels=target_labels,
            target_centers=target_centers,
            next_layer=next_layer,
            next_probe=next_probe,
            args=args,
            device=device,
            rng=rng,
            layer_dir=layer_dir,
        )
        next_probe_path = layer_dir / "probe.pt"
        next_layer = layer


def train_one_layer(
    layer: int,
    z_all: torch.Tensor,
    target_labels: torch.Tensor,
    target_centers: torch.Tensor,
    next_layer: int | None,
    next_probe: PairDistogramBottleneck | None,
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
    layer_dir: Path,
) -> None:
    length = z_all.shape[1]
    pair_dim = z_all.shape[-1]
    probe = PairDistogramBottleneck(
        pair_dim=pair_dim,
        bins=ESMFOLD_DISTO_BINS,
        hidden_dim=args.probe_hidden_dim,
    ).to(device)
    opt = AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-4)
    z_layer = z_all[layer].float()
    z_next = z_all[next_layer].float() if next_layer is not None else None

    with (layer_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss",
                "recon_mse",
                "recon_var",
                "recon_r2",
                "distogram_ce",
                "bin_acc",
                "consistency_mse",
            ],
        )
        writer.writeheader()
        progress = trange(1, args.steps_per_layer + 1, desc=f"layer {layer}")
        for step in progress:
            i, j = sample_pair_indices(length, args.pairs_per_step, rng)
            pair = z_layer[i, j].to(device)
            i_dev = i.to(device)
            j_dev = j.to(device)
            labels = target_labels[i, j].to(device)

            logits, reconstructed = probe(pair, i_dev, j_dev, length)
            recon_mse = F.mse_loss(reconstructed, pair)
            disto_ce = F.cross_entropy(logits, labels)
            bin_acc = (logits.argmax(dim=-1) == labels).float().mean()
            consistency_mse = logits.new_zeros(())
            if next_probe is not None and z_next is not None:
                with torch.no_grad():
                    next_pair = z_next[i, j].to(device)
                    next_logits = next_probe.encode(next_pair, i_dev, j_dev, length)
                consistency_mse = expected_distance_mse(logits, next_logits, target_centers.to(device))
            loss = (
                recon_mse
                + args.distogram_weight * disto_ce
                + args.consistency_weight * consistency_mse
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(probe.parameters(), args.grad_clip)
            opt.step()

            if step == 1 or step % args.log_every == 0:
                recon_var = torch.mean((pair - pair.mean(dim=0, keepdim=True)) ** 2)
                recon_r2 = 1.0 - recon_mse / recon_var.clamp_min(1e-8)
                row = {
                    "step": step,
                    "loss": loss.item(),
                    "recon_mse": recon_mse.item(),
                    "recon_var": recon_var.item(),
                    "recon_r2": recon_r2.item(),
                    "distogram_ce": disto_ce.item(),
                    "bin_acc": bin_acc.item(),
                    "consistency_mse": consistency_mse.item(),
                }
                writer.writerow(row)
                f.flush()
                progress.set_postfix({k: f"{v:.3f}" for k, v in row.items() if k != "step"})

    torch.save(probe.state_dict(), layer_dir / "probe.pt")


def cache_esmfold_activations(
    sequence: str,
    cache_path: Path,
    device: torch.device,
    chunk_size: int,
) -> dict[str, torch.Tensor | str | int]:
    from transformers import AutoTokenizer, EsmForProteinFolding

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.to(device).eval()
    if device.type == "cuda":
        model.esm = model.esm.half()
    model.trunk.set_chunk_size(chunk_size)

    ids = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    capture = ESMFoldTrunkCapture(model)
    t0 = time.time()
    with capture.capturing(), torch.no_grad():
        _ = model(ids)
    cache = {
        "sequence": sequence,
        "length": len(sequence),
        "z": capture.stacked_z(),
        "seconds": time.time() - t0,
    }
    torch.save(cache, cache_path)
    return cache


def load_mdcath_sequence_and_coords(
    path: Path,
    domain: str | None,
    temperature: str,
    replica: str,
    target_frame: int,
) -> tuple[str, torch.Tensor]:
    with h5py.File(path, "r") as h5:
        if domain is None:
            domain = next(iter(h5.keys()))
        group = h5[domain]
        ca_indices, resnames = parse_ca_atoms(read_scalar_text(group["pdbProteinAtoms"]))
        sequence = "".join(AA3_TO_1.get(name, "A") for name in resnames)
        coords_ds = group[temperature][replica]["coords"]
        frame = target_frame if target_frame >= 0 else coords_ds.shape[0] + target_frame
        coords = np.asarray(coords_ds[frame, ca_indices, :], dtype=np.float32)
    coords_t = torch.tensor(coords, dtype=torch.float32)
    return sequence, coords_t - coords_t.mean(dim=0, keepdim=True)


def load_training_target(args: argparse.Namespace) -> tuple[str, torch.Tensor]:
    if args.target_distance_npz is not None:
        data = np.load(args.target_distance_npz, allow_pickle=True)
        distances = torch.tensor(np.asarray(data["distances"], dtype=np.float32))
        sequence = args.sequence or str(data["sequence"].item() if hasattr(data["sequence"], "item") else data["sequence"])
        if distances.shape[0] != len(sequence):
            raise ValueError(
                f"Target distance size {distances.shape[0]} does not match sequence length {len(sequence)}"
            )
        return sequence, distances

    if args.sequence is not None:
        raise ValueError("--sequence requires --target-distance-npz so distogram labels are defined")

    if args.mdcath_path is None:
        raise ValueError("Either --mdcath-path or --target-distance-npz must be provided")

    sequence, coords = load_mdcath_sequence_and_coords(
        args.mdcath_path,
        args.domain,
        args.temperature,
        args.replica,
        args.target_frame,
    )
    return sequence, pairwise_distances(coords[None])[0]


def sample_pair_indices(length: int, count: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    i_np = rng.integers(0, length, size=count)
    j_np = rng.integers(0, length - 1, size=count)
    j_np = j_np + (j_np >= i_np)
    return torch.from_numpy(i_np).long(), torch.from_numpy(j_np).long()


def distogram_labels(coords: torch.Tensor) -> torch.Tensor:
    dists = pairwise_distances(coords[None])[0]
    return distogram_labels_from_distances(dists)


def distogram_labels_from_distances(dists: torch.Tensor) -> torch.Tensor:
    boundaries = torch.linspace(
        ESMFOLD_DISTO_MIN,
        ESMFOLD_DISTO_MAX,
        ESMFOLD_DISTO_BINS + 1,
        dtype=dists.dtype,
    )[1:-1]
    return torch.bucketize(dists, boundaries)


def distogram_bin_centers(device: str | torch.device, dtype: torch.dtype) -> torch.Tensor:
    edges = torch.linspace(ESMFOLD_DISTO_MIN, ESMFOLD_DISTO_MAX, ESMFOLD_DISTO_BINS + 1, device=device, dtype=dtype)
    return ((edges[:-1] + edges[1:]) * 0.5).view(1, -1)


def expected_distance_mse(
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    pred = torch.sum(torch.softmax(logits, dim=-1) * centers, dim=-1)
    target = torch.sum(torch.softmax(target_logits, dim=-1) * centers, dim=-1)
    return F.mse_loss(pred, target)


if __name__ == "__main__":
    main()
