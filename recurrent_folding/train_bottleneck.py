from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm import trange

from recurrent_folding.bottleneck import DistogramBottleneck
from recurrent_folding.geometry import pairwise_distances
from recurrent_folding.losses import DISTOGRAM_BINS
from recurrent_folding.mdcath import MdcathTrajectory
from recurrent_folding.model import build_model
from recurrent_folding.train import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdcath-path", type=Path, required=True)
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--domain")
    parser.add_argument("--temperature", default="320")
    parser.add_argument("--replica", default="0")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/distogram_bottleneck"))
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--distogram-weight", type=float, default=0.2)
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--next-layer-index", type=int)
    parser.add_argument("--next-bottleneck", type=Path)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    teacher_config = json.loads((args.teacher_run / "config.json").read_text())
    teacher = build_model(
        teacher_config["architecture"],
        teacher_config["window"],
        teacher_config["max_delta"],
        teacher_config.get("pair_dim", 96),
        teacher_config.get("bond_project_iters", 2),
        teacher_config.get("transformer_dim", 128),
        teacher_config.get("transformer_depth", 4),
        teacher_config.get("transformer_heads", 4),
    ).to(device)
    teacher.load_state_dict(torch.load(args.teacher_run / "model.pt", map_location=device))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    hidden_dim = teacher_config.get("transformer_dim", 128) * 2
    bottleneck = DistogramBottleneck(hidden_dim=hidden_dim, bottleneck_dim=args.bottleneck_dim).to(device)
    opt = AdamW(bottleneck.parameters(), lr=args.lr, weight_decay=1e-4)

    next_bottleneck = None
    if args.next_bottleneck is not None:
        next_bottleneck = DistogramBottleneck(hidden_dim=hidden_dim, bottleneck_dim=args.bottleneck_dim).to(device)
        next_bottleneck.load_state_dict(torch.load(args.next_bottleneck, map_location=device))
        next_bottleneck.eval()
        for param in next_bottleneck.parameters():
            param.requires_grad_(False)

    config = vars(args).copy()
    config["mdcath_path"] = str(args.mdcath_path)
    config["teacher_run"] = str(args.teacher_run)
    config["next_bottleneck"] = str(args.next_bottleneck) if args.next_bottleneck is not None else None
    config["run_dir"] = str(args.run_dir)
    config["device"] = str(device)
    config["teacher_config"] = teacher_config
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2))

    with MdcathTrajectory(
        args.mdcath_path,
        domain=args.domain or teacher_config.get("domain"),
        temperature=args.temperature,
        replica=args.replica,
        frame_stride=args.frame_stride,
    ) as traj:
        log_path = args.run_dir / "metrics.csv"
        with log_path.open("w", newline="") as f:
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
            progress = trange(1, args.steps + 1)
            for step in progress:
                batch = traj.sample_batch(args.batch_size, args.window, device, rng)
                with torch.no_grad():
                    _, aux = teacher(batch.seq, initial_coords=batch.start, return_aux=True)
                    layer_hiddens = aux.get("layer_hiddens")
                    if layer_hiddens is None:
                        raise RuntimeError("Teacher does not expose layer_hiddens. Use a transformer teacher.")
                    hidden = select_layer_hidden(layer_hiddens, args.layer_index).detach()

                logits, reconstructed = bottleneck(hidden)
                recon_mse = F.mse_loss(reconstructed, hidden)
                disto_ce, bin_acc = distogram_ce_and_accuracy(logits, batch.target[:, -1])
                consistency_mse = logits.new_zeros(())
                if next_bottleneck is not None:
                    next_index = args.next_layer_index if args.next_layer_index is not None else args.layer_index + 1
                    with torch.no_grad():
                        next_hidden = select_layer_hidden(layer_hiddens, next_index)
                        next_logits = next_bottleneck.encode(next_hidden)
                    consistency_mse = expected_distance_mse(logits, next_logits)
                loss = (
                    recon_mse
                    + args.distogram_weight * disto_ce
                    + args.consistency_weight * consistency_mse
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(bottleneck.parameters(), args.grad_clip)
                opt.step()

                if step % args.log_every == 0 or step == 1:
                    recon_var = torch.mean((hidden - hidden.mean(dim=(0, 1), keepdim=True)) ** 2)
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

    torch.save(bottleneck.state_dict(), args.run_dir / "bottleneck.pt")


def distogram_ce_and_accuracy(logits: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dists = pairwise_distances(coords)
    bins = DISTOGRAM_BINS.to(device=coords.device, dtype=coords.dtype)
    labels = torch.bucketize(dists, bins)
    length = coords.shape[1]
    mask = ~torch.eye(length, device=coords.device, dtype=torch.bool)
    flat_logits = logits[:, mask, :].reshape(-1, logits.shape[-1])
    flat_labels = labels[:, mask].reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_labels)
    acc = (flat_logits.argmax(dim=-1) == flat_labels).float().mean()
    return ce, acc


def select_layer_hidden(layer_hiddens: list[torch.Tensor], index: int) -> torch.Tensor:
    if not layer_hiddens:
        raise RuntimeError("Teacher returned no layer hiddens.")
    if index < 0:
        index = len(layer_hiddens) + index
    if index < 0 or index >= len(layer_hiddens):
        raise IndexError(f"Layer index {index} is out of range for {len(layer_hiddens)} layers.")
    return layer_hiddens[index]


def expected_distance_mse(logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    centers = distogram_bin_centers(logits.device, logits.dtype)
    pred = torch.sum(torch.softmax(logits, dim=-1) * centers, dim=-1)
    target = torch.sum(torch.softmax(target_logits, dim=-1) * centers, dim=-1)
    length = logits.shape[1]
    mask = ~torch.eye(length, device=logits.device, dtype=torch.bool)
    return F.mse_loss(pred[:, mask], target[:, mask])


def distogram_bin_centers(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    edges = DISTOGRAM_BINS.to(device=device, dtype=dtype)
    left = torch.cat([edges.new_tensor([0.0]), edges])
    right = torch.cat([edges, edges.new_tensor([edges[-1] + (edges[-1] - edges[-2])])])
    return ((left + right) * 0.5).view(1, 1, 1, -1)


if __name__ == "__main__":
    main()
