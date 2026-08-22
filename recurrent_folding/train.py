from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from tqdm import trange

from recurrent_folding.data import make_toy_batch
from recurrent_folding.losses import final_structure_loss, metrics, physical_loss, reachability_loss
from recurrent_folding.model import RecurrentFolder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--recurrent-steps", type=int, default=8)
    parser.add_argument("--max-delta", type=float, default=1.25)
    parser.add_argument("--reach-max-step", type=float, default=1.35)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/toy_lstm"))
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--phys-weight", type=float, default=0.05)
    parser.add_argument("--reach-weight", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    model = RecurrentFolder(steps=args.recurrent_steps, max_delta=args.max_delta).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    config = vars(args).copy()
    config["run_dir"] = str(args.run_dir)
    config["device"] = str(device)
    config["torch_version"] = torch.__version__
    config["cuda_name"] = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2))

    log_path = args.run_dir / "metrics.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "loss", "final", "physical", "reach", "rmsd", "dist_mae", "bond_mae"],
        )
        writer.writeheader()

        progress = trange(1, args.steps + 1)
        for step in progress:
            batch = make_toy_batch(args.batch_size, args.length, device=device)
            trajectory = model(batch.seq)
            final = trajectory[:, -1]

            final_loss = final_structure_loss(final, batch.coords)
            phys = physical_loss(trajectory)
            reach = reachability_loss(trajectory, max_step=args.reach_max_step)
            loss = final_loss + args.phys_weight * phys + args.reach_weight * reach

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            if step % args.log_every == 0 or step == 1:
                row = {
                    "step": step,
                    "loss": loss.item(),
                    "final": final_loss.item(),
                    "physical": phys.item(),
                    "reach": reach.item(),
                    **metrics(final, batch.coords),
                }
                writer.writerow(row)
                f.flush()
                progress.set_postfix({k: f"{v:.3f}" for k, v in row.items() if k != "step"})

    torch.save(model.state_dict(), args.run_dir / "model.pt")


if __name__ == "__main__":
    main()
