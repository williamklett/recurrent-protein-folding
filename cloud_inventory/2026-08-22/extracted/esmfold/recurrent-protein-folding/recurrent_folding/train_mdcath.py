from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import trange

from recurrent_folding.losses import (
    ca_fape_lite_loss,
    distogram_loss,
    metrics,
    physical_loss,
    reachability_from_start_loss,
    trajectory_loss,
)
from recurrent_folding.mdcath import MdcathTrajectory, available_trajectories
from recurrent_folding.model import build_model
from recurrent_folding.train import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdcath-path", type=Path, required=True)
    parser.add_argument("--domain")
    parser.add_argument("--temperature", default="320")
    parser.add_argument("--replica", default="0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/mdcath_1domain"))
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--fape-weight", type=float, default=1.0)
    parser.add_argument("--distogram-weight", type=float, default=0.3)
    parser.add_argument("--phys-weight", type=float, default=0.05)
    parser.add_argument("--reach-weight", type=float, default=0.2)
    parser.add_argument("--traj-weight", type=float, default=0.2)
    parser.add_argument("--constraint-ramp-start", type=int, default=150)
    parser.add_argument("--constraint-ramp-steps", type=int, default=250)
    parser.add_argument("--max-delta", type=float, default=0.8)
    parser.add_argument("--architecture", choices=["residue", "pairwise", "transformer"], default="residue")
    parser.add_argument("--pair-dim", type=int, default=96)
    parser.add_argument("--transformer-dim", type=int, default=128)
    parser.add_argument("--transformer-depth", type=int, default=4)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--bond-project-iters", type=int, default=2)
    parser.add_argument("--reach-max-step", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for row in available_trajectories(args.mdcath_path):
            print("\t".join(map(str, row)))
        return

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    with MdcathTrajectory(
        args.mdcath_path,
        domain=args.domain,
        temperature=args.temperature,
        replica=args.replica,
        frame_stride=args.frame_stride,
    ) as traj:
        model = build_model(
            architecture=args.architecture,
            steps=args.window,
            max_delta=args.max_delta,
            pair_dim=args.pair_dim,
            bond_project_iters=args.bond_project_iters,
            transformer_dim=args.transformer_dim,
            transformer_depth=args.transformer_depth,
            transformer_heads=args.transformer_heads,
        ).to(device)
        opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        config = vars(args).copy()
        config["mdcath_path"] = str(args.mdcath_path)
        config["run_dir"] = str(args.run_dir)
        config["device"] = str(device)
        config["domain"] = traj.domain
        config["length"] = traj.length
        config["num_frames"] = traj.num_frames
        config["torch_version"] = torch.__version__
        config["cuda_name"] = torch.cuda.get_device_name(0) if device.type == "cuda" else None
        (args.run_dir / "config.json").write_text(json.dumps(config, indent=2))

        log_path = args.run_dir / "metrics.csv"
        with log_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "loss",
                    "fape",
                    "distogram",
                    "traj",
                    "physical",
                    "reach",
                    "constraint_scale",
                    "rmsd",
                    "dist_mae",
                    "bond_mae",
                ],
            )
            writer.writeheader()

            progress = trange(1, args.steps + 1)
            for step in progress:
                batch = traj.sample_batch(args.batch_size, args.window, device, rng)
                if args.architecture in {"pairwise", "transformer"}:
                    pred, aux = model(batch.seq, initial_coords=batch.start, return_aux=True)
                else:
                    pred = model(batch.seq, initial_coords=batch.start)
                    aux = {}
                fape = ca_fape_lite_loss(pred, batch.target)
                disto = (
                    distogram_loss(aux["distogram_logits"], batch.target[:, -1])
                    if "distogram_logits" in aux
                    else pred.new_zeros(())
                )
                traj_term = trajectory_loss(pred, batch.target)
                phys = physical_loss(pred)
                reach = reachability_from_start_loss(batch.start, pred, max_step=args.reach_max_step)
                constraint_scale = ramp_value(step, args.constraint_ramp_start, args.constraint_ramp_steps)
                loss = (
                    args.fape_weight * fape
                    + args.distogram_weight * disto
                    + args.traj_weight * traj_term
                    + constraint_scale * (args.phys_weight * phys + args.reach_weight * reach)
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

                if step % args.log_every == 0 or step == 1:
                    row = {
                        "step": step,
                        "loss": loss.item(),
                        "fape": fape.item(),
                        "distogram": disto.item(),
                        "traj": traj_term.item(),
                        "physical": phys.item(),
                        "reach": reach.item(),
                        "constraint_scale": constraint_scale,
                        **metrics(pred[:, -1], batch.target[:, -1]),
                    }
                    writer.writerow(row)
                    f.flush()
                    progress.set_postfix({k: f"{v:.3f}" for k, v in row.items() if k != "step"})

    torch.save(model.state_dict(), args.run_dir / "model.pt")


def ramp_value(step: int, start: int, width: int) -> float:
    if width <= 0:
        return 1.0 if step >= start else 0.0
    return float(np.clip((step - start) / width, 0.0, 1.0))


if __name__ == "__main__":
    main()
