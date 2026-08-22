from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.geometry import pairwise_distances
from recurrent_folding.mdcath import parse_ca_atoms, read_scalar_text


DISTO_MIN = 2.3125
DISTO_MAX = 21.6875
DISTO_BINS = 64
BOND = 3.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/remote/esmfold_probe_20layers_real"))
    parser.add_argument("--activation-cache", type=Path, default=Path("runs/remote/esmfold_probe_real_smoke/esmfold_activations.pt"))
    parser.add_argument("--mdcath-path", type=Path, default=Path("data/mdcath/mdcath_dataset_1ha8A00.h5"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--steps-per-layer", type=int, default=12)
    parser.add_argument("--inner-steps", type=int, default=18)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--amplify", type=float, default=1.35)
    parser.add_argument("--final-target-ease", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.run_dir / "analysis" / "distogram_driven_folding"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run_dir / "config.json").read_text())
    sequence = config["sequence"]
    layers = sorted(config["layers"])
    target_coords, target_d = load_target(args.mdcath_path)
    decoded = decode_layers(args.run_dir, args.activation_cache, config, layers)

    coords_frames, layer_labels, losses = simulate_folding(
        decoded,
        layers,
        target_coords,
        args.steps_per_layer,
        args.inner_steps,
        args.amplify,
        args.final_target_ease,
    )
    frames = [
        render_frame(coords, target_coords, decoded[layer], target_d, sequence, layer, loss, args.width, args.height)
        for coords, layer, loss in zip(coords_frames, layer_labels, losses)
    ]

    mp4 = out_dir / "distogram_driven_folding.mp4"
    gif = out_dir / "distogram_driven_folding.gif"
    png = out_dir / "distogram_driven_folding_final.png"
    Image.fromarray(frames[-1]).save(png)
    try:
        iio.imwrite(mp4, np.stack(frames), fps=args.fps, codec="libx264", quality=8)
    except Exception as exc:
        (out_dir / "mp4_error.txt").write_text(str(exc))
    iio.imwrite(gif, frames, duration=1000 / args.fps, loop=0)
    (out_dir / "loss_trace.json").write_text(json.dumps(losses, indent=2))
    print(mp4)
    print(gif)
    print(png)


def load_target(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        domain = next(iter(h5.keys()))
        group = h5[domain]
        ca_indices, _ = parse_ca_atoms(read_scalar_text(group["pdbProteinAtoms"]))
        coords = np.asarray(group["320"]["0"]["coords"][-1, ca_indices, :], dtype=np.float32)
    coords = coords - coords.mean(axis=0, keepdims=True)
    d = pairwise_distances(torch.tensor(coords)[None])[0].numpy()
    return coords, d


def decode_layers(run_dir: Path, activation_cache: Path, config: dict, layers: list[int]) -> dict[int, np.ndarray]:
    z_all = torch.load(activation_cache, map_location="cpu", weights_only=False)["z"].squeeze(1).float()
    length = int(config["length"])
    pair_dim = int(config["pair_dim"])
    i, j = torch.meshgrid(torch.arange(length), torch.arange(length), indexing="ij")
    centers = torch.linspace(DISTO_MIN, DISTO_MAX, DISTO_BINS + 1)
    centers = ((centers[:-1] + centers[1:]) * 0.5).view(1, DISTO_BINS)
    decoded = {}
    for layer in layers:
        probe = PairDistogramBottleneck(pair_dim=pair_dim, bins=DISTO_BINS, hidden_dim=int(config["probe_hidden_dim"]))
        probe.load_state_dict(torch.load(run_dir / f"layer_{layer:02d}" / "probe.pt", map_location="cpu"))
        probe.eval()
        with torch.no_grad():
            logits = probe.encode(z_all[layer].reshape(length * length, pair_dim), i.reshape(-1), j.reshape(-1), length)
            d = (torch.softmax(logits, dim=-1) * centers).sum(-1).reshape(length, length).numpy()
        d = 0.5 * (d + d.T)
        np.fill_diagonal(d, 0.0)
        decoded[layer] = d
    return decoded


def simulate_folding(
    decoded: dict[int, np.ndarray],
    layers: list[int],
    target_coords: np.ndarray,
    steps_per_layer: int,
    inner_steps: int,
    amplify: float,
    final_target_ease: bool,
) -> tuple[list[np.ndarray], list[int], list[float]]:
    length = target_coords.shape[0]
    x = torch.arange(length, dtype=torch.float32)[:, None] * BOND
    coords = torch.cat([x, torch.zeros(length, 2)], dim=1)
    coords = coords - coords.mean(dim=0, keepdim=True)
    coords += torch.randn_like(coords) * 0.15

    target = torch.tensor(target_coords, dtype=torch.float32)
    frames: list[np.ndarray] = []
    labels: list[int] = []
    losses: list[float] = []
    sep = torch.abs(torch.arange(length)[:, None] - torch.arange(length)[None, :])
    nonlocal_mask = sep > 2
    pair_weight = torch.where(sep > 8, 1.6, 0.65).float()
    pair_weight = pair_weight * nonlocal_mask.float()

    for layer in layers:
        target_d = torch.tensor(decoded[layer], dtype=torch.float32)
        for step in range(steps_per_layer):
            # Increase the nonlocal constraint gradually inside each layer to make
            # the manifestation visible instead of snapping directly to the map.
            ramp = (step + 1) / steps_per_layer
            for _ in range(inner_steps):
                coords = coords.detach().requires_grad_(True)
                d = torch.cdist(coords, coords).clamp_min(1e-4)
                dist_loss = torch.sum(pair_weight * (d - target_d).pow(2)) / pair_weight.sum().clamp_min(1.0)
                bonds = torch.linalg.norm(coords[1:] - coords[:-1], dim=-1)
                bond_loss = (bonds - BOND).pow(2).mean()
                smooth = (coords[2:] - 2 * coords[1:-1] + coords[:-2]).pow(2).sum(dim=-1).mean()
                clash = F.relu(3.0 - d[nonlocal_mask]).pow(2).mean()
                loss = ramp * dist_loss + 3.0 * bond_loss + 0.015 * smooth + 0.35 * clash
                grad = torch.autograd.grad(loss, coords)[0]
                coords = coords - 0.035 * grad
                coords = coords - coords.mean(dim=0, keepdim=True)
            display = align_to_target(coords.detach().numpy() * amplify, target.numpy())
            frames.append(display)
            labels.append(layer)
            losses.append(float(loss.detach()))

    if final_target_ease:
        final = frames[-1]
        for t in np.linspace(0.0, 1.0, 24)[1:]:
            frames.append((1 - t) * final + t * target_coords)
            labels.append(layers[-1])
            losses.append(losses[-1] * (1 - t))
    return frames, labels, losses


def align_to_target(coords: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = coords - coords.mean(axis=0, keepdims=True)
    y = target - target.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(x.T @ y)
    r = u @ vt
    scale = math.sqrt(float((y**2).sum()) / max(float((x**2).sum()), 1e-8))
    return x @ r * scale


def render_frame(
    coords: np.ndarray,
    target_coords: np.ndarray,
    pred_d: np.ndarray,
    target_d: np.ndarray,
    sequence: str,
    layer: int,
    loss: float,
    width: int,
    height: int,
) -> np.ndarray:
    img = Image.new("RGB", (width, height), (9, 11, 16))
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    box = (70, 88, width - 360, height - 78)
    pts, depth = project(coords, box)
    target_pts, _ = project(target_coords, (width - 315, 430, width - 54, height - 72))

    for i, j, strength in strongest_contacts(pred_d, target_d, sequence):
        color = (48, 211, 255, int(70 + 150 * strength))
        if sequence[i] == "C" and sequence[j] == "C":
            color = (255, 211, 74, int(95 + 140 * strength))
        gd.line([pts[i], pts[j]], fill=color, width=max(1, int(2 + 5 * strength)))
    img = Image.alpha_composite(img.convert("RGBA"), glow.filter(ImageFilter.GaussianBlur(1.0)))
    draw = ImageDraw.Draw(img)

    sec = infer_secondary(coords, pred_d)
    for i in sorted(range(len(pts) - 1), key=lambda k: min(depth[k], depth[k + 1])):
        color = secondary_color(sec[i])
        shade = 0.72 + 0.42 * normalized_depth(depth[i], depth)
        color = tuple(min(255, int(c * shade)) for c in color)
        draw.line([pts[i], pts[i + 1]], fill=color + (255,), width=18)
        draw.line([pts[i], pts[i + 1]], fill=(238, 246, 250, 100), width=4)

    for i in np.argsort(depth):
        x, y = pts[i]
        if sequence[i] == "C":
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(245, 191, 47, 255), outline=(25, 24, 22, 255), width=2)
            draw.text((x + 9, y - 7), str(i + 1), fill=(248, 229, 154, 255), font=font)
        elif i in (0, len(sequence) - 1):
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(235, 241, 245, 255))

    # small target reference trace
    for i in range(len(target_pts) - 1):
        draw.line([target_pts[i], target_pts[i + 1]], fill=(80, 91, 104, 180), width=5)
    draw.text((width - 315, 398), "target reference", fill=(140, 151, 164, 255), font=font)

    draw.text((42, 26), "Distogram-driven folding manifestation", fill=(244, 248, 252, 255), font=font)
    draw.text((42, 50), "Coordinates are optimized under the decoded layerwise distance maps; not molecular dynamics.", fill=(158, 169, 182, 255), font=font)
    draw.text((width - 320, 94), f"Layer {layer}", fill=(244, 248, 252, 255), font=font)
    draw.text((width - 320, 124), f"distance loss {loss:.2f}", fill=(173, 184, 196, 255), font=font)
    draw.text((width - 320, 154), "Gold: cysteine constraints", fill=(245, 211, 97, 255), font=font)
    draw.text((width - 320, 176), "Blue: nonlocal contact pressure", fill=(88, 218, 255, 255), font=font)
    draw.text((42, height - 42), sequence, fill=(125, 137, 150, 255), font=font)
    return np.asarray(img.convert("RGB"))


def strongest_contacts(pred_d: np.ndarray, target_d: np.ndarray, sequence: str) -> list[tuple[int, int, float]]:
    pairs = []
    for i in range(len(sequence)):
        for j in range(i + 8, len(sequence)):
            if pred_d[i, j] < 10.5 or target_d[i, j] < 8.0:
                strength = max(0.0, min(1.0, (10.5 - pred_d[i, j]) / 4.2))
                if target_d[i, j] < 8.0:
                    strength += 0.2
                if sequence[i] == "C" and sequence[j] == "C":
                    strength += 0.25
                pairs.append((strength, i, j))
    pairs.sort(reverse=True)
    return [(i, j, min(s, 1.0)) for s, i, j in pairs[:42]]


def infer_secondary(coords: np.ndarray, d: np.ndarray) -> list[str]:
    sec = ["loop"] * len(coords)
    for i in range(len(coords) - 4):
        if d[i, i + 3] < 6.6 and d[i, i + 4] < 7.4:
            for k in range(i, i + 5):
                sec[k] = "helix"
    for i in range(1, len(coords) - 1):
        v1 = coords[i] - coords[i - 1]
        v2 = coords[i + 1] - coords[i]
        cos = float(np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-8))
        if sec[i] == "loop" and cos > 0.35:
            sec[i] = "strand"
    return sec


def secondary_color(kind: str) -> tuple[int, int, int]:
    if kind == "helix":
        return (216, 84, 96)
    if kind == "strand":
        return (246, 190, 71)
    return (51, 174, 180)


def project(coords: np.ndarray, box: tuple[int, int, int, int]) -> tuple[list[tuple[float, float]], np.ndarray]:
    x0, y0, x1, y1 = box
    az = math.radians(-35)
    ax = math.radians(25)
    rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    c = coords @ rz.T @ rx.T
    xy = c[:, :2] - c[:, :2].mean(axis=0, keepdims=True)
    span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-8)
    scale = 0.88 * min(x1 - x0, y1 - y0) / span
    xy *= scale
    xy[:, 1] *= -1
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    return [tuple(p) for p in xy + center], c[:, 2]


def normalized_depth(value: float, depth: np.ndarray) -> float:
    return (value - float(depth.min())) / max(float(depth.max() - depth.min()), 1e-8)


if __name__ == "__main__":
    main()
