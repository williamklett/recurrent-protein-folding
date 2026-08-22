from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.geometry import pairwise_distances
from recurrent_folding.mdcath import parse_ca_atoms, read_scalar_text


DISTO_MIN = 2.3125
DISTO_MAX = 21.6875
DISTO_BINS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/remote/esmfold_probe_20layers_real"))
    parser.add_argument("--activation-cache", type=Path, default=Path("runs/remote/esmfold_probe_real_smoke/esmfold_activations.pt"))
    parser.add_argument("--mdcath-path", type=Path, default=Path("data/mdcath/mdcath_dataset_1ha8A00.h5"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--hold-frames", type=int, default=6)
    parser.add_argument("--interp-frames", type=int, default=7)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.run_dir / "analysis" / "fold_video"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.run_dir / "config.json").read_text())
    sequence = config["sequence"]
    layers = sorted(config["layers"])
    z_all = torch.load(args.activation_cache, map_location="cpu", weights_only=False)["z"].squeeze(1).float()
    target_coords = load_target_coords(args.mdcath_path)
    target_d = pairwise_distances(target_coords[None])[0].numpy()

    decoded = {layer: decode_expected_distance(args.run_dir, z_all, config, layer) for layer in layers}
    coords = {layer: classical_mds(decoded[layer]) for layer in layers}
    target_aligned = target_coords.numpy()
    coords = {layer: align_to_target(x, target_aligned) for layer, x in coords.items()}

    frames = []
    ordered = layers
    for index, layer in enumerate(ordered):
        frames.extend(
            render_frame(
                coords[layer],
                decoded[layer],
                target_d,
                sequence,
                layer,
                ordered[0],
                ordered[-1],
                args.width,
                args.height,
            )
            for _ in range(args.hold_frames)
        )
        if index == len(ordered) - 1:
            continue
        next_layer = ordered[index + 1]
        for t in np.linspace(0.0, 1.0, args.interp_frames + 2)[1:-1]:
            interp_coords = (1.0 - t) * coords[layer] + t * coords[next_layer]
            interp_d = (1.0 - t) * decoded[layer] + t * decoded[next_layer]
            frames.append(
                render_frame(
                    interp_coords,
                    interp_d,
                    target_d,
                    sequence,
                    f"{layer}->{next_layer}",
                    ordered[0],
                    ordered[-1],
                    args.width,
                    args.height,
                )
            )

    mp4_path = out_dir / "probe_distogram_fold.mp4"
    gif_path = out_dir / "probe_distogram_fold.gif"
    try:
        iio.imwrite(mp4_path, np.stack(frames), fps=args.fps, codec="libx264", quality=8)
    except Exception as exc:
        (out_dir / "mp4_error.txt").write_text(str(exc))
    iio.imwrite(gif_path, frames, duration=1000 / args.fps, loop=0)
    print(mp4_path)
    print(gif_path)


def decode_expected_distance(run_dir: Path, z_all: torch.Tensor, config: dict, layer: int) -> np.ndarray:
    length = int(config["length"])
    pair_dim = int(config["pair_dim"])
    probe = PairDistogramBottleneck(pair_dim=pair_dim, bins=DISTO_BINS, hidden_dim=int(config["probe_hidden_dim"]))
    probe.load_state_dict(torch.load(run_dir / f"layer_{layer:02d}" / "probe.pt", map_location="cpu"))
    probe.eval()
    i, j = torch.meshgrid(torch.arange(length), torch.arange(length), indexing="ij")
    centers = torch.linspace(DISTO_MIN, DISTO_MAX, DISTO_BINS + 1)
    centers = ((centers[:-1] + centers[1:]) * 0.5).view(1, DISTO_BINS)
    with torch.no_grad():
        logits = probe.encode(z_all[layer].reshape(length * length, pair_dim), i.reshape(-1), j.reshape(-1), length)
        d = (torch.softmax(logits, dim=-1) * centers).sum(-1).reshape(length, length).numpy()
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d


def load_target_coords(path: Path) -> torch.Tensor:
    with h5py.File(path, "r") as h5:
        domain = next(iter(h5.keys()))
        group = h5[domain]
        ca_indices, _ = parse_ca_atoms(read_scalar_text(group["pdbProteinAtoms"]))
        coords = np.asarray(group["320"]["0"]["coords"][-1, ca_indices, :], dtype=np.float32)
    coords_t = torch.tensor(coords)
    return coords_t - coords_t.mean(dim=0, keepdim=True)


def classical_mds(d: np.ndarray) -> np.ndarray:
    d2 = d**2
    n = d.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order[:3]], 0.0)
    vecs = vecs[:, order[:3]]
    return vecs * np.sqrt(vals)[None, :]


def align_to_target(coords: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = coords - coords.mean(axis=0, keepdims=True)
    y = target - target.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(x.T @ y)
    # Allow reflection; distograms do not encode chirality.
    r = u @ vt
    scale = np.sqrt((y**2).sum() / max((x**2).sum(), 1e-8))
    return x @ r * scale


def render_frame(
    coords: np.ndarray,
    pred_d: np.ndarray,
    target_d: np.ndarray,
    sequence: str,
    layer: int | str,
    first_layer: int,
    last_layer: int,
    width: int,
    height: int,
) -> np.ndarray:
    img = Image.new("RGB", (width, height), (250, 250, 247))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    left = (40, 92, 700, 660)
    right = (760, 92, 1208, 540)
    draw.text((40, 28), "Probe-implied fold from decoded distograms", fill=(24, 32, 42), font=font)
    draw.text((40, 52), f"Layer {layer}  |  sequence length {len(sequence)}  |  dark contacts are predicted < 8A", fill=(70, 77, 86), font=font)

    pts = project(coords, left)
    depth = coords[:, 2]
    order = np.argsort(depth)
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=(88, 99, 115), width=4)

    contacts = []
    for i in range(len(sequence)):
        for j in range(i + 8, len(sequence)):
            if pred_d[i, j] < 8.5:
                contacts.append((pred_d[i, j], i, j))
    for _, i, j in sorted(contacts)[:28]:
        draw.line([pts[i], pts[j]], fill=(40, 120, 155), width=1)

    for i in order:
        x, y = pts[i]
        aa = sequence[i]
        color = (204, 142, 35) if aa == "C" else (55, 92, 160)
        radius = 7 if aa == "C" else 5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(245, 245, 245))
        if aa == "C":
            draw.text((x + 6, y - 7), str(i + 1), fill=(85, 63, 20), font=font)

    draw.text((left[0], left[1] - 24), "3D embedding of probe distances (cysteines labeled)", fill=(24, 32, 42), font=font)
    draw_heatmap(draw, pred_d, target_d, right)
    draw.text((right[0], right[1] - 24), "Predicted contacts vs target misses", fill=(24, 32, 42), font=font)

    progress_x = 40
    progress_y = height - 40
    progress_w = width - 80
    draw.line((progress_x, progress_y, progress_x + progress_w, progress_y), fill=(190, 190, 185), width=6)
    if isinstance(layer, int):
        t = (layer - first_layer) / max(last_layer - first_layer, 1)
    else:
        a, b = [int(x) for x in str(layer).split("->")]
        t = ((a + b) / 2 - first_layer) / max(last_layer - first_layer, 1)
    draw.ellipse((progress_x + t * progress_w - 8, progress_y - 8, progress_x + t * progress_w + 8, progress_y + 8), fill=(26, 102, 168))
    draw.text((40, height - 24), f"Layer progression {first_layer} -> {last_layer}; coordinates are MDS embeddings of expected distances.", fill=(70, 77, 86), font=font)
    return np.asarray(img)


def project(coords: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = box
    angle_z = math.radians(-28)
    angle_x = math.radians(22)
    rz = np.array([[math.cos(angle_z), -math.sin(angle_z), 0], [math.sin(angle_z), math.cos(angle_z), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(angle_x), -math.sin(angle_x)], [0, math.sin(angle_x), math.cos(angle_x)]])
    c = coords @ rz.T @ rx.T
    xy = c[:, :2]
    xy = xy - xy.mean(axis=0, keepdims=True)
    span = np.max(np.ptp(xy, axis=0))
    scale = 0.82 * min(x1 - x0, y1 - y0) / max(span, 1e-6)
    xy = xy * scale
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    xy[:, 1] *= -1
    return [tuple(p) for p in xy + center]


def draw_heatmap(draw: ImageDraw.ImageDraw, pred_d: np.ndarray, target_d: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    n = pred_d.shape[0]
    cell = min((x1 - x0) / n, (y1 - y0) / n)
    for i in range(n):
        for j in range(n):
            if abs(i - j) <= 2:
                color = (230, 230, 228)
            elif pred_d[i, j] < 8 and target_d[i, j] < 8:
                color = (25, 35, 45)
            elif pred_d[i, j] < 8 <= target_d[i, j]:
                color = (190, 55, 45)
            elif pred_d[i, j] >= 8 > target_d[i, j]:
                color = (235, 184, 66)
            else:
                color = (255, 255, 255)
            draw.rectangle((x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell, y0 + (i + 1) * cell), fill=color)
    draw.rectangle((x0, y0, x0 + n * cell, y0 + n * cell), outline=(80, 80, 80), width=1)
    legend_y = y0 + n * cell + 18
    labels = [
        ((25, 35, 45), "hit"),
        ((190, 55, 45), "false +"),
        ((235, 184, 66), "miss"),
    ]
    lx = x0
    for color, text in labels:
        draw.rectangle((lx, legend_y, lx + 14, legend_y + 14), fill=color)
        draw.text((lx + 20, legend_y + 2), text, fill=(55, 60, 68))
        lx += 92


if __name__ == "__main__":
    main()
