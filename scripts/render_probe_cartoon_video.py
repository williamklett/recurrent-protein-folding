from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

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
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument("--interp-frames", type=int, default=8)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.run_dir / "analysis" / "cartoon_video"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run_dir / "config.json").read_text())
    sequence = config["sequence"]
    layers = sorted(config["layers"])
    z_all = torch.load(args.activation_cache, map_location="cpu", weights_only=False)["z"].squeeze(1).float()
    target_coords = load_target_coords(args.mdcath_path).numpy()
    target_d = pairwise_distances(torch.tensor(target_coords)[None])[0].numpy()

    decoded = {layer: decode_expected_distance(args.run_dir, z_all, config, layer) for layer in layers}
    coords = {layer: align_to_target(classical_mds(decoded[layer]), target_coords) for layer in layers}
    cysteine_pairs = closest_cysteine_pairs(sequence, target_d, max_pairs=6)

    frames = []
    for idx, layer in enumerate(layers):
        for _ in range(args.hold_frames):
            frames.append(render_cartoon(coords[layer], decoded[layer], target_d, sequence, cysteine_pairs, str(layer), args.width, args.height))
        if idx == len(layers) - 1:
            continue
        nxt = layers[idx + 1]
        for t in np.linspace(0, 1, args.interp_frames + 2)[1:-1]:
            frames.append(
                render_cartoon(
                    (1 - t) * coords[layer] + t * coords[nxt],
                    (1 - t) * decoded[layer] + t * decoded[nxt],
                    target_d,
                    sequence,
                    cysteine_pairs,
                    f"{layer}->{nxt}",
                    args.width,
                    args.height,
                )
            )

    mp4_path = out_dir / "probe_distogram_cartoon.mp4"
    gif_path = out_dir / "probe_distogram_cartoon.gif"
    png_path = out_dir / "probe_distogram_cartoon_final.png"
    Image.fromarray(frames[-1]).save(png_path)
    try:
        iio.imwrite(mp4_path, np.stack(frames), fps=args.fps, codec="libx264", quality=8)
    except Exception as exc:
        (out_dir / "mp4_error.txt").write_text(str(exc))
    iio.imwrite(gif_path, frames, duration=1000 / args.fps, loop=0)
    print(mp4_path)
    print(gif_path)
    print(png_path)


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
    n = d.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d**2) @ j
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order[:3]], 0)
    return vecs[:, order[:3]] * np.sqrt(vals)[None, :]


def align_to_target(coords: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = coords - coords.mean(axis=0, keepdims=True)
    y = target - target.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(x.T @ y)
    return x @ (u @ vt) * math.sqrt(float((y**2).sum()) / max(float((x**2).sum()), 1e-8))


def closest_cysteine_pairs(sequence: str, target_d: np.ndarray, max_pairs: int) -> list[tuple[int, int]]:
    pairs = []
    cys = [i for i, aa in enumerate(sequence) if aa == "C"]
    for a, i in enumerate(cys):
        for j in cys[a + 1 :]:
            if abs(i - j) > 3:
                pairs.append((target_d[i, j], i, j))
    return [(i, j) for _, i, j in sorted(pairs)[:max_pairs]]


def render_cartoon(
    coords: np.ndarray,
    pred_d: np.ndarray,
    target_d: np.ndarray,
    sequence: str,
    cysteine_pairs: list[tuple[int, int]],
    layer_label: str,
    width: int,
    height: int,
) -> np.ndarray:
    bg = Image.new("RGB", (width, height), (17, 20, 24))
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw_glow = ImageDraw.Draw(glow)
    draw = ImageDraw.Draw(bg)
    font = ImageFont.load_default()
    box = (84, 86, width - 330, height - 74)
    pts, depth = project(coords, box)
    sec = infer_secondary(pred_d, coords)

    # Contact scaffold, drawn behind the backbone.
    for i, j in long_range_contacts(pred_d, limit=36):
        alpha = int(np.clip(230 - 12 * pred_d[i, j], 60, 150))
        draw_glow.line([pts[i], pts[j]], fill=(35, 170, 205, alpha), width=2)

    # Disulfide-like cysteine bridges from target topology, highlighted if the probe brings them close.
    for i, j in cysteine_pairs:
        width_bridge = 5 if pred_d[i, j] < 9.5 else 2
        color = (232, 185, 54, 230) if pred_d[i, j] < 9.5 else (126, 101, 42, 150)
        draw_glow.line([pts[i], pts[j]], fill=color, width=width_bridge)

    bg = Image.alpha_composite(bg.convert("RGBA"), glow.filter(ImageFilter.GaussianBlur(1.2)))
    draw = ImageDraw.Draw(bg)

    # Backbone tube with approximate secondary-structure coloring.
    segment_order = sorted(range(len(pts) - 1), key=lambda idx: min(depth[idx], depth[idx + 1]))
    for i in segment_order:
        color = secondary_color(sec[i])
        shade = 0.75 + 0.35 * normalized_depth(depth[i], depth)
        color = tuple(int(min(255, c * shade)) for c in color)
        draw.line([pts[i], pts[i + 1]], fill=color + (255,), width=17)
    for i in segment_order:
        draw.line([pts[i], pts[i + 1]], fill=(235, 244, 246, 115), width=4)

    residue_order = np.argsort(depth)
    for i in residue_order:
        x, y = pts[i]
        aa = sequence[i]
        if aa == "C":
            r = 8
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(242, 188, 48, 255), outline=(32, 28, 18, 255), width=2)
            draw.text((x + 9, y - 8), str(i + 1), fill=(245, 224, 150, 255), font=font)
        elif i in (0, len(sequence) - 1):
            r = 7
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(235, 235, 238, 255), outline=(30, 30, 32, 255))

    draw_panel(draw, width, height, layer_label, sequence, cysteine_pairs, pred_d, target_d)
    return np.asarray(bg.convert("RGB"))


def infer_secondary(d: np.ndarray, coords: np.ndarray) -> list[str]:
    n = d.shape[0]
    sec = ["loop"] * n
    for i in range(n):
        if i + 4 < n and d[i, i + 3] < 6.4 and d[i, i + 4] < 7.2:
            for k in range(i, min(i + 5, n)):
                sec[k] = "helix"
    for i in range(1, n - 1):
        v1 = coords[i] - coords[i - 1]
        v2 = coords[i + 1] - coords[i]
        cos = float(np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-8))
        if sec[i] == "loop" and cos > 0.45 and i + 2 < n and d[i, i + 2] > 6.0:
            sec[i] = "strand"
    return sec


def secondary_color(kind: str) -> tuple[int, int, int]:
    if kind == "helix":
        return (213, 78, 89)
    if kind == "strand":
        return (245, 190, 71)
    return (55, 169, 173)


def long_range_contacts(d: np.ndarray, limit: int) -> list[tuple[int, int]]:
    pairs = []
    n = d.shape[0]
    for i in range(n):
        for j in range(i + 8, n):
            if d[i, j] < 9.0:
                pairs.append((d[i, j], i, j))
    return [(i, j) for _, i, j in sorted(pairs)[:limit]]


def project(coords: np.ndarray, box: tuple[int, int, int, int]) -> tuple[list[tuple[float, float]], np.ndarray]:
    x0, y0, x1, y1 = box
    az = math.radians(-34)
    ax = math.radians(24)
    rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    c = coords @ rz.T @ rx.T
    xy = c[:, :2] - c[:, :2].mean(axis=0, keepdims=True)
    span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-6)
    scale = 0.88 * min(x1 - x0, y1 - y0) / span
    xy = xy * scale
    xy[:, 1] *= -1
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    return [tuple(p) for p in xy + center], c[:, 2]


def normalized_depth(value: float, depth: np.ndarray) -> float:
    return (value - float(depth.min())) / max(float(depth.max() - depth.min()), 1e-8)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    layer_label: str,
    sequence: str,
    cysteine_pairs: list[tuple[int, int]],
    pred_d: np.ndarray,
    target_d: np.ndarray,
) -> None:
    font = ImageFont.load_default()
    draw.text((42, 28), "Probe Distogram Cartoon", fill=(244, 247, 250, 255), font=font)
    draw.text((42, 50), "MDS fold from decoded expected distances; gold bridges mark cysteine topology", fill=(167, 177, 188, 255), font=font)
    draw.text((width - 292, 88), f"Layer {layer_label}", fill=(244, 247, 250, 255), font=font)
    draw.text((width - 292, 116), "Color key", fill=(210, 218, 226, 255), font=font)
    entries = [((213, 78, 89), "helix-like"), ((245, 190, 71), "strand-like"), ((55, 169, 173), "loop/tube"), ((242, 188, 48), "cysteine")]
    y = 142
    for color, label in entries:
        draw.rectangle((width - 292, y, width - 274, y + 12), fill=color + (255,))
        draw.text((width - 266, y - 1), label, fill=(190, 201, 211, 255), font=font)
        y += 24
    draw.text((width - 292, y + 8), "Closest cysteine pairs", fill=(210, 218, 226, 255), font=font)
    y += 32
    for i, j in cysteine_pairs[:6]:
        draw.text(
            (width - 292, y),
            f"C{i+1:02d}-C{j+1:02d}  pred {pred_d[i,j]:4.1f}A  target {target_d[i,j]:4.1f}A",
            fill=(190, 201, 211, 255),
            font=font,
        )
        y += 18
    draw.text((42, height - 44), f"{sequence}", fill=(142, 153, 164, 255), font=font)


if __name__ == "__main__":
    main()
