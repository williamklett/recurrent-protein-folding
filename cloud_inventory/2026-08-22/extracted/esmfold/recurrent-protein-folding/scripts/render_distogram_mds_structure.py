from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from recurrent_folding.esmfold_probes import PairDistogramBottleneck


DISTO_MIN = 2.3125
DISTO_MAX = 21.6875
DISTO_BINS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/remote/esmfold_probe_20layers_real"))
    parser.add_argument("--activation-cache", type=Path, default=Path("runs/remote/esmfold_probe_real_smoke/esmfold_activations.pt"))
    parser.add_argument("--probe-root", type=Path, help="optional shared/general probe directory containing layer_01..layer_46")
    parser.add_argument("--layers", default=None, help="comma-separated layer list or ranges, e.g. 1-46 or 1,4,8,16,32,46")
    parser.add_argument("--sequence-file", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--interp-frames", type=int, default=5)
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--spin-degrees", type=float, default=220.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.run_dir / "analysis" / "mds_structure_video"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_meta = torch.load(args.activation_cache, map_location="cpu", weights_only=False, mmap=True)
    z_shape = cache_meta["z"].shape
    length = int(cache_meta.get("length", z_shape[-3]))
    sequence = str(cache_meta.get("sequence", ""))
    del cache_meta
    if args.sequence_file and args.sequence_file.exists():
        sequence = args.sequence_file.read_text().strip()
    if not sequence and (args.run_dir / "sequence.txt").exists():
        sequence = (args.run_dir / "sequence.txt").read_text().strip()
    if not sequence and (args.run_dir / "config.json").exists():
        sequence = json.loads((args.run_dir / "config.json").read_text()).get("sequence", "")
    if not sequence:
        sequence = "A" * length
    if len(sequence) != length:
        sequence = sequence[:length].ljust(length, "A")
    if args.layers:
        layers = parse_layers(args.layers)
    elif (args.run_dir / "config.json").exists() and not args.probe_root:
        layers = sorted(json.loads((args.run_dir / "config.json").read_text())["layers"])
    else:
        layers = list(range(1, min(46, z_shape[0] - 1) + 1))
    config = {
        "length": length,
        "pair_dim": int(z_shape[-1]),
        "probe_hidden_dim": 256,
        "probe_root": str(args.probe_root) if args.probe_root else "",
    }
    distances = decode_layers(args.run_dir, args.activation_cache, config, layers)
    coords = {layer: classical_mds(distances[layer]) for layer in layers}
    coords = align_sequence(coords, layers)

    write_ca_multimodel_pdb(out_dir / "distogram_mds_models.pdb", coords, sequence, layers)
    write_metrics(out_dir / "mds_embedding_metrics.csv", distances, coords, layers)

    frames = []
    frame_layers: list[str] = []
    frame_specs = []
    for idx, layer in enumerate(layers):
        frame_specs.append((coords[layer], str(layer)))
        frame_layers.append(str(layer))
        if idx == len(layers) - 1:
            continue
        nxt = layers[idx + 1]
        for t in np.linspace(0.0, 1.0, args.interp_frames + 2)[1:-1]:
            interp = (1.0 - t) * coords[layer] + t * coords[nxt]
            frame_specs.append((interp, f"{layer}->{nxt}"))
            frame_layers.append(f"{layer}->{nxt}")
    for frame_idx, (frame_coords, layer_label) in enumerate(frame_specs):
        spin = math.radians(args.spin_degrees) * frame_idx / max(len(frame_specs) - 1, 1)
        frames.append(render_frame(frame_coords, sequence, layer_label, args.width, args.height, spin))

    mp4 = out_dir / "distogram_mds_structure.mp4"
    gif = out_dir / "distogram_mds_structure.gif"
    png = out_dir / "distogram_mds_structure_final.png"
    Image.fromarray(frames[-1]).save(png)
    try:
        iio.imwrite(mp4, np.stack(frames), fps=args.fps, codec="libx264", quality=8)
    except Exception as exc:
        (out_dir / "mp4_error.txt").write_text(str(exc))
    iio.imwrite(gif, frames, duration=1000 / args.fps, loop=0)
    print(mp4)
    print(gif)
    print(png)
    print(out_dir / "distogram_mds_models.pdb")


def parse_layers(text: str) -> list[int]:
    layers: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            step = 1 if b >= a else -1
            layers.extend(range(a, b + step, step))
        else:
            layers.append(int(part))
    return sorted(dict.fromkeys(layers))


def decode_layers(run_dir: Path, activation_cache: Path, config: dict, layers: list[int]) -> dict[int, np.ndarray]:
    z_all = torch.load(activation_cache, map_location="cpu", weights_only=False)["z"].squeeze(1).float()
    length = int(config["length"])
    pair_dim = int(config["pair_dim"])
    i, j = torch.meshgrid(torch.arange(length), torch.arange(length), indexing="ij")
    centers = torch.linspace(DISTO_MIN, DISTO_MAX, DISTO_BINS + 1)
    centers = ((centers[:-1] + centers[1:]) * 0.5).view(1, DISTO_BINS)
    decoded = {}
    for layer in layers:
        probe_root = config.get("probe_root")
        probe_dir = Path(probe_root) if probe_root else run_dir
        probe = PairDistogramBottleneck(pair_dim=pair_dim, bins=DISTO_BINS, hidden_dim=int(config["probe_hidden_dim"]))
        probe.load_state_dict(torch.load(probe_dir / f"layer_{layer:02d}" / "probe.pt", map_location="cpu", weights_only=True))
        probe.eval()
        with torch.no_grad():
            logits = probe.encode(z_all[layer].reshape(length * length, pair_dim), i.reshape(-1), j.reshape(-1), length)
            d = (torch.softmax(logits, dim=-1) * centers).sum(-1).reshape(length, length).numpy()
        d = 0.5 * (d + d.T)
        np.fill_diagonal(d, 0.0)
        decoded[layer] = d
    return decoded


def classical_mds(d: np.ndarray) -> np.ndarray:
    n = d.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (d**2) @ centering
    vals, vecs = np.linalg.eigh(gram)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    coords = vecs[:, :3] * np.sqrt(np.maximum(vals[:3], 0.0))[None, :]
    return coords - coords.mean(axis=0, keepdims=True)


def align_sequence(coords: dict[int, np.ndarray], layers: list[int]) -> dict[int, np.ndarray]:
    """Chain-align: each layer is Kabsch-aligned to the previous layer.

    This minimises apparent motion between adjacent frames so the animation
    shows genuine structural change rather than coordinate-system jumps.
    Layer 1 is placed in a canonical PCA orientation with stable axis signs.
    """
    aligned = {layers[0]: canonical_orientation(coords[layers[0]])}
    for prev, layer in zip(layers, layers[1:]):
        aligned[layer] = kabsch_align(coords[layer], aligned[prev])
    return aligned


def canonical_orientation(x: np.ndarray) -> np.ndarray:
    """PCA orientation with sign-stabilised axes.

    SVD eigenvectors have arbitrary sign; we fix each axis to point in the
    direction of positive mean projection so the starting frame is
    deterministic and consistent across runs and proteins.
    """
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    signs = np.sign(np.mean(x @ vt.T, axis=0))
    signs[signs == 0] = 1.0
    vt = (vt.T * signs).T
    if np.linalg.det(vt) < 0:
        vt[-1] *= -1
    return x @ vt.T


def kabsch_align(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Reflection-safe Kabsch rotation of x onto reference.

    Plain SVD procrustes (r = u @ vt) can give det(r) = -1, a mirror flip.
    The Kabsch correction forces det(R) = +1 (proper rotation only).
    """
    x0 = x - x.mean(axis=0, keepdims=True)
    y0 = reference - reference.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(x0.T @ y0)
    d = np.linalg.det(u @ vt)
    r = u @ np.diag([1.0, 1.0, np.sign(d)]) @ vt
    return x0 @ r


def procrustes(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return kabsch_align(x, reference)


def render_frame(coords: np.ndarray, sequence: str, layer_label: str, width: int, height: int, spin: float) -> np.ndarray:
    img = Image.new("RGB", (width, height), (246, 248, 250))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    pts, depth = project(coords, (70, 92, width - 70, height - 94), spin)
    for i in sorted(range(len(pts) - 1), key=lambda k: min(depth[k], depth[k + 1])):
        shade = 0.58 + 0.42 * norm_depth(depth[i], depth)
        color = tuple(int(c * shade) for c in (44, 132, 185))
        draw.line([pts[i], pts[i + 1]], fill=color, width=9)
    for i in np.argsort(depth):
        x, y = pts[i]
        if sequence[i] == "C":
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(220, 164, 38), outline=(90, 68, 20))
            draw.text((x + 7, y - 7), str(i + 1), fill=(74, 56, 16), font=font)
        elif i in (0, len(sequence) - 1):
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(38, 70, 100))

    draw.text((34, 26), "3D structure entailed by decoded distogram readout", fill=(20, 28, 36), font=font)
    cam_note = "Classical MDS embedding of expected distances; fixed camera viewpoint." if spin == 0.0 else "Classical MDS embedding of expected distances; camera rotates to reveal depth."
    draw.text((34, 48), cam_note, fill=(82, 91, 102), font=font)
    draw.text((width - 180, 34), f"Layer {layer_label}", fill=(20, 28, 36), font=font)
    draw.text((34, height - 42), sequence, fill=(95, 104, 116), font=font)
    return np.asarray(img)


def project(coords: np.ndarray, box: tuple[int, int, int, int], spin: float) -> tuple[list[tuple[float, float]], np.ndarray]:
    x0, y0, x1, y1 = box
    az = math.radians(-35) + spin
    ax = math.radians(22)
    rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    c = coords @ rz.T @ rx.T
    xy = c[:, :2] - c[:, :2].mean(axis=0, keepdims=True)
    span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-8)
    scale = 0.86 * min(x1 - x0, y1 - y0) / span
    xy *= scale
    xy[:, 1] *= -1
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    return [tuple(p) for p in xy + center], c[:, 2]


def norm_depth(value: float, depth: np.ndarray) -> float:
    return (value - float(depth.min())) / max(float(depth.max() - depth.min()), 1e-8)


def write_ca_multimodel_pdb(path: Path, coords: dict[int, np.ndarray], sequence: str, layers: list[int]) -> None:
    aa1_to_3 = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
        "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
        "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }
    lines = []
    for model_idx, layer in enumerate(layers, start=1):
        lines.append(f"MODEL     {model_idx:4d}  LAYER {layer}")
        for i, (aa, xyz) in enumerate(zip(sequence, coords[layer]), start=1):
            resn = aa1_to_3.get(aa, "ALA")
            x, y, z = xyz
            lines.append(f"ATOM  {i:5d}  CA  {resn} A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
        lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def write_metrics(path: Path, distances: dict[int, np.ndarray], coords: dict[int, np.ndarray], layers: list[int]) -> None:
    lines = ["layer,stress_rmse,mean_bond_length"]
    for layer in layers:
        embedded_d = np.linalg.norm(coords[layer][:, None, :] - coords[layer][None, :, :], axis=-1)
        mask = np.triu(np.ones_like(embedded_d, dtype=bool), 1)
        stress = float(np.sqrt(np.mean((embedded_d[mask] - distances[layer][mask]) ** 2)))
        bonds = np.linalg.norm(coords[layer][1:] - coords[layer][:-1], axis=-1)
        lines.append(f"{layer},{stress:.6f},{float(bonds.mean()):.6f}")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
