#!/usr/bin/env python3
"""Render a layerwise MDS trajectory with consistent global chirality.

This script uses Classical MDS to extract coordinates but solves the "rotation jump"
problem by aligning every layer to the FINAL layer using a reflection-allowed
Procrustes transformation. This ensures that if MDS returns a mirror image for
an intermediate layer, it is flipped back to match the handedness of the 
final prediction, preventing violent 180-degree alignment rotations.
"""

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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--activation-cache", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--interp-frames", type=int, default=5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    return parser.parse_args()


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


def canonical_orientation(x: np.ndarray) -> np.ndarray:
    """PCA orientation with sign-stabilised axes and forced right-handedness."""
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    signs = np.sign(np.mean(x @ vt.T, axis=0))
    signs[signs == 0] = 1.0
    vt = (vt.T * signs).T
    if np.linalg.det(vt) < 0:
        vt[-1] *= -1
    return x @ vt.T


def align_to_reference(coords_dict: dict[int, np.ndarray], layers: list[int]) -> dict[int, np.ndarray]:
    """Align every layer to the FINAL layer, allowing reflections to fix chirality."""
    ref_layer = layers[-1]
    reference = canonical_orientation(coords_dict[ref_layer])
    
    aligned = {}
    for layer in layers:
        mobile = coords_dict[layer]
        m0 = mobile - mobile.mean(axis=0, keepdims=True)
        t0 = reference - reference.mean(axis=0, keepdims=True)
        
        # Procrustes alignment (u @ vt) naturally allows reflection if it 
        # improves the fit. This "fixes" the arbitrary chirality of MDS.
        u, _, vt = np.linalg.svd(m0.T @ t0)
        r = u @ vt
        aligned[layer] = m0 @ r + reference.mean(axis=0, keepdims=True)
        
    return aligned


def decode_distogram(run_dir: Path, z_all: torch.Tensor, layer: int, config: dict) -> np.ndarray:
    length = z_all.shape[1]
    pair_dim = z_all.shape[-1]
    centers = torch.linspace(DISTO_MIN, DISTO_MAX, DISTO_BINS + 1)
    centers = ((centers[:-1] + centers[1:]) * 0.5).view(1, DISTO_BINS)
    
    probe = PairDistogramBottleneck(pair_dim=pair_dim, bins=DISTO_BINS, hidden_dim=config["probe_hidden_dim"])
    probe_path = run_dir / f"layer_{layer:02d}" / "probe.pt"
    probe.load_state_dict(torch.load(probe_path, map_location="cpu", weights_only=True))
    probe.eval()
    
    i, j = torch.meshgrid(torch.arange(length), torch.arange(length), indexing="ij")
    with torch.no_grad():
        logits = probe.encode(z_all[layer].reshape(-1, pair_dim), i.reshape(-1), j.reshape(-1), length)
        d = (torch.softmax(logits, dim=-1) * centers).sum(-1).reshape(length, length).numpy()
    
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d


def render_frame(coords: np.ndarray, layer: int, width: int, height: int) -> np.ndarray:
    img = Image.new("RGB", (width, height), (246, 248, 250))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    
    # Fixed projection
    az = np.radians(-35)
    ax = np.radians(22)
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    c = coords @ rz.T @ rx.T
    
    xy = c[:, :2] - c[:, :2].mean(axis=0)
    span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-8)
    scale = 0.75 * min(width, height) / span
    xy = xy * scale + np.array([width / 2, height / 2])
    
    depth = c[:, 2]
    for i in sorted(range(len(coords) - 1), key=lambda k: min(depth[k], depth[k + 1])):
        d_norm = (depth[i] - depth.min()) / (depth.max() - depth.min() + 1e-8)
        shade = int(100 + 100 * d_norm)
        draw.line([tuple(xy[i]), tuple(xy[i + 1])], fill=(44, 132, shade), width=7)
        
    draw.text((40, 40), f"Layer {layer:02d} (Consistent Chirality MDS)", fill=(20, 28, 36))
    draw.text((40, 70), "Aligning all layers to a global reference fixes handedness flips.", fill=(82, 91, 102))
    
    return np.asarray(img)


def main():
    args = parse_args()
    cache_path = args.activation_cache or args.run_dir / "esmfold_activations.pt"
    out_dir = args.out_dir or args.run_dir / "analysis" / "consistent_mds_video"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
    z_all = data["z"].squeeze(1).float()
    config = json.loads((args.run_dir / "config.json").read_text())
    
    layers = sorted([int(p.parent.name.split("_")[-1]) for p in args.run_dir.glob("layer_*/probe.pt")])
    
    print(f"Processing {len(layers)} layers for {args.run_dir.name}...")
    
    # 1. Compute Classical MDS for every layer
    raw_coords = {}
    for layer in layers:
        print(f"  Layer {layer}: decoding + MDS")
        d = decode_distogram(args.run_dir, z_all, layer, config)
        raw_coords[layer] = classical_mds(d)
        
    # 2. Align all to FINAL layer (reflection-allowed)
    aligned_coords = align_to_reference(raw_coords, layers)
    
    # 3. Interpolate and Render
    frames = []
    layer_list = sorted(aligned_coords.keys())
    for i, layer in enumerate(layer_list):
        # Current layer frame
        frames.append(render_frame(aligned_coords[layer], layer, args.width, args.height))
        
        # Interpolate to next layer
        if i < len(layer_list) - 1:
            next_layer = layer_list[i+1]
            c1 = aligned_coords[layer]
            c2 = aligned_coords[next_layer]
            for f in range(1, args.interp_frames + 1):
                t = f / (args.interp_frames + 1)
                interp_coords = (1 - t) * c1 + t * c2
                # We label it with a fractional layer for clarity
                frames.append(render_frame(interp_coords, layer, args.width, args.height))
        
    mp4 = out_dir / "consistent_mds_folding.mp4"
    iio.imwrite(mp4, np.stack(frames), fps=args.fps, codec="libx264", quality=8)
    print(f"Done! Saved to {mp4}")


if __name__ == "__main__":
    main()
