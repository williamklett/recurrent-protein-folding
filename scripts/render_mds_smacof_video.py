#!/usr/bin/env python3
"""Render a layerwise SMACOF MDS trajectory from probe distograms.

This script implements backward SMACOF optimization: it starts with a stable
Classical MDS embedding of the final layer and then iterates backwards
(from layer 46 down to 1). By initializing each layer N from the coordinates
of layer N+1, we maintain perfect temporal coherence and eliminate
coordinate-system flips, mirror-image jumps, and axis swaps.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--smacof-steps", type=int, default=15, help="steps per layer")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    return parser.parse_args()


def classical_mds(d: np.ndarray) -> np.ndarray:
    """Standard eigenvalue-based MDS for the initial (final) frame."""
    n = d.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (d**2) @ centering
    vals, vecs = np.linalg.eigh(gram)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    coords = vecs[:, :3] * np.sqrt(np.maximum(vals[:3], 0.0))[None, :]
    
    # Sign-stabilize axes via PCA
    coords = coords - coords.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(coords, full_matrices=False)
    signs = np.sign(np.mean(coords @ vt.T, axis=0))
    signs[signs == 0] = 1.0
    vt = (vt.T * signs).T
    if np.linalg.det(vt) < 0:
        vt[-1] *= -1
    return coords @ vt.T


def smacof_step(x: np.ndarray, d_target: np.ndarray) -> np.ndarray:
    """A single Guttman transform step for SMACOF."""
    n = x.shape[0]
    d_current = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    d_current[d_current == 0] = 1e-9
    
    # B matrix construction
    b = -d_target / d_current
    np.fill_diagonal(b, 0)
    np.fill_diagonal(b, -b.sum(axis=1))
    
    return (1.0 / n) * (b @ x)


def decode_distogram(run_dir: Path, z_all: torch.Tensor, layer: int, config: dict) -> np.ndarray:
    length = z_all.shape[1]
    pair_dim = z_all.shape[-1]
    
    # Use standard bin centers
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


def render_frame(coords: np.ndarray, sequence: str, layer: int, width: int, height: int) -> np.ndarray:
    img = Image.new("RGB", (width, height), (246, 248, 250))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    
    # Simple fixed projection
    az = np.radians(-35)
    ax = np.radians(22)
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    c = coords @ rz.T @ rx.T
    
    # Center and scale
    xy = c[:, :2] - c[:, :2].mean(axis=0)
    span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-8)
    scale = 0.75 * min(width, height) / span
    xy = xy * scale + np.array([width / 2, height / 2])
    
    # Depth-sorted lines
    depth = c[:, 2]
    for i in sorted(range(len(coords) - 1), key=lambda k: min(depth[k], depth[k + 1])):
        d_norm = (depth[i] - depth.min()) / (depth.max() - depth.min() + 1e-8)
        shade = int(100 + 100 * d_norm)
        draw.line([tuple(xy[i]), tuple(xy[i + 1])], fill=(44, 132, shade), width=7)
        
    draw.text((40, 40), f"Layer {layer:02d} (SMACOF Iterative MDS)", fill=(20, 28, 36))
    draw.text((40, 70), "Initializing each layer from the previous ensures perfect temporal coherence.", fill=(82, 91, 102))
    
    return np.asarray(img)


def main():
    args = parse_args()
    cache_path = args.activation_cache or args.run_dir / "esmfold_activations.pt"
    out_dir = args.out_dir or args.run_dir / "analysis" / "smacof_video"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    data = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
    z_all = data["z"].squeeze(1).float() # (48, L, L, 128)
    sequence = data.get("sequence", "A" * z_all.shape[1])
    config = json.loads((args.run_dir / "config.json").read_text())
    
    # Determine which layers we have probes for
    layers = sorted([int(p.parent.name.split("_")[-1]) for p in args.run_dir.glob("layer_*/probe.pt")])
    if not layers:
        raise ValueError(f"No probe layers found in {args.run_dir}")
        
    print(f"Rendering {len(layers)} layers for {args.run_dir.name}...")
    
    # 2. Backward Optimization (The Secret Sauce)
    # We start at the end (Layer 46/47) because it's the most stable/defined.
    coords_per_layer = {}
    
    # Initial layer via Classical MDS
    final_layer = layers[-1]
    print(f"  Layer {final_layer}: Classical MDS initialization")
    d_final = decode_distogram(args.run_dir, z_all, final_layer, config)
    coords_per_layer[final_layer] = classical_mds(d_final)
    
    # Work backwards
    prev_coords = coords_per_layer[final_layer]
    for layer in reversed(layers[:-1]):
        print(f"  Layer {layer}: SMACOF (from layer {layer+1})")
        d_target = decode_distogram(args.run_dir, z_all, layer, config)
        
        # Start with coords of layer above
        x = prev_coords.copy()
        for _ in range(args.smacof_steps):
            x = smacof_step(x, d_target)
            
        coords_per_layer[layer] = x
        prev_coords = x
        
    # 3. Render
    frames = []
    for layer in layers:
        frames.append(render_frame(coords_per_layer[layer], sequence, layer, args.width, args.height))
        
    mp4 = out_dir / "smacof_folding.mp4"
    iio.imwrite(mp4, np.stack(frames), fps=args.fps, codec="libx264", quality=8)
    print(f"Done! Saved to {mp4}")


if __name__ == "__main__":
    main()
