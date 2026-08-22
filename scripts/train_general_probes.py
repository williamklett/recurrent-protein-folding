#!/usr/bin/env python3
"""Train per-layer general probes TOP-DOWN with consistency loss, anchored to
ESMFold's OWN distogram head as the layer-47 reference.

Key change from previous version:
  - We do NOT train a probe at layer 47. ESMFold's native distogram head is
    the gold-standard "probe" there — trained jointly on millions of structures.
  - For layer 46 we use ESMFold's distogram(z_47) as the consistency target.
  - For layers 45..1 we chain consistency to the just-trained probe at layer t+1.

This keeps the layerwise distogram trajectory anchored to a physically real
prediction (the model's own answer), not to an arbitrary fitted bottleneck.

The default training mode is continual across proteins. For each layer, proteins
are visited one at a time, and a bounded replay buffer of pair examples from
previous proteins is mixed into later updates. This lets us train one general
probe per layer while reducing forgetting across the 20-protein panel.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import (
    ESMFOLD_DISTO_BINS,
    distogram_bin_centers,
)


# ─────────────────────────────────────────────────────────────────────────────
# ESMFold distogram head — we extract it from the pre-trained model and apply
# it to cached z_47. This gives the model's actual final distogram prediction.

class EsmfoldDistogramHead:
    """Light wrapper that loads only the distogram head module from ESMFold."""

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        from transformers import EsmForProteinFolding
        print("  loading ESMFold to extract distogram head...")
        m = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
        # The distogram head in HF ESMFold is at trunk.trunk2sm_z output then
        # specifically the distogram_head attribute on the trunk module.
        # We probe several known names and pick the one that exists.
        head_mod = None
        for attr in ("distogram_head", "disto_head"):
            if hasattr(m.trunk, attr):
                head_mod = getattr(m.trunk, attr)
                self.attr_name = f"trunk.{attr}"
                break
        if head_mod is None:
            # Search by name in named_modules
            for name, mod in m.named_modules():
                if "distogram" in name.lower() and isinstance(mod, nn.Module) and any(
                    isinstance(c, nn.Linear) for c in mod.modules()
                ):
                    head_mod = mod
                    self.attr_name = name
                    break
        if head_mod is None:
            raise RuntimeError("Could not locate ESMFold's distogram head; "
                               "inspect model.named_modules() and patch this loader.")
        print(f"  using ESMFold distogram head at '{self.attr_name}': {type(head_mod).__name__}")
        self.head = head_mod.to(device).eval()
        for p in self.head.parameters(): p.requires_grad_(False)
        self.device = device
        self.dtype = dtype
        self.num_bins = None
        # Free the rest of ESMFold from memory
        del m
        import gc; gc.collect()
        torch.cuda.empty_cache()

    def __call__(self, z_layer: torch.Tensor) -> torch.Tensor:
        """z_layer: (L, L, pair_dim) → logits: (L, L, num_bins). Symmetrized."""
        # ESMFold's head is typically a Linear (pair_dim -> num_bins) applied
        # to z, with symmetrization (z + z.T) / 2.
        with torch.no_grad():
            z = z_layer.to(self.device, self.dtype)
            z_sym = 0.5 * (z + z.transpose(0, 1))
            logits = self.head(z_sym)
            if self.num_bins is None:
                self.num_bins = logits.shape[-1]
                print(f"  distogram head outputs {self.num_bins} bins")
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Training data loader

def load_protein_pair(d: Path, layer: int, next_layer: int | None, device: torch.device):
    """Load z[layer], z[next_layer], z[47] for this protein. No GT needed —
    pure model-belief tracking."""
    cache_p = d / "esmfold_activations.pt"
    if not cache_p.exists():
        return None
    cache = torch.load(cache_p, map_location="cpu", weights_only=False)
    z_full = cache["z"].squeeze(1)
    L = z_full.shape[1]
    z_layer = z_full[layer].float().to(device)
    z_next = z_full[next_layer].float().to(device) if next_layer is not None else None
    z_47 = z_full[-1].float().to(device)   # always grab layer 47 (final z)
    del cache, z_full
    return {"z": z_layer, "z_next": z_next, "z_47": z_47,
            "L": L, "name": d.name}


def sample_pair_indices(L: int, count: int, rng: np.random.Generator):
    i = rng.integers(0, L, size=count)
    j = rng.integers(0, L, size=count)
    return torch.from_numpy(i).long(), torch.from_numpy(j).long()


class PairReplayBuffer:
    """Reservoir replay buffer for pair-level training examples."""

    def __init__(self, capacity: int, device: torch.device, seed: int) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.n_seen = 0
        self.z: list[torch.Tensor] = []
        self.target_logits: list[torch.Tensor] = []
        self.i: list[torch.Tensor] = []
        self.j: list[torch.Tensor] = []
        self.length: list[int] = []

    def __len__(self) -> int:
        return len(self.z)

    def add_examples(
        self,
        z: torch.Tensor,
        target_logits: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
        length: int,
    ) -> None:
        z = z.detach().to("cpu", torch.float16)
        target_logits = target_logits.detach().to("cpu", torch.float16)
        i = i.detach().to("cpu", torch.int16)
        j = j.detach().to("cpu", torch.int16)
        for k in range(z.shape[0]):
            self.n_seen += 1
            if len(self.z) < self.capacity:
                slot = len(self.z)
                self.z.append(z[k].clone())
                self.target_logits.append(target_logits[k].clone())
                self.i.append(i[k].clone())
                self.j.append(j[k].clone())
                self.length.append(int(length))
                continue
            slot = int(self.rng.integers(0, self.n_seen))
            if slot < self.capacity:
                self.z[slot] = z[k].clone()
                self.target_logits[slot] = target_logits[k].clone()
                self.i[slot] = i[k].clone()
                self.j[slot] = j[k].clone()
                self.length[slot] = int(length)

    def sample_grouped(self, count: int) -> list[dict]:
        if not self.z or count <= 0:
            return []
        n = min(count, len(self.z))
        idx = self.rng.choice(len(self.z), size=n, replace=False)
        by_length: dict[int, list[int]] = {}
        for raw_idx in idx:
            k = int(raw_idx)
            by_length.setdefault(self.length[k], []).append(k)
        batches = []
        for length, ids in by_length.items():
            batches.append(
                {
                    "z": torch.stack([self.z[k] for k in ids]).to(self.device, torch.float32),
                    "target_logits": torch.stack([self.target_logits[k] for k in ids]).to(self.device, torch.float32),
                    "i": torch.stack([self.i[k] for k in ids]).to(self.device, torch.long),
                    "j": torch.stack([self.j[k] for k in ids]).to(self.device, torch.long),
                    "L": int(length),
                }
            )
        return batches


def batch_loss(
    probe: PairDistogramBottleneck,
    pair: torch.Tensor,
    target_lg: torch.Tensor,
    i_dev: torch.Tensor,
    j_dev: torch.Tensor,
    length: int,
    target_centers: torch.Tensor,
    consistency_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits, reconstructed = probe(pair, i_dev, j_dev, length)
    target_probs = F.softmax(target_lg.float(), dim=-1).detach()
    log_probs = F.log_softmax(logits, dim=-1)
    kl = F.kl_div(log_probs, target_probs, reduction="batchmean")
    recon_mse = F.mse_loss(reconstructed, pair)
    recon_var = ((pair - pair.mean(0, keepdim=True)) ** 2).mean()
    recon_r2 = 1.0 - recon_mse / recon_var.clamp_min(1e-8)
    with torch.no_grad():
        ed_probe = (F.softmax(logits, -1) * target_centers).sum(-1)
        ed_target = (target_probs * target_centers).sum(-1)
        ed_diff = (ed_probe - ed_target).abs().mean()
        top1 = (logits.argmax(-1) == target_lg.argmax(-1)).float().mean()
    loss = recon_mse + consistency_weight * kl
    return loss, {
        "loss": float(loss.detach()),
        "kl_to_target": float(kl.detach()),
        "recon_mse": float(recon_mse.detach()),
        "recon_r2": float(recon_r2.detach()),
        "ed_diff": float(ed_diff),
        "match_top1": float(top1),
    }


def train_one_layer(
    layer: int,
    train_dirs: list[Path],
    out_dir: Path,
    next_logits_provider,           # callable: protein_info -> (L, L, num_bins)  ── target logits
    next_logits_label: str,
    next_layer_for_z: int | None,   # next layer to load for cache, or None
    steps: int, batch_per_step: int, proteins_per_step: int,
    lr: float, distogram_weight: float, consistency_weight: float,
    device: torch.device, seed: int, target_centers: torch.Tensor,
    continual: bool = True,
    replay_buffer_pairs: int = 50000,
    replay_batch_per_step: int = 1024,
    replay_examples_per_protein: int = 4096,
):
    rng = np.random.default_rng(seed + layer)
    torch.manual_seed(seed + layer)
    print(f"\n[layer {layer:02d}] consistency target: {next_logits_label}", flush=True)
    print(f"  loading {len(train_dirs)} proteins...", flush=True)

    proteins = []
    pair_dim = 128
    for d in train_dirs:
        p = load_protein_pair(d, layer, next_layer_for_z, device)
        if p is None:
            print(f"  skip {d.name}", flush=True); continue
        proteins.append(p); pair_dim = p["z"].shape[-1]
    if not proteins:
        raise RuntimeError(f"no proteins loaded at layer {layer}")
    Ls = [p["L"] for p in proteins]
    print(f"  loaded {len(proteins)} proteins, L {min(Ls)}..{max(Ls)} median={int(np.median(Ls))}", flush=True)

    # Pre-compute the target logits per protein using next_logits_provider
    print(f"  precomputing consistency targets...", flush=True)
    for p in proteins:
        with torch.no_grad():
            p["target_logits"] = next_logits_provider(p).detach()    # (L, L, num_bins)

    probe = PairDistogramBottleneck(pair_dim=pair_dim, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device)
    opt = AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "protein", "phase", "replay_size",
                                          "loss", "kl_to_target", "recon_mse",
                                          "recon_r2", "ed_diff", "match_top1"])
        w.writeheader()
        global_step = 0
        replay = PairReplayBuffer(replay_buffer_pairs, device, seed + 10000 + layer)
        if continual:
            order = list(range(len(proteins)))
            for protein_epoch, pk in enumerate(order, 1):
                p = proteins[pk]
                print(f"  protein {protein_epoch:02d}/{len(proteins)} {p['name']} replay={len(replay)}", flush=True)
                for local_step in range(1, steps + 1):
                    global_step += 1
                    opt.zero_grad(set_to_none=True)
                    i, j = sample_pair_indices(p["L"], batch_per_step, rng)
                    i_dev = i.to(device); j_dev = j.to(device)
                    pair = p["z"][i_dev, j_dev]
                    target_lg = p["target_logits"][i_dev, j_dev].float()
                    losses_t, rows = [], []
                    loss, stats = batch_loss(
                        probe, pair, target_lg, i_dev, j_dev, p["L"],
                        target_centers, consistency_weight,
                    )
                    losses_t.append(loss)
                    rows.append(stats)
                    for replay_batch in replay.sample_grouped(replay_batch_per_step):
                        r_loss, r_stats = batch_loss(
                            probe,
                            replay_batch["z"],
                            replay_batch["target_logits"],
                            replay_batch["i"],
                            replay_batch["j"],
                            replay_batch["L"],
                            target_centers,
                            consistency_weight,
                        )
                        losses_t.append(r_loss)
                        rows.append(r_stats)
                    (sum(losses_t) / len(losses_t)).backward()
                    nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                    opt.step()

                    if global_step == 1 or local_step % 50 == 0:
                        row = {
                            "step": global_step,
                            "protein": p["name"],
                            "phase": f"{protein_epoch}:{local_step}",
                            "replay_size": len(replay),
                            "loss": float(np.mean([x["loss"] for x in rows])),
                            "kl_to_target": float(np.mean([x["kl_to_target"] for x in rows])),
                            "recon_mse": float(np.mean([x["recon_mse"] for x in rows])),
                            "recon_r2": float(np.mean([x["recon_r2"] for x in rows])),
                            "ed_diff": float(np.mean([x["ed_diff"] for x in rows])),
                            "match_top1": float(np.mean([x["match_top1"] for x in rows])),
                        }
                        w.writerow(row); f.flush()
                        if local_step % 200 == 0 or local_step == steps:
                            print(f"    step {local_step:4d}/{steps} global={global_step:5d} "
                                  f"KL={row['kl_to_target']:.4f} recon_r2={row['recon_r2']:.3f} "
                                  f"E[d]_diff={row['ed_diff']:.3f}A replay={len(replay)}",
                                  flush=True)
                i, j = sample_pair_indices(p["L"], replay_examples_per_protein, rng)
                with torch.no_grad():
                    replay.add_examples(
                        p["z"][i.to(device), j.to(device)],
                        p["target_logits"][i.to(device), j.to(device)],
                        i,
                        j,
                        p["L"],
                    )
        else:
            for step in range(1, steps + 1):
                chosen = rng.choice(len(proteins),
                                     size=min(proteins_per_step, len(proteins)),
                                     replace=False)
                opt.zero_grad(set_to_none=True)
                losses_t, rows = [], []
                for ck in chosen:
                    p = proteins[ck]
                    i, j = sample_pair_indices(p["L"], batch_per_step, rng)
                    i_dev = i.to(device); j_dev = j.to(device)
                    loss, stats = batch_loss(
                        probe,
                        p["z"][i_dev, j_dev],
                        p["target_logits"][i_dev, j_dev].float(),
                        i_dev,
                        j_dev,
                        p["L"],
                        target_centers,
                        consistency_weight,
                    )
                    losses_t.append(loss)
                    rows.append(stats)
                (sum(losses_t) / len(losses_t)).backward()
                nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                opt.step()

                if step == 1 or step % 50 == 0:
                    row = {
                        "step": step,
                        "protein": "joint",
                        "phase": "joint",
                        "replay_size": 0,
                        "loss": float(np.mean([x["loss"] for x in rows])),
                        "kl_to_target": float(np.mean([x["kl_to_target"] for x in rows])),
                        "recon_mse": float(np.mean([x["recon_mse"] for x in rows])),
                        "recon_r2": float(np.mean([x["recon_r2"] for x in rows])),
                        "ed_diff": float(np.mean([x["ed_diff"] for x in rows])),
                        "match_top1": float(np.mean([x["match_top1"] for x in rows])),
                    }
                    w.writerow(row); f.flush()

    torch.save(probe.state_dict(), out_dir / "probe.pt")
    for p in proteins:
        del p["z"]; del p["z_47"]; del p["target_logits"]
        if p.get("z_next") is not None: del p["z_next"]
    proteins.clear()
    torch.cuda.empty_cache()
    return probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dirs-file", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-per-step", type=int, default=2048)
    ap.add_argument("--proteins-per-step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--distogram-weight", type=float, default=5.0)
    ap.add_argument("--consistency-weight", type=float, default=1.0)
    ap.add_argument("--joint", action="store_true", help="Use old joint random-protein training instead of continual replay.")
    ap.add_argument("--replay-buffer-pairs", type=int, default=50000)
    ap.add_argument("--replay-batch-per-step", type=int, default=1024)
    ap.add_argument("--replay-examples-per-protein", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-layer", type=int, default=46)
    ap.add_argument("--end-layer", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dirs = [Path(l.strip()) for l in args.train_dirs_file.read_text().splitlines() if l.strip()]
    print(f"Training on {len(train_dirs)} proteins, layers {args.start_layer} -> {args.end_layer} "
          f"(top-down, anchored to ESMFold head)",
          flush=True)

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"train_dirs": [str(d) for d in train_dirs], "args": vars(args).copy()}
    manifest["args"]["out_root"] = str(manifest["args"]["out_root"])
    manifest["args"]["train_dirs_file"] = str(manifest["args"]["train_dirs_file"])
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    target_centers = distogram_bin_centers(device, torch.float32)

    # ── Load ESMFold's distogram head as the layer-47 anchor ────────────────
    esm_head = EsmfoldDistogramHead(device)
    def esm_provider(protein_info):
        # Apply head to that protein's z_47
        return esm_head(protein_info["z_47"]).float()

    next_probe = None
    if args.start_layer > 46 or args.start_layer < args.end_layer or args.end_layer < 1:
        raise ValueError("--start-layer/--end-layer must satisfy 46 >= start >= end >= 1")

    # Train layer 46 with ESMFold head as target, then 45 with probe_46, ...
    for layer in range(args.start_layer, args.end_layer - 1, -1):
        layer_dir = args.out_root / f"layer_{layer:02d}"
        if (layer_dir / "probe.pt").exists():
            print(f"\n[layer {layer:02d}] already exists, loading as next_probe", flush=True)
            next_probe = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
            next_probe.load_state_dict(torch.load(layer_dir / "probe.pt", map_location=device, weights_only=True))
            for pp in next_probe.parameters(): pp.requires_grad_(False)
            continue

        if layer == 46:
            provider = esm_provider
            label = "ESMFold native distogram head on z_47"
            next_layer_for_z = None
        else:
            if next_probe is None:
                next_path = args.out_root / f"layer_{layer + 1:02d}" / "probe.pt"
                if not next_path.exists():
                    raise RuntimeError(
                        f"Need trained layer {layer + 1} probe at {next_path} "
                        f"before starting at layer {layer}"
                    )
                next_probe = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
                next_probe.load_state_dict(torch.load(next_path, map_location=device, weights_only=True))
                for pp in next_probe.parameters(): pp.requires_grad_(False)
            local_next = next_probe   # closure on this
            def provider(p, _local_next=local_next):
                with torch.no_grad():
                    z = p["z_next"]                  # (L, L, pair_dim)
                    L = p["L"]
                    flat = z.reshape(-1, z.shape[-1])
                    i_grid, j_grid = torch.meshgrid(
                        torch.arange(L, device=device), torch.arange(L, device=device),
                        indexing="ij"
                    )
                    return _local_next.encode(flat, i_grid.flatten(), j_grid.flatten(), L).reshape(L, L, -1)
            label = f"probe at layer {layer+1}"
            next_layer_for_z = layer + 1

        next_probe = train_one_layer(
            layer=layer, train_dirs=train_dirs, out_dir=layer_dir,
            next_logits_provider=provider, next_logits_label=label,
            next_layer_for_z=next_layer_for_z,
            steps=args.steps, batch_per_step=args.batch_per_step,
            proteins_per_step=args.proteins_per_step,
            lr=args.lr,
            distogram_weight=args.distogram_weight,
            consistency_weight=args.consistency_weight,
            device=device, seed=args.seed,
            target_centers=target_centers,
            continual=not args.joint,
            replay_buffer_pairs=args.replay_buffer_pairs,
            replay_batch_per_step=args.replay_batch_per_step,
            replay_examples_per_protein=args.replay_examples_per_protein,
        )
        next_probe.eval()
        for pp in next_probe.parameters(): pp.requires_grad_(False)

    print(f"\nDONE. Wrote layer 46..1 to {args.out_root}/  (layer 47 = ESMFold native head)", flush=True)


if __name__ == "__main__":
    main()
