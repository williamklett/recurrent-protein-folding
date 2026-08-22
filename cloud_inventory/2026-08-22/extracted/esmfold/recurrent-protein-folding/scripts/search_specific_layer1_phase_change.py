#!/usr/bin/env python3
"""Search for a ubq-specific layer-1 distogram phase-change perturbation.

The optimized object is a layer-1 distogram-logit perturbation for a target
protein. Injection preserves information outside the probe bottleneck:

    z_inject = z_orig + (decode(D_orig + delta) - decode(D_orig))

Specificity is enforced during search. The same normalized delta field is
resampled onto nearby replay proteins; those proteins are penalized if their
final native-output distogram changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import EsmForProteinFolding

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers


def load_probe(path: Path, device: torch.device) -> PairDistogramBottleneck:
    probe = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
    probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


def load_cache(run_dir: Path, layer: int) -> dict:
    cache = torch.load(run_dir / "esmfold_activations.pt", map_location="cpu", weights_only=False)
    z_all = cache["z"].squeeze(1)
    z_layer = z_all[layer].float().contiguous()
    sequence = cache["sequence"]
    length = int(z_layer.shape[0])
    del z_all, cache
    return {"name": clean_name(run_dir), "run_dir": run_dir, "sequence": sequence, "z_layer": z_layer, "length": length}


def cheap_length(run_dir: Path) -> int | None:
    seq_p = run_dir / "sequence.txt"
    if seq_p.exists():
        return len(seq_p.read_text().strip())
    src_p = run_dir / "source.json"
    if src_p.exists():
        try:
            src = json.loads(src_p.read_text())
            if "resolved_length" in src:
                return int(src["resolved_length"])
            if "length" in src:
                return int(src["length"])
        except Exception:
            return None
    return None


def clean_name(path: Path) -> str:
    name = path.name
    for prefix in ("benchmark_", "hard_synthetic_miss_", "synthetic_design_"):
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def all_pair_indices(length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ii, jj = torch.meshgrid(torch.arange(length, device=device), torch.arange(length, device=device), indexing="ij")
    return ii.reshape(-1), jj.reshape(-1)


def expected_distance_from_logits(logits: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (F.softmax(logits, -1) * centers).sum(-1)


def find_distogram_head(model: EsmForProteinFolding) -> torch.nn.Module:
    for attr in ("distogram_head", "disto_head"):
        if hasattr(model.trunk, attr):
            return getattr(model.trunk, attr)
    for name, mod in model.named_modules():
        if "distogram" in name.lower() and any(isinstance(c, torch.nn.Linear) for c in mod.modules()):
            print(f"using ESMFold native distogram head at {name}", flush=True)
            return mod
    raise RuntimeError("Could not locate ESMFold native distogram head")


def final_expected_from_output(out, centers: torch.Tensor) -> torch.Tensor:
    logits = out["distogram_logits"] if isinstance(out, dict) else out.distogram_logits
    logits = 0.5 * (logits + logits.transpose(1, 2))
    return expected_distance_from_logits(logits[0], centers)


def kabsch_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pc = pred - pred.mean(0)
    tc = target - target.mean(0)
    h = pc.T @ tc
    u, _, vt = np.linalg.svd(h)
    det = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, det]) @ u.T
    return pc @ rot.T + target.mean(0)


def ca_tm_and_rmsd(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    n = min(len(a), len(b))
    if n < 3:
        return float("nan"), float("nan")
    aa = kabsch_align(a[:n], b[:n])
    d = np.sqrt(((aa - b[:n]) ** 2).sum(-1))
    d0 = max(0.5, 1.24 * np.cbrt(max(n - 15, 1)) - 1.8)
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2))), float(np.sqrt((d * d).mean()))


class ZInjectHook:
    def __init__(self, z_inject: torch.Tensor) -> None:
        self.z_inject = z_inject

    def __call__(self, _module, _args, output):
        s, z = output
        return s, self.z_inject[None].to(device=z.device, dtype=z.dtype)


def encode_full(probe: PairDistogramBottleneck, z_layer: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = int(z_layer.shape[0])
    ii, jj = all_pair_indices(length, device)
    pairs = z_layer.reshape(length * length, -1).to(device)
    logits = probe.encode(pairs, ii, jj, length).reshape(length, length, -1)
    logits = 0.5 * (logits + logits.transpose(0, 1))
    return logits, ii, jj


def decode_full(probe: PairDistogramBottleneck, logits: torch.Tensor, device: torch.device) -> torch.Tensor:
    length = int(logits.shape[0])
    ii, jj = all_pair_indices(length, device)
    z = probe.decode(logits.reshape(length * length, -1), ii, jj, length)
    return z.reshape(length, length, -1)


def transfer_delta(delta_source: torch.Tensor, target_length: int) -> torch.Tensor:
    source_length = int(delta_source.shape[0])
    if target_length == source_length:
        return delta_source
    device = delta_source.device
    pos = torch.arange(target_length, device=device, dtype=torch.float32)
    mapped = torch.round(pos * float(source_length - 1) / float(max(target_length - 1, 1))).long()
    return delta_source[mapped[:, None], mapped[None, :]]


def z_with_residual_injection(
    z_layer: torch.Tensor,
    probe: PairDistogramBottleneck,
    base_logits: torch.Tensor,
    delta_logits: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    z_dev = z_layer.to(device)
    base_recon = decode_full(probe, base_logits, device)
    adv_recon = decode_full(probe, base_logits + delta_logits, device)
    z_adv = z_dev + (adv_recon - base_recon)
    diag = torch.arange(z_adv.shape[0], device=device)
    z_adv[diag, diag] = z_dev[diag, diag]
    return z_adv


def fold_with_optional_injection(
    model: EsmForProteinFolding,
    ids: torch.Tensor,
    inject_layer: int,
    z_inject: torch.Tensor | None,
) -> tuple[torch.Tensor, np.ndarray]:
    handle = None
    if z_inject is not None:
        handle = model.trunk.blocks[inject_layer].register_forward_hook(ZInjectHook(z_inject))
    out = model(ids, num_recycles=0)
    if handle is not None:
        handle.remove()
    ca = out["positions"][-1, 0, :, 1, :].detach().cpu().float().numpy()
    final_ed = final_expected_from_output(out, distogram_bin_centers(ids.device, torch.float32))
    return final_ed, ca


def sample_final_pairs(length: int, count: int, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed + length)
    i = rng.integers(0, length, size=count)
    j = rng.integers(0, length - 1, size=count)
    j = j + (j >= i)
    return torch.from_numpy(i).long().to(device), torch.from_numpy(j).long().to(device)


def dist_change(a: torch.Tensor, b: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(a[i, j] - b[i, j]))


def corr_np(a: np.ndarray, b: np.ndarray) -> float:
    if np.nanstd(a) <= 1e-12 or np.nanstd(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def select_neighbors(
    target: dict,
    candidates: list[dict],
    probe: PairDistogramBottleneck,
    device: torch.device,
    layer: int,
    n: int,
) -> list[dict]:
    centers = distogram_bin_centers(device, torch.float32)
    lt = target["length"]
    # Fixed normalized pair grid makes different lengths comparable.
    grid = np.linspace(0.05, 0.95, 32)
    pairs = [(u, v) for u in grid for v in grid if abs(u - v) > 0.05]

    def signature(protein: dict) -> np.ndarray:
        length = protein["length"]
        ii = torch.tensor([min(length - 1, max(0, round(u * (length - 1)))) for u, _ in pairs], device=device)
        jj = torch.tensor([min(length - 1, max(0, round(v * (length - 1)))) for _, v in pairs], device=device)
        with torch.no_grad():
            z = protein["z_layer"]
            logits = probe.encode(z[ii.cpu(), jj.cpu()].to(device), ii, jj, length)
            ed = expected_distance_from_logits(logits, centers)
        return ed.detach().cpu().numpy()

    target_sig = signature(target)
    scored = []
    for cand in candidates:
        if cand["name"] == target["name"]:
            continue
        if cand["length"] > 256:
            continue
        sig = signature(cand)
        scored.append((corr_np(target_sig, sig), cand))
    scored.sort(key=lambda x: (-np.nan_to_num(x[0], nan=-2.0), abs(x[1]["length"] - lt)))
    out = [c for _, c in scored[:n]]
    print("selected neighbors:", ", ".join(f"{c['name']}(L={c['length']})" for c in out), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-run-dir", type=Path, default=Path("runs/benchmark_ubq"))
    ap.add_argument("--candidate-dirs-file", type=Path, default=Path("runs/benchmark_general_train_dirs.txt"))
    ap.add_argument("--probes-root", type=Path, default=Path("runs/general_belief_replay_benchmark20_cw30"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/specific_phase_change_ubq_layer1"))
    ap.add_argument("--inject-layer", type=int, default=1)
    ap.add_argument("--neighbors", type=int, default=4)
    ap.add_argument("--max-neighbor-length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.12)
    ap.add_argument("--max-delta-logit", type=float, default=3.0)
    ap.add_argument("--final-pairs", type=int, default=4096)
    ap.add_argument("--break-margin", type=float, default=1.25, help="Target final expected-distance MAE vs baseline in Angstroms.")
    ap.add_argument("--replay-margin", type=float, default=0.18, help="Allowed replay final expected-distance MAE vs baseline.")
    ap.add_argument("--layer1-weight", type=float, default=1.0)
    ap.add_argument("--break-weight", type=float, default=12.0)
    ap.add_argument("--replay-weight", type=float, default=40.0)
    ap.add_argument("--l2-weight", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    centers = distogram_bin_centers(device, torch.float32)

    print("loading target/candidates", flush=True)
    target = load_cache(args.target_run_dir, args.inject_layer)
    cand_dirs = [Path(x.strip()) for x in args.candidate_dirs_file.read_text().splitlines() if x.strip()]
    candidates = []
    for p in cand_dirs:
        if not (p / "esmfold_activations.pt").exists():
            continue
        length = cheap_length(p)
        if length is not None and length > args.max_neighbor_length:
            print(f"skip candidate {clean_name(p)} L={length} > {args.max_neighbor_length}", flush=True)
            continue
        candidates.append(load_cache(p, args.inject_layer))

    probe = load_probe(args.probes_root / f"layer_{args.inject_layer:02d}" / "probe.pt", device)
    neighbors = select_neighbors(target, candidates, probe, device, args.inject_layer, args.neighbors)

    print("loading ESMFold", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to(device).eval()
    if device.type == "cuda":
        model.esm = model.esm.half()
    model.trunk.set_chunk_size(64)
    for p in model.parameters():
        p.requires_grad_(False)
    find_distogram_head(model)

    proteins = [target] + neighbors
    for p in proteins:
        p["ids"] = tokenizer([p["sequence"]], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        p["base_logits_l1"] = encode_full(probe, p["z_layer"], device)[0].detach()
        p["base_ed_l1"] = expected_distance_from_logits(p["base_logits_l1"], centers).detach()
        fi, fj = sample_final_pairs(p["length"], args.final_pairs, device, 17)
        p["fi"], p["fj"] = fi, fj
        with torch.no_grad():
            zero_delta = torch.zeros_like(p["base_logits_l1"], device=device)
            z_zero = z_with_residual_injection(p["z_layer"], probe, p["base_logits_l1"], zero_delta, device)
            base_final, base_ca = fold_with_optional_injection(model, p["ids"], args.inject_layer, z_zero)
        p["base_final_ed"] = base_final.detach()
        p["base_ca"] = base_ca

    raw_delta = (1e-3 * torch.randn_like(target["base_logits_l1"], device=device)).requires_grad_(True)
    opt = torch.optim.AdamW([raw_delta], lr=args.lr, weight_decay=0.0)
    rows = []
    best_feasible = None
    best_break_candidate = None
    best_nonbreaking_candidate = None
    t0 = time.time()

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        delta = args.max_delta_logit * torch.tanh(raw_delta)
        delta = 0.5 * (delta + delta.transpose(0, 1))
        diag = torch.arange(delta.shape[0], device=device)
        delta = delta.clone()
        delta[diag, diag] = 0.0

        z_target = z_with_residual_injection(
            target["z_layer"], probe, target["base_logits_l1"], delta, device
        )
        target_final, _ = fold_with_optional_injection(model, target["ids"], args.inject_layer, z_target)
        target_change = dist_change(target_final, target["base_final_ed"], target["fi"], target["fj"])
        target_l1_adv = expected_distance_from_logits(target["base_logits_l1"] + delta, centers)
        layer1_change = dist_change(target_l1_adv, target["base_ed_l1"], target["fi"], target["fj"])

        replay_terms = []
        replay_max = torch.tensor(0.0, device=device)
        active_neighbor = neighbors[(step - 1) % len(neighbors)] if neighbors else None
        replay_eval = [active_neighbor] if active_neighbor is not None else []
        for nb in replay_eval:
            d_nb = transfer_delta(delta, nb["length"])
            nb_l1_adv = expected_distance_from_logits(nb["base_logits_l1"] + d_nb, centers)
            nb_change = dist_change(nb_l1_adv, nb["base_ed_l1"], nb["fi"], nb["fj"])
            replay_terms.append(F.relu(nb_change - args.replay_margin) ** 2)
            replay_max = torch.maximum(replay_max, nb_change.detach())

        replay_loss = torch.stack(replay_terms).mean() if replay_terms else torch.tensor(0.0, device=device)
        break_loss = F.relu(args.break_margin - target_change) ** 2
        l2 = torch.mean(delta * delta)
        loss = (
            args.layer1_weight * layer1_change
            + args.break_weight * break_loss
            + args.replay_weight * replay_loss
            + args.l2_weight * l2
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_delta], 5.0)
        opt.step()

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "target_final_mae": float(target_change.detach().cpu()),
            "target_layer1_mae": float(layer1_change.detach().cpu()),
            "replay_active_max_mae": float(replay_max.detach().cpu()),
            "delta_l2": float(l2.detach().cpu()),
            "elapsed_s": round(time.time() - t0, 1),
        }

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                zt = z_with_residual_injection(target["z_layer"], probe, target["base_logits_l1"], delta, device)
                _, ca_adv = fold_with_optional_injection(model, target["ids"], args.inject_layer, zt)
                tm_self, rmsd_self = ca_tm_and_rmsd(ca_adv, target["base_ca"])
                replay_all = []
                for nb in neighbors:
                    d_nb = transfer_delta(delta, nb["length"])
                    zn = z_with_residual_injection(nb["z_layer"], probe, nb["base_logits_l1"], d_nb, device)
                    nb_final, ca_nb = fold_with_optional_injection(model, nb["ids"], args.inject_layer, zn)
                    nb_change = dist_change(nb_final, nb["base_final_ed"], nb["fi"], nb["fj"])
                    tm_nb, rmsd_nb = ca_tm_and_rmsd(ca_nb, nb["base_ca"])
                    replay_all.append(
                        {"name": nb["name"], "final_mae": float(nb_change.cpu()), "tm_to_base": tm_nb, "rmsd_to_base": rmsd_nb}
                    )
                row["target_tm_to_base"] = tm_self
                row["target_rmsd_to_base"] = rmsd_self
                row["replay_max_mae"] = max([x["final_mae"] for x in replay_all], default=0.0)
                row["replay_min_tm"] = min([x["tm_to_base"] for x in replay_all], default=1.0)
                print(
                    f"step {step:04d} loss={row['loss']:.3f} "
                    f"target_final={row['target_final_mae']:.3f} target_l1={row['target_layer1_mae']:.3f} "
                    f"target_TMbase={tm_self:.3f} replay_max={row['replay_max_mae']:.3f} "
                    f"replay_minTM={row['replay_min_tm']:.3f}",
                    flush=True,
                )
                candidate = {
                    "step": step,
                    "row": row,
                    "replay": replay_all,
                    "delta": delta.detach().cpu(),
                }
                feasible = target_change.detach().item() >= args.break_margin and row["replay_max_mae"] <= args.replay_margin
                row["feasible"] = feasible
                if row["target_final_mae"] >= args.break_margin:
                    if best_break_candidate is None or row["target_layer1_mae"] < best_break_candidate["row"]["target_layer1_mae"]:
                        best_break_candidate = candidate
                elif best_nonbreaking_candidate is None or row["target_final_mae"] > best_nonbreaking_candidate["row"]["target_final_mae"]:
                    best_nonbreaking_candidate = candidate
                if feasible:
                    if best_feasible is None or row["target_layer1_mae"] < best_feasible["row"]["target_layer1_mae"]:
                        best_feasible = candidate
            (args.out_dir / "latest_replay.json").write_text(json.dumps(row, indent=2))
        else:
            print(
                f"step {step:04d} loss={row['loss']:.3f} "
                f"target_final={row['target_final_mae']:.3f} target_l1={row['target_layer1_mae']:.3f} "
                f"active_replay={row['replay_active_max_mae']:.3f}",
                flush=True,
            )
        rows.append(row)

    torch.save(delta.detach().cpu(), args.out_dir / "final_delta_logits.pt")
    with (args.out_dir / "search_trace.csv").open("w", newline="") as f:
        fields = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    best = best_feasible if best_feasible is not None else best_break_candidate
    if best is None:
        best = best_nonbreaking_candidate
    if best is not None:
        torch.save(best["delta"], args.out_dir / "best_delta_logits.pt")
        if best_feasible is not None:
            best["selection_status"] = "feasible"
        elif best_break_candidate is not None:
            best["selection_status"] = "best_breaking_target_candidate"
        else:
            best["selection_status"] = "best_nonbreaking_target_candidate"
        (args.out_dir / "best_summary.json").write_text(
            json.dumps({"best": {k: v for k, v in best.items() if k != "delta"}}, indent=2)
        )
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
