#!/usr/bin/env bash
# benchmark_one_protein.sh — Run the full convergence-diagnostic pipeline on one protein.
#
# Per-protein steps:
#   1. Extract sequence AND Cα distance matrix from ground-truth PDB
#      (uses PDB-derived sequence so length always matches the structure)
#   2. ESMFold forward + cache z + compute pLDDT
#   3. Train 47 per-layer distogram probes (skipped if all probe files exist)
#   4. Generate layerwise MDS structures (one per block) from probe distograms
#   5. Compute layerwise distance correlations
#
# Resumable: each step checks for existing outputs and skips if found.
# Per-protein failures don't kill the batch.

set -euo pipefail

# ── Per-protein args ──────────────────────────────────────────────────────────
NAME="${NAME:?NAME required}"
SEQ_REGISTRY="${SEQ:-}"   # registry sequence is now just a hint; we extract from PDB
GT_PDB="${GT_PDB:?GT_PDB required}"
GT_CHAIN="${GT_CHAIN:- }"

# ── Training hyperparams ──────────────────────────────────────────────────────
STEPS_PER_LAYER="${STEPS_PER_LAYER:-400}"
PAIRS_PER_STEP="${PAIRS_PER_STEP:-8192}"
PROBE_HIDDEN_DIM="${PROBE_HIDDEN_DIM:-256}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-1.0}"
DISTOGRAM_WEIGHT="${DISTOGRAM_WEIGHT:-5.0}"

RUN_PREFIX="${RUN_PREFIX:-benchmark}"
RUN_DIR="runs/${RUN_PREFIX}_${NAME}"
CACHE_PATH="$RUN_DIR/esmfold_activations.pt"
GT_NPZ="$RUN_DIR/gt_ca_distances.npz"
SEQ_FILE="$RUN_DIR/sequence.txt"

mkdir -p "$RUN_DIR/analysis"

# ── 1. Extract sequence + Cα distances from PDB ──────────────────────────────
echo "[$NAME] step 1/5: extract sequence + Cα distances from $GT_PDB"
uv run python - <<EOF
"""Extract resolved-residue sequence AND distance matrix from PDB.
The key invariant: sequence length == distance-matrix size, always.
Picks the chain with most CA atoms (overridden by GT_CHAIN if non-empty)."""
import numpy as np, sys
from collections import Counter
from pathlib import Path

AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
       "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
       "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
       "MSE":"M","SEC":"U","PYL":"O"}

# First pass: tally CA atoms per chain
chain_counts = Counter()
with open("$GT_PDB") as f:
    for ln in f:
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            chain_counts[ln[21]] += 1

if not chain_counts:
    print("  ERROR: no CA atoms found"); sys.exit(1)

# Pick chain: user-specified if it has CA atoms, else the most-populated chain
chain_hint = "$GT_CHAIN".strip() or " "
if chain_hint in chain_counts and chain_counts[chain_hint] >= 20:
    chain = chain_hint
else:
    chain = chain_counts.most_common(1)[0][0]
    print(f"  using auto-selected chain '{chain}' "
          f"(hint was '{chain_hint}', has {chain_counts.get(chain_hint, 0)} CA)")

# Second pass: extract sequence + coords for that chain only
seq_chars = []
coords = []
seen = set()
with open("$GT_PDB") as f:
    for ln in f:
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[21] == chain:
            rs = (int(ln[22:26]), ln[26])  # resnum + insertion code
            if rs in seen: continue
            seen.add(rs)
            aa3 = ln[17:20].strip()
            seq_chars.append(AA3.get(aa3, "X"))
            coords.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])

coords = np.array(coords)
seq = "".join(seq_chars)
# Keep coords and seq aligned by construction — same length, same order
assert len(seq) == len(coords), f"seq {len(seq)} vs coords {len(coords)} mismatch"

print(f"  parsed {len(coords)} Cα atoms on chain '{chain}' (length {len(seq)})")
diff = coords[:, None, :] - coords[None, :, :]
D = np.sqrt((diff ** 2).sum(-1))
np.savez("$GT_NPZ", distances=D, ca_coords=coords, sequence=seq)
Path("$SEQ_FILE").write_text(seq)
print(f"  saved {D.shape} distance matrix to $GT_NPZ")
print(f"  saved sequence ({len(seq)} aa) to $SEQ_FILE")
EOF

# ── Load the PDB-derived sequence for downstream steps ───────────────────────
SEQ=$(cat "$SEQ_FILE")
echo "[$NAME] PDB-derived sequence: ${SEQ:0:60}... (${#SEQ} aa)"

# ── 2. Check if probe training is already done ───────────────────────────────
TRAIN_NEEDED=1
if [ -f "$CACHE_PATH" ] && [ -d "$RUN_DIR/layer_47" ] && [ -d "$RUN_DIR/layer_01" ]; then
    # Count completed probe layers
    n_done=$(ls -d "$RUN_DIR"/layer_*/probe.pt 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n_done" = "47" ]; then
        echo "[$NAME] step 2/5: probes already trained ($n_done/47 layers) — skipping"
        TRAIN_NEEDED=0
    else
        echo "[$NAME] step 2/5: incomplete training ($n_done/47 layers) — restarting"
    fi
fi

if [ "$TRAIN_NEEDED" = "1" ]; then
    echo "[$NAME] step 2/5: ESMFold + 47-layer probe training ($STEPS_PER_LAYER steps/layer)"
    uv run python -m recurrent_folding.train_esmfold_distogram_probes \
        --sequence "$SEQ" \
        --target-distance-npz "$GT_NPZ" \
        --run-dir "$RUN_DIR" \
        --activation-cache "$CACHE_PATH" \
        --start-layer 47 \
        --num-layers 47 \
        --steps-per-layer "$STEPS_PER_LAYER" \
        --pairs-per-step "$PAIRS_PER_STEP" \
        --probe-hidden-dim "$PROBE_HIDDEN_DIM" \
        --distogram-weight "$DISTOGRAM_WEIGHT" \
        --consistency-weight "$CONSISTENCY_WEIGHT" \
        --device cuda
fi

# ── 3. ESMFold-direct accuracy + pLDDT (skip if done) ────────────────────────
METRICS_JSON="$RUN_DIR/analysis/esmfold_direct_eval/metrics.json"
if [ -f "$METRICS_JSON" ]; then
    echo "[$NAME] step 3/5: pLDDT already computed — skipping"
else
    echo "[$NAME] step 3/5: ESMFold-direct evaluation"
    uv run python - <<EOF
import json, torch, numpy as np
from pathlib import Path
from transformers import AutoTokenizer, EsmForProteinFolding

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to("cuda").eval()
model.esm = model.esm.half()

ids = tokenizer(["$SEQ"], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
with torch.no_grad():
    out = model(ids)

plddt = out["plddt"][0, :, 1].cpu().float()
mean_plddt = float(plddt.mean())
if mean_plddt > 1.5:
    mean_plddt /= 100.0; plddt = plddt / 100.0

pred_ca = out["positions"][-1, 0, :, 1, :].cpu().numpy()
gt = np.load("$GT_NPZ")
gt_ca = gt["ca_coords"]
n = min(len(pred_ca), len(gt_ca))
P, Q = pred_ca[:n], gt_ca[:n]
Pc, Qc = P - P.mean(0), Q - Q.mean(0)
H = Pc.T @ Qc; U,_,Vt = np.linalg.svd(H)
d = np.sign(np.linalg.det(Vt.T @ U.T))
P_aligned = Pc @ (Vt.T @ np.diag([1,1,d]) @ U.T).T + Q.mean(0)
rmsd = float(np.sqrt(((P_aligned - Q)**2).sum(-1).mean()))

def pdist(c):
    dd = c[:,None]-c[None]; return np.sqrt((dd**2).sum(-1))
iu = np.triu_indices(n, k=1)
Dp, Dt = pdist(P)[iu], pdist(Q)[iu]

m = {
    "target": "$NAME",
    "length": int(len(plddt)),
    "ca_rmsd_after_kabsch": rmsd,
    "distance_mae_all_pairs": float(np.abs(Dp - Dt).mean()),
    "distance_corr_all_pairs": float(np.corrcoef(Dp, Dt)[0,1]),
    "mean_plddt": mean_plddt,
    "plddt_min": float(plddt.min()),
    "plddt_med": float(plddt.median()),
    "plddt_max": float(plddt.max()),
}
out_dir = Path("$RUN_DIR/analysis/esmfold_direct_eval"); out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
np.savez(out_dir / "esmfold_prediction_ca.npz", pred_ca=pred_ca)
print(f"  RMSD={rmsd:.2f}A  pLDDT={mean_plddt:.3f}")
EOF
fi

# ── 4. MDS reconstruction (skip if already done) ─────────────────────────────
MDS_PDB="$RUN_DIR/analysis/mds_structure_video_fixed/distogram_mds_models.pdb"
if [ -f "$MDS_PDB" ]; then
    echo "[$NAME] step 4/5: MDS structures already exist — skipping"
else
    echo "[$NAME] step 4/5: layerwise MDS reconstruction"
    uv run python - <<EOF
"""Standalone MDS reconstruction — extracts distance matrices from probe distograms
and uses sklearn MDS or torch-based SMACOF to embed each layer in 3D."""
import torch, numpy as np
from pathlib import Path
from sklearn.manifold import MDS
from recurrent_folding.esmfold_probes import PairDistogramBottleneck

ESMFOLD_DISTO_MIN = 2.3125
ESMFOLD_DISTO_MAX = 21.6875
ESMFOLD_DISTO_BINS = 64

def bin_centers(device, dtype):
    edges = torch.linspace(ESMFOLD_DISTO_MIN, ESMFOLD_DISTO_MAX, ESMFOLD_DISTO_BINS + 1,
                            device=device, dtype=dtype)
    return ((edges[:-1] + edges[1:]) * 0.5).view(1, -1)

# Load z cache
data = torch.load("$CACHE_PATH", map_location="cpu", weights_only=False)
z_all = data["z"].squeeze(1).float()  # (48, L, L, 128)
L = z_all.shape[1]
pair_dim = z_all.shape[-1]
print(f"  z shape: {tuple(z_all.shape)}")

bc = bin_centers(torch.device("cpu"), torch.float32).squeeze(0)
out_dir = Path("$RUN_DIR/analysis/mds_structure_video_fixed")
out_dir.mkdir(parents=True, exist_ok=True)

# Load probes per layer, compute E[d] matrices, MDS-embed
ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
i_t = torch.from_numpy(ii.flatten()).long()
j_t = torch.from_numpy(jj.flatten()).long()

pdb_lines = []
prev_coords = None
for layer in range(1, 48):
    probe_path = Path("$RUN_DIR") / f"layer_{layer:02d}" / "probe.pt"
    if not probe_path.exists():
        continue
    probe = PairDistogramBottleneck(pair_dim=pair_dim, bins=ESMFOLD_DISTO_BINS, hidden_dim=$PROBE_HIDDEN_DIM)
    probe.load_state_dict(torch.load(probe_path, map_location="cpu", weights_only=True))
    probe.eval()
    for p in probe.parameters(): p.requires_grad_(False)

    z_layer = z_all[layer - 1]  # layers indexed 1..47, z indexed 0..47
    pairs = z_layer.reshape(-1, pair_dim)
    with torch.no_grad():
        logits = probe.encode(pairs, i_t, j_t, L)
    ed = (torch.softmax(logits, dim=-1) * bc).sum(-1).reshape(L, L).numpy()
    ed = 0.5 * (ed + ed.T)
    np.fill_diagonal(ed, 0.0)

    # MDS embed using sklearn (deterministic init from previous layer for continuity)
    mds = MDS(n_components=3, dissimilarity="precomputed", n_init=1, max_iter=200,
              random_state=42, normalized_stress=False)
    if prev_coords is not None:
        # Use previous as init by feeding into the embedding via SMACOF approx
        coords_3d = mds.fit_transform(ed, init=prev_coords)
    else:
        coords_3d = mds.fit_transform(ed)
    prev_coords = coords_3d

    pdb_lines.append(f"MODEL    {layer:5d}  LAYER {layer}")
    for i in range(L):
        x, y, z = coords_3d[i]
        pdb_lines.append(
            f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    pdb_lines.append("ENDMDL")

(out_dir / "distogram_mds_models.pdb").write_text("\n".join(pdb_lines) + "\n")
print(f"  wrote {out_dir / 'distogram_mds_models.pdb'}")
EOF
fi

# ── 5. Layerwise distance correlations ───────────────────────────────────────
CORR_CSV="$RUN_DIR/analysis/layerwise_distance_correlations.csv"
if [ -f "$CORR_CSV" ]; then
    echo "[$NAME] step 5/5: correlations already computed — skipping"
else
    echo "[$NAME] step 5/5: layerwise correlation analysis"
    uv run python - <<EOF
import numpy as np, csv
from pathlib import Path

pdb_path = "$MDS_PDB"
models, cur = [], None
with open(pdb_path) as f:
    for ln in f:
        if ln.startswith("MODEL"):    cur = []
        elif ln.startswith("ENDMDL"): models.append(cur); cur = None
        elif ln.startswith("ATOM") and cur is not None:
            cur.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
coords = np.array(models)
L, R = coords.shape[:2]
iu = np.triu_indices(R, k=1)
def pdist(c):
    dd = c[:,None]-c[None]; return np.sqrt((dd**2).sum(-1))
D_layers = [pdist(c)[iu] for c in coords]
D_final = D_layers[-1]
D_centroid = np.mean(D_layers, axis=0)

with open("$CORR_CSV", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer","corr_vs_47","dist_mae_vs_47","coord_rmsd_vs_47",
                "corr_vs_centroid","dist_mae_vs_centroid"])
    for t in range(L):
        c47 = float(np.corrcoef(D_layers[t], D_final)[0,1])
        m47 = float(np.abs(D_layers[t]-D_final).mean())
        cc  = float(np.corrcoef(D_layers[t], D_centroid)[0,1])
        mc  = float(np.abs(D_layers[t]-D_centroid).mean())
        w.writerow([t+1, c47, m47, "", cc, mc])
print(f"  wrote $CORR_CSV ({L} layers)")
EOF
fi

echo "[$NAME] DONE  ->  $RUN_DIR"
