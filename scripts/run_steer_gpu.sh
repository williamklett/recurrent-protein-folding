#!/usr/bin/env bash
# run_steer_gpu.sh — Train ESMFold distogram probes on a GPU instance, then run
# the distogram steering experiment.
#
# Usage:
#   bash scripts/run_steer_gpu.sh
#
# Override any variable with an env prefix, e.g.:
#   INJECT_LAYER=35 TARGET_PAIRS="10,40+20,55" bash scripts/run_steer_gpu.sh
#
# Set REUSE_CACHE=1 to skip regenerating the ESMFold activation cache (useful
# when only rerunning probe training or steering with different hyper-params).
set -euo pipefail

# ── Instance config (mirrors run_gcloud_gpu.sh) ───────────────────────────────
PROJECT="${PROJECT:-rugged-weaver-406003}"
INSTANCE="${INSTANCE:-psa-l4}"
ZONE="${ZONE:-us-central1-a}"
REMOTE_DIR="${REMOTE_DIR:-~/recurrent-protein-folding}"

# ── Data ──────────────────────────────────────────────────────────────────────
DOMAIN="${DOMAIN:-1ha8A00}"
MDCATH_H5="data/mdcath/mdcath_dataset_1ha8A00.h5"

# ── Probe training ────────────────────────────────────────────────────────────
START_LAYER="${START_LAYER:-47}"          # train from this block downward
NUM_LAYERS="${NUM_LAYERS:-20}"            # number of layers to train probes for
STEPS_PER_LAYER="${STEPS_PER_LAYER:-1000}"
PROBE_HIDDEN_DIM="${PROBE_HIDDEN_DIM:-256}"
REUSE_CACHE="${REUSE_CACHE:-0}"           # set to 1 to skip ESMFold re-run

# ── Steering ──────────────────────────────────────────────────────────────────
INJECT_LAYER="${INJECT_LAYER:-40}"        # which trunk block to patch
TARGET_PAIRS="${TARGET_PAIRS:-10,40}"     # residue pairs to steer (use + for multiple)
TARGET_DIST="${TARGET_DIST:-}"            # target Ca-Ca Å; empty = auto (expected - 5)
STEER_STRENGTH="${STEER_STRENGTH:-8.0}"  # Gaussian logit bump height
STEER_SIGMA="${STEER_SIGMA:-2.0}"        # Gaussian sigma in Å

# ── Derived remote paths ──────────────────────────────────────────────────────
RUN_NAME="${RUN_NAME:-steer_${DOMAIN}}"
PROBE_RUN="${RUN_NAME}_probes"
CACHE_PT="${RUN_NAME}_cache.pt"
STEER_OUT="${RUN_NAME}_steering"

# ── Build optional flags (expanded locally, passed into remote command) ───────
REUSE_FLAG=""
[ "$REUSE_CACHE" = "1" ] && REUSE_FLAG="--reuse-activations"

TARGET_DIST_FLAG=""
[ -n "$TARGET_DIST" ] && TARGET_DIST_FLAG="--target-dist $TARGET_DIST"

# ─────────────────────────────────────────────────────────────────────────────
echo "==> [1/3] Syncing code and data to $INSTANCE..."
gcloud compute ssh "$INSTANCE" --project "$PROJECT" --zone "$ZONE" \
    --command "mkdir -p $REMOTE_DIR/data/mdcath $REMOTE_DIR/runs"

gcloud compute scp --recurse \
    pyproject.toml README.md recurrent_folding scripts \
    "$INSTANCE:$REMOTE_DIR" \
    --project "$PROJECT" --zone "$ZONE"

gcloud compute scp \
    "$MDCATH_H5" \
    "$INSTANCE:$REMOTE_DIR/data/mdcath/" \
    --project "$PROJECT" --zone "$ZONE"

# ─────────────────────────────────────────────────────────────────────────────
echo "==> [2/3] Running probe training + steering on $INSTANCE (GPU)..."
gcloud compute ssh "$INSTANCE" --project "$PROJECT" --zone "$ZONE" --command "
    set -euo pipefail
    cd $REMOTE_DIR

    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH=\"\$HOME/.local/bin:\$PATH\"

    echo ''
    echo '=== Step A: Train ESMFold distogram probes ==='
    echo '    Layers $START_LAYER down to $((START_LAYER - NUM_LAYERS + 1)), $STEPS_PER_LAYER steps each'
    uv run python -m recurrent_folding.train_esmfold_distogram_probes \
        --mdcath-path $MDCATH_H5 \
        --domain $DOMAIN \
        --run-dir runs/$PROBE_RUN \
        --activation-cache runs/$CACHE_PT \
        --start-layer $START_LAYER \
        --num-layers $NUM_LAYERS \
        --steps-per-layer $STEPS_PER_LAYER \
        --probe-hidden-dim $PROBE_HIDDEN_DIM \
        --distogram-weight 5.0 \
        --consistency-weight 1.0 \
        --device cuda \
        $REUSE_FLAG

    echo ''
    echo '=== Step B: Steering experiment ==='
    echo '    Inject layer: $INJECT_LAYER  |  Pairs: $TARGET_PAIRS  |  Strength: $STEER_STRENGTH'
    uv run python scripts/steer_esmfold.py \
        --probe-run-dir runs/$PROBE_RUN \
        --activation-cache runs/$CACHE_PT \
        --inject-layer $INJECT_LAYER \
        --target-pairs \"$TARGET_PAIRS\" \
        $TARGET_DIST_FLAG \
        --steer-strength $STEER_STRENGTH \
        --steer-sigma $STEER_SIGMA \
        --probe-hidden-dim $PROBE_HIDDEN_DIM \
        --output-dir runs/$STEER_OUT \
        --device cuda
"

# ─────────────────────────────────────────────────────────────────────────────
echo "==> [3/3] Downloading results..."
mkdir -p runs
gcloud compute scp --recurse \
    "$INSTANCE:$REMOTE_DIR/runs/$STEER_OUT" \
    runs/ \
    --project "$PROJECT" --zone "$ZONE"

echo ""
echo "Done. Results in runs/$STEER_OUT/"
echo "  summary.json    — per-pair distances before/after, global RMSD"
echo "  ca_baseline.npy — (L, 3) Ca coordinates, unsteered"
echo "  ca_steered.npy  — (L, 3) Ca coordinates, steered"
