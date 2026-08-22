#!/usr/bin/env bash
# run_ablation_gpu.sh — Run the probe-subspace ablation on the GPU instance.
#
# Assumes probes are already trained (e.g. by run_steer_gpu.sh).
# For each trunk block T it:
#   1. Computes z_hat = probe.decode(probe.encode(z_T))
#   2. Injects z_residual = z_T - z_hat into that block
#   3. Measures Ca-RMSD vs baseline
#
# Usage:
#   bash scripts/run_ablation_gpu.sh
#
# Override defaults:
#   PROBE_RUN=my_probes SEQUENCE=ACDE... bash scripts/run_ablation_gpu.sh

set -euo pipefail

# ── Instance config ───────────────────────────────────────────────────────────
PROJECT="${PROJECT:-rugged-weaver-406003}"
INSTANCE="${INSTANCE:-esmfold-l4-probes}"
ZONE="${ZONE:-us-west1-a}"
REMOTE_DIR="${REMOTE_DIR:-~/recurrent-protein-folding}"

# ── Experiment config ─────────────────────────────────────────────────────────
# 1ha8A00: 51-residue cysteine-rich protein used for probe training
SEQUENCE="${SEQUENCE:-GECEQCFSDGGDCTTCFNNGTGPCANCLAGYPAGCSNSDCTAFLSQCYGGC}"
PROBE_RUN="${PROBE_RUN:-steer_1ha8A00_probes}"   # must already exist on instance
PROBE_HIDDEN_DIM="${PROBE_HIDDEN_DIM:-256}"
ABLATION_OUT="${ABLATION_OUT:-ablate_probe_1ha8A00}"

# ─────────────────────────────────────────────────────────────────────────────
echo "==> [1/3] Syncing code to $INSTANCE …"
gcloud compute scp --recurse \
    pyproject.toml README.md recurrent_folding scripts \
    "$INSTANCE:$REMOTE_DIR" \
    --project "$PROJECT" --zone "$ZONE"

# ─────────────────────────────────────────────────────────────────────────────
echo "==> [2/3] Running ablation on $INSTANCE …"
echo "    Probe run : $PROBE_RUN"
echo "    Sequence  : $SEQUENCE"

gcloud compute ssh "$INSTANCE" --project "$PROJECT" --zone "$ZONE" --command "
    set -euo pipefail
    cd $REMOTE_DIR
    export PATH=\"\$HOME/.local/bin:\$PATH\"

    uv run python scripts/ablate_probe_subspace.py \
        --probe-run-dir runs/$PROBE_RUN \
        --sequence '$SEQUENCE' \
        --probe-hidden-dim $PROBE_HIDDEN_DIM \
        --output-dir runs/$ABLATION_OUT \
        --device cuda
"

# ─────────────────────────────────────────────────────────────────────────────
echo "==> [3/3] Downloading results …"
mkdir -p runs
gcloud compute scp --recurse \
    "$INSTANCE:$REMOTE_DIR/runs/$ABLATION_OUT" \
    runs/ \
    --project "$PROJECT" --zone "$ZONE"

echo ""
echo "Done. Results in runs/$ABLATION_OUT/"
echo "  summary.json                  — per-layer probe_r2 and RMSD"
echo "  ca_baseline.npy               — (L, 3) baseline Ca coordinates"
echo "  ca_ablated_layer_XX.npy       — (L, 3) Ca after residual injection at layer XX"
echo ""
echo "Key question: at which layers is RMSD >> 0 with probe_r2 >> 0?"
echo "Those layers are where the probe subspace causally drives ESMFold output."
