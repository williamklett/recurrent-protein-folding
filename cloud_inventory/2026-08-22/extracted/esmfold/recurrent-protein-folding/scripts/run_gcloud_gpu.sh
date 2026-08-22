#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-rugged-weaver-406003}"
INSTANCE="${INSTANCE:-psa-l4}"
ZONE="${ZONE:-us-central1-a}"
REMOTE_DIR="${REMOTE_DIR:-~/recurrent-protein-folding}"

gcloud compute ssh "$INSTANCE" --project "$PROJECT" --zone "$ZONE" --command "mkdir -p $REMOTE_DIR"
gcloud compute scp --recurse \
  pyproject.toml README.md recurrent_folding scripts \
  "$INSTANCE:$REMOTE_DIR" \
  --project "$PROJECT" --zone "$ZONE"

gcloud compute ssh "$INSTANCE" --project "$PROJECT" --zone "$ZONE" --command "
  set -euo pipefail
  cd $REMOTE_DIR
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=\"\$HOME/.local/bin:\$PATH\"
  fi
  export PATH=\"\$HOME/.local/bin:\$PATH\"
  uv run python -m recurrent_folding.train \
    --steps \"${TRAIN_STEPS:-2000}\" \
    --batch-size \"${BATCH_SIZE:-128}\" \
    --recurrent-steps \"${RECURRENT_STEPS:-8}\" \
    --max-delta \"${MAX_DELTA:-1.25}\" \
    --reach-max-step \"${REACH_MAX_STEP:-1.35}\" \
    --device cuda \
    --run-dir \"${RUN_DIR:-runs/toy_lstm_gpu}\"
"
