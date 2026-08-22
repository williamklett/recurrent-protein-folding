#!/usr/bin/env bash
# Run the synthetic-design OOD benchmark panel on the GPU instance.

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

MANIFEST="${MANIFEST:-data/synthetic_designs_manifest.tsv}"
RUN_PREFIX="${RUN_PREFIX:-synthetic_design}"
STEPS_PER_LAYER="${STEPS_PER_LAYER:-400}"
PAIRS_PER_STEP="${PAIRS_PER_STEP:-8192}"
PROBE_HIDDEN_DIM="${PROBE_HIDDEN_DIM:-256}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-1.0}"
DISTOGRAM_WEIGHT="${DISTOGRAM_WEIGHT:-5.0}"
LOG="runs/${RUN_PREFIX}_benchmark.log"

mkdir -p runs

echo "============================================================" | tee -a "$LOG"
echo "SYNTHETIC DESIGN OOD BENCHMARK" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "Manifest: $MANIFEST" | tee -a "$LOG"
echo "Run prefix: $RUN_PREFIX" | tee -a "$LOG"
echo "Steps/layer: $STEPS_PER_LAYER" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

if [ ! -f "$MANIFEST" ]; then
    echo "Missing manifest: $MANIFEST" | tee -a "$LOG"
    exit 1
fi

success_list=()
fail_list=()

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r NAME GT_PDB GT_CHAIN CLASS PDB_ID LENGTH RELEASE METHOD TITLE SOURCE; do
    [ -n "${NAME:-}" ] || continue
    echo "" | tee -a "$LOG"
    echo "────────────────────────────────────────────────────────────" | tee -a "$LOG"
    echo "[$NAME] pdb=$PDB_ID chain=$GT_CHAIN length=$LENGTH release=$RELEASE class=$CLASS" | tee -a "$LOG"
    echo "title: $TITLE" | tee -a "$LOG"

    if [ ! -f "$GT_PDB" ]; then
        echo "  SKIP: ground-truth PDB not found at $GT_PDB" | tee -a "$LOG"
        fail_list+=("$NAME missing_pdb")
        continue
    fi

    if RUN_PREFIX="$RUN_PREFIX" \
       STEPS_PER_LAYER="$STEPS_PER_LAYER" \
       PAIRS_PER_STEP="$PAIRS_PER_STEP" \
       PROBE_HIDDEN_DIM="$PROBE_HIDDEN_DIM" \
       CONSISTENCY_WEIGHT="$CONSISTENCY_WEIGHT" \
       DISTOGRAM_WEIGHT="$DISTOGRAM_WEIGHT" \
       NAME="$NAME" SEQ="" GT_PDB="$GT_PDB" GT_CHAIN="$GT_CHAIN" \
       bash scripts/benchmark_one_protein.sh 2>&1 | tee -a "$LOG"; then
        success_list+=("$NAME")
        echo "  done $NAME" | tee -a "$LOG"
    else
        fail_list+=("$NAME pipeline_error")
        echo "  failed $NAME; continuing" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "Outputs: runs/${RUN_PREFIX}_*/" | tee -a "$LOG"
