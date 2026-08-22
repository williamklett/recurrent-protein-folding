#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
mkdir -p runs/replication_logs
log(){ echo "[$(date -Is)] $*"; }
log "replication suite start"
log "manifest $(wc -l data/manifest.csv)"
log "stage 1 cache_activations"
python3 scripts/cache_activations.py 2>&1 | tee runs/replication_logs/01_cache_activations.log
log "stage 2 extract_block_distograms"
python3 scripts/extract_block_distograms.py 2>&1 | tee runs/replication_logs/02_extract_block_distograms.log
log "stage 3 pq_signature"
python3 scripts/analyze_pq_signature.py 2>&1 | tee runs/replication_logs/03_pq_signature.log
log "stage 4 signed_vs_psd_descriptive"
python3 scripts/test_signed_vs_psd_compression.py 2>&1 | tee runs/replication_logs/04_signed_vs_psd_compression.log
log "stage 5 signature_projection_grid"
python3 scripts/test_signature_projection_grid.py --max-val 60 --blocks 5,9,13,17,21,25,29,33,35,39,43,47 --ranks 8,16,32 --out signature_projection_grid_n300.json 2>&1 | tee runs/replication_logs/05_signature_projection_grid.log
log "stage 6 adapter_capacity_sweep"
python3 scripts/test_adapter_capacity_sweep.py --max-train-proteins 220 --max-val-proteins 60 --epochs 30 --hidden-sizes 0,64,256,1024,4096 --out adapter_capacity_sweep_n300.json 2>&1 | tee runs/replication_logs/06_adapter_capacity_sweep.log
log "stage 7 k2_sweep"
python3 scripts/sweep_k2_blocks.py 2>&1 | tee runs/replication_logs/07_k2_sweep.log
mv runs/k2_sweep_summary.json runs/k2_sweep_summary_n300.json
mv runs/k2_sweep_summary.csv runs/k2_sweep_summary_n300.csv
log "stage 8 k5_sweep"
python3 scripts/sweep_all_blocks.py 2>&1 | tee runs/replication_logs/08_k5_sweep.log
mv runs/sweep_summary.json runs/sweep_summary_n300.json
mv runs/sweep_summary.csv runs/sweep_summary_n300.csv
log "replication suite complete"
df -h .
