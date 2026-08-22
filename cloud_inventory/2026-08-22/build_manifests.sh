#!/usr/bin/env bash
set -euo pipefail

OUT=/tmp/codex-recovery-manifests
ESMFOLD_HOME=/mnt/recovery/esmfold/home/williamklett
PSA_ONDEMAND_HOME=/mnt/recovery/psa-ondemand/home/williamklett
PSA_REPL_HOME=/mnt/recovery/psa-repl/home/williamklett

mkdir -p "$OUT"

user_manifest() {
  local root=$1
  local output=$2
  sudo find "$root" -xdev \
    \( -path "$root/.ssh" -o \
       -path "$root/.config/gcloud" -o \
       -path "$root/.gsutil" \) -prune -o \
    -printf '%y\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%P\n' \
    | LC_ALL=C sort > "$output"
}

project_manifest() {
  local root=$1
  local output=$2
  sudo find "$root" -xdev \
    -printf '%y\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%P\n' \
    | LC_ALL=C sort > "$output"
}

source_checksums() {
  local root=$1
  local output=$2
  sudo find "$root" -xdev \
    \( -path '*/.venv' -o \
       -path '*/venv' -o \
       -path '*/.cache' -o \
       -path '*/__pycache__' -o \
       -path '*/.git' -o \
       -path '*/data' \) -prune -o \
    -type f -size -100M \
    \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o \
       -name '*.toml' -o -name '*.yaml' -o -name '*.yml' -o \
       -name '*.tex' -o -name '*.bib' -o -name '*.json' -o \
       -name '*.jsonl' -o -name '*.csv' -o -name '*.tsv' -o \
       -name '*.ipynb' \) \
    -exec sha256sum {} + \
    | sed "s#  $root/#  #" \
    | LC_ALL=C sort > "$output"
}

user_manifest "$ESMFOLD_HOME" "$OUT/esmfold_user_tree.tsv"
user_manifest "$PSA_ONDEMAND_HOME" "$OUT/psa_ondemand_user_tree.tsv"
user_manifest "$PSA_REPL_HOME" "$OUT/psa_repl_user_tree.tsv"

project_manifest "$ESMFOLD_HOME/recurrent-protein-folding" \
  "$OUT/esmfold_recurrent_project.tsv"
project_manifest "$PSA_ONDEMAND_HOME/psa" \
  "$OUT/psa_ondemand_project.tsv"
project_manifest "$PSA_REPL_HOME/psa" \
  "$OUT/psa_repl_project.tsv"

source_checksums "$ESMFOLD_HOME/recurrent-protein-folding" \
  "$OUT/esmfold_recurrent_source_sha256.txt"
source_checksums "$PSA_ONDEMAND_HOME/psa" \
  "$OUT/psa_ondemand_source_sha256.txt"
source_checksums "$PSA_REPL_HOME/psa" \
  "$OUT/psa_repl_source_sha256.txt"

sudo find "$PSA_ONDEMAND_HOME/psa/data" -xdev -type f \
  -printf '%P\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\n' \
  | LC_ALL=C sort > "$OUT/psa_ondemand_data.tsv"
sudo find "$PSA_REPL_HOME/psa/data" -xdev -type f \
  -printf '%P\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\n' \
  | LC_ALL=C sort > "$OUT/psa_repl_data.tsv"

sudo diff -qr --exclude=data --exclude=__pycache__ \
  "$PSA_REPL_HOME/psa" "$PSA_ONDEMAND_HOME/psa" \
  > "$OUT/psa_snapshots_no_data.diff" || true

sudo tar -C "$ESMFOLD_HOME" -czf "$OUT/esmfold_recurrent_no_venv.tar.gz" \
  --exclude='recurrent-protein-folding/.venv' \
  --exclude='recurrent-protein-folding/**/__pycache__' \
  --exclude='recurrent-protein-folding/.cache' \
  recurrent-protein-folding

sudo tar -C "$PSA_ONDEMAND_HOME" -czf "$OUT/psa_ondemand_no_data.tar.gz" \
  --exclude='psa/data' --exclude='psa/**/__pycache__' psa
sudo tar -C "$PSA_REPL_HOME" -czf "$OUT/psa_repl_no_data.tar.gz" \
  --exclude='psa/data' --exclude='psa/**/__pycache__' psa

sudo cp --preserve=timestamps \
  "$PSA_ONDEMAND_HOME/psa_replication_inputs.tgz" \
  "$OUT/psa_replication_inputs.tgz"

sudo chown -R "$(id -u):$(id -g)" "$OUT"
sha256sum "$OUT"/*.tar.gz "$OUT"/psa_replication_inputs.tgz \
  > "$OUT/recovered_archives_sha256.txt"

printf '%s\n' \
  'Generated from three snapshot-derived disks mounted read-only with journal replay disabled.' \
  'Credential stores and SSH directories were excluded from user-tree manifests.' \
  'Project manifests include dependency environments; recovery archives exclude them.' \
  'The PSA recovery archives omit the 74-GB data tree, which is fully enumerated separately.' \
  > "$OUT/README.txt"

ls -lh "$OUT"
