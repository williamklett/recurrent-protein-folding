# Google Cloud code and file recovery report

Date: 2026-08-22

Account: `williamklett0808@gmail.com`

Primary project: `rugged-weaver-406003`

## Outcome

All three archived Compute Engine boot disks were restored to temporary disks,
attached to an isolated CPU-only VM in read-only mode, mounted with journal
replay disabled, inventoried, and packaged. The temporary VM and all temporary
disks were deleted after the local artifacts were checksum-verified. The three
original snapshots remain `READY`; `kalshi-wt-micro` and `kalshi-east` remain
`RUNNING` and were not modified.

## June ESMFold archive

Snapshot: `esmfold-archive-jun10`

Original path: `/home/williamklett/recurrent-protein-folding`

- Full project manifest: 27,824 regular files and 4.970 GiB, almost entirely
  the archived `.venv`.
- Recoverable project content without the dependency environment: 29 MiB on
  disk and 6.8 MiB compressed, including source, logs, 20 MiB of OOD PDBs, and
  the paper figures.
- Source/document checksum list: 144 files.
- The complete archived user home contained 150,573 regular files and 43.836
  GiB after credential stores were excluded from the manifest.
- The same snapshot also contains approximately 33.1 GiB under `kalshi`, plus
  smaller `kalshi_wt`, `kalshi_ts`, and standalone diagnostic files.

Comparison with the current workspace found 28 archive-only entries, 9
current-only entries, and 4 common files with different content. Important
archive-only scripts include the `belief_gd*` series, `belief_manifold.py`,
`code_correction.py`, `cosine_natural_grad.py`, `decoder_inversion.py`,
`decouple_allfolds.py`, `energy_decomp.py`, `energy_traj.py`,
`eval_l45_vs_new_l46.py`, `fit_transport.py`, `inspect_head.py`,
`make_paper_figs.py`, `meaning_drift.py`, `min_interatomic_distance.py`, and
`transport_fidelity.py`. Archive-only outputs include the original
`paper_figs`, OOD500 PDB set and manifests, `benchmark_overnight.local.log`,
`residual_analysis.log`, and `vm_60_consistent.json`.

The four shared files with changed content are `pyproject.toml`, `uv.lock`,
`scripts/compare_500_metrics.py`, and
`scripts/cosine_alignment_backprop.py`. No recovered files were merged over
the current workspace.

## May PSA archives

Snapshots:

- `psa-l4-ondemand-inspect-20260519-124941`
- `psa-l4-repl-20260519-092306`

Original path: `/home/williamklett/psa`

- Each project contains 2,432 data files totaling 73.105 GiB.
- `activations`: 352 files, 71.967 GiB.
- `distograms`: 300 files, 0.986 GiB.
- `physics`: 627 files, 0.077 GiB.
- `replication`: 1,151 files, 0.074 GiB.
- The two data trees match by relative path and byte size.
- The later on-demand snapshot contains the additional stage-5 resume script
  and stage-5 logs/results. Otherwise, the source trees match; two result files
  differ because the later run progressed further.
- A standalone 64 MiB `psa_replication_inputs.tgz` was recovered and verified
  identical in both snapshots. It expands to 587 input files.

The 73-GiB activation corpus was not downloaded, but every data file is listed
with its exact relative path, size, and modification timestamp. Both original
snapshots remain available if the tensors themselves are needed later.

## Live machines and Cloud Shell

- `kalshi-wt-micro`: 444 non-dependency, non-cache, non-credential files in
  the local manifest, totaling 297.64 MiB.
- `kalshi-east`: 11 corresponding files totaling 0.11 MiB.
- Cloud Shell: 90 non-credential files; these are almost entirely Google's
  installed Gemini/Vertex extension plus the standard Cloud Shell README.

Private keys, `.env` files, SSH directories, and Google credential stores were
excluded from local manifests and recovery bundles.

## Other project storage

The account exposes three active projects. No Cloud Storage buckets or
BigQuery datasets were present in any of them, and the project with Dataform
enabled contained no repositories. No custom Compute Engine images or machine
images were present. Cloud Asset Inventory was not enabled and was not enabled
during this audit; relevant enabled file-bearing services were queried
directly.

## Local artifacts

- `codex-recovery-manifests/esmfold_recurrent_no_venv.tar.gz`: recovered June
  ESMFold project without its dependency environment.
- `extracted/esmfold/recurrent-protein-folding/`: extracted copy of that
  recovery archive.
- `current_vs_esmfold_archive.diff`: exact high-level comparison with the
  current workspace.
- `codex-recovery-manifests/psa_ondemand_no_data.tar.gz`: later PSA source and
  result tree without the 73-GiB data directory.
- `codex-recovery-manifests/psa_replication_inputs.tgz`: recovered replication
  inputs.
- `codex-recovery-manifests/psa_ondemand_data.tsv`: complete 2,432-file PSA
  data inventory.
- `*_user_tree.tsv`: broad snapshot-home manifests excluding credential
  stores.
- `kalshi-wt-micro_user_files.tsv`, `kalshi-east_user_files.tsv`, and
  `cloud-shell_user_files.tsv`: current remote manifests.
- `gcloud_*.json`: final project, instance, disk, and snapshot metadata after
  cleanup.
