# Recurrent Protein Folding

Toy scaffold for an interpretable iterative folding model.

Core idea:

- A latent/bioinformatic channel can carry abstract sequence pressure.
- A physical/action channel is the only channel decoded into coordinates.
- Every decoded step must stay chain-coherent.
- The final physical channel is trained against the target fold.

Run locally:

```bash
uv run python -m recurrent_folding.train --steps 200 --batch-size 16 --device cpu
```

Run on CUDA:

```bash
uv run python -m recurrent_folding.train --steps 2000 --batch-size 128 --device cuda
```

Outputs are written to `runs/`.

Small real-data pilot with one mdCATH domain:

```bash
uv run python -m recurrent_folding.train_mdcath \
  --mdcath-path data/mdcath/mdcath_dataset_1ha8A00.h5 \
  --steps 500 \
  --batch-size 32 \
  --window 8 \
  --architecture transformer \
  --transformer-dim 128 \
  --transformer-depth 4 \
  --transformer-heads 4
```

List available trajectories inside a downloaded mdCATH HDF5 file:

```bash
uv run python -m recurrent_folding.train_mdcath \
  --mdcath-path data/mdcath/mdcath_dataset_1ha8A00.h5 \
  --list
```
