# GFANC source layout

- `data/`: synthetic audio, label generation, filter decomposition, pretraining
- `training/`: datasets, 1D CNN architectures, training and evaluation
- `anc/`: controller selection, FxLMS/SFANC and noise cancellation
- `common/`: shared signal and audio helpers

Run notebooks from anywhere under this repository. Their first cell locates the project root, adds the workflow source directories to `sys.path`, and sets the project root as the working directory so `models/`, `Pz and Sz/`, and `Real Noise Examples/` paths continue to resolve from the repository root.

Non-identical implementations carry `_v1` / `_v2` suffixes where both are kept. Exact byte-identical files were consolidated.
