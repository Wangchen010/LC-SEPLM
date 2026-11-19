# LC-SEPLM

Code and selected result artifacts for the manuscript *Long-range contact supervision enriches global structural representations in protein language models*. This repository documents the training implementation, evaluation workflows, selected result tables, and source data for figures.

> **Release scope:** This is a source-code and selected-results release. It does not include raw third-party databases, pretrained ESM weights, LC-SEPLM checkpoints, or every processed input manifest. As a result, the full paper cannot be reproduced from this repository alone. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and [`docs/DATA_AND_SOFTWARE_NOTICES.md`](docs/DATA_AND_SOFTWARE_NOTICES.md).

## Repository layout

- `code/` — training, data-preparation, evaluation, and result-summary scripts.
- `figure_generation/` — standalone scripts for the included architecture and filtering figures.
- `results/main_tables/` — selected manuscript tables and diagnostics.
- `results/figure_source_data/` — selected numerical figure source data.
- `results/training_run/` — run configuration and compact training history metadata; no model checkpoint.
- `docs/` — environment recovery, provenance, data/software notices, and reproducibility limits.
- `environment.graphgpu.observed.yml` — snapshot of the currently available Conda environment; it is not confirmed as an exact historical lockfile.

## Method snapshot 

The reported model adapts ESM2 t30 150M with LoRA on query/value projections in layers 22–29, using long-range contact supervision and a masked-language-modeling objective. The manuscript reports epoch 10 of a 500,000-protein training run as the selected checkpoint. Configuration and summarized metrics are included, but the checkpoint itself is not.

## Setup

The environment snapshot is provided as a recovery reference. The original launcher used an environment named `graphgpu`; the currently available environment reports Python 3.10.20, PyTorch 2.5.1+cu121 and fair-esm 2.0.0. The local ESM source used during some work reports version 2.0.1. The exact source/version used for the historical training run must be confirmed; details are in [`docs/ENVIRONMENT_RECOVERY.md`](docs/ENVIRONMENT_RECOVERY.md).

The scripts generally require Python, PyTorch, NumPy, pandas and scikit-learn. Some workflows also need Biopython, Matplotlib, Seaborn, MMseqs2, GPU access, external datasets and model weights. Their requirements and access conditions vary by script; this repository does not yet provide a fully pinned installable environment or a single end-to-end reproduction command.

The figure scripts read bundled source data or use fixed package-relative inputs and write outputs under `results/generated_figures/`. They can be run from the repository root:

```bash
python figure_generation/build_figure1_v8_clean.py
python figure_generation/build_filtering_figures_v13.py
```

These regenerate two figure groups only. They do not reproduce every manuscript table or figure.

## Data and model access

Raw AlphaFold/UniProt/PDB databases, benchmark inputs, pretrained ESM2/ESM-S weights, and LC-SEPLM checkpoints are not included. Obtain third-party resources from their original providers, follow their current terms, and record the versions used. See [`docs/DATA_AND_SOFTWARE_NOTICES.md`](docs/DATA_AND_SOFTWARE_NOTICES.md).

## Reproducibility

The repository provides code and selected reported outputs for inspection. Exact processed train/validation/test manifests, all preprocessing provenance, some external inputs, and the trained model weights are not present. The code has not been validated by a full clean-room rerun of the 500,000-protein training and all benchmark evaluations. Please treat the result tables as reported artifacts, not as results independently regenerated from this repository.

## Citation

Please cite the associated LC-SEPLM manuscript. Add its final bibliographic details and this repository's permanent release identifier once assigned.
## License

Original LC-SEPLM code and its documentation in this repository are released under the [MIT License](LICENSE). This permission does not extend to third-party software, datasets, pretrained model weights, or other materials subject to their own terms. See [`docs/DATA_AND_SOFTWARE_NOTICES.md`](docs/DATA_AND_SOFTWARE_NOTICES.md) before redistributing external or derived resources.

## Contact
For questions about this work, please contact the corresponding author, Qinghua Cui: [cuiqinghua@bjmu.edu.cn](mailto:cuiqinghua@bjmu.edu.cn).
