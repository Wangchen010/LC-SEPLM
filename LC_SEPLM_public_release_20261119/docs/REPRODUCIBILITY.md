# Reproducibility scope

This release is intended to make the study's code and selected reported artifacts inspectable. It is not a complete, one-command reproduction package.

## Included

- The main ESM2-LoRA training implementation and local helper modules used by candidate code.
- Selected evaluation and data-processing scripts.
- Selected manuscript tables, numerical figure source data, and compact training metadata.
- Two figure-generation workflows that can be run using files included in this repository.
- A snapshot of the presently available `graphgpu` environment, with uncertainty about historical version parity documented.

## Not included

- The exact processed train, validation, and test manifests and all intermediate input files.
- Raw external databases and benchmark datasets.
- Pretrained ESM2/ESM-S weights or the LC-SEPLM checkpoint/adapter weights.
- Some external model/data access instructions, checksums, and complete preprocessing provenance.
- A tested, fully pinned environment and full end-to-end command sequence for all reported experiments.

The included source files and metrics have not been used to rerun the full training and all manuscript evaluations from a clean environment. Users should obtain external inputs under their providers' terms and confirm exact split, model, and code versions before attempting comparisons.

## Environment note

The launcher history points to a Conda environment named `graphgpu`. The currently available environment was recorded on 2026-11-19. It is a recovered baseline, not conclusive proof of the historical environment. The local ESM source reports 2.0.1, whereas the installed `fair-esm` package reports 2.0.0; this implementation choice needs author confirmation. See [`ENVIRONMENT_RECOVERY.md`](ENVIRONMENT_RECOVERY.md).
