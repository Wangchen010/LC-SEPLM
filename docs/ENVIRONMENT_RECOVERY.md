# Environment recovery note

The manuscript's historical parallel-evaluation launcher explicitly pointed to the Conda environment named `graphgpu`. That environment is still present. We queried it on 2026-11-19 and observed Python 3.10.20, PyTorch 2.5.1+cu121 (CUDA build 12.1), fair-esm 2.0.0, NumPy 1.26.3, pandas 2.3.3, scikit-learn 1.7.2, SciPy 1.15.3, Matplotlib 3.10.9, Seaborn 0.13.2, Biopython 1.87, requests 2.34.2, tqdm 4.67.3, einops 0.8.0, hydra-core 1.3.2 and omegaconf 2.3.0. The observed package snapshot is in `environment.graphgpu.observed.yml`.

This is useful evidence, but it does not prove that the environment had the same package versions at every historical training and evaluation date. The current shell also cannot communicate with the NVIDIA driver, so we cannot verify GPU execution here. Treat these versions as a recovered baseline; the authors should confirm them against archived logs or the training host if available. Do not describe this snapshot as a fully reproduced historical environment without that confirmation.

The source workspace itself reports fair-esm version 2.0.1 in `esm/version.py`, while the current `graphgpu` environment imports fair-esm 2.0.0 from site-packages. The original training code prepends the workspace to Python's import path, so whether the run used workspace source 2.0.1 or installed fair-esm 2.0.0 needs confirmation from the original working directory and launch command. The user's local ESM tree also includes modifications beyond the package version; record its git commit and any local diffs for the exact training implementation before calling the environment locked.

Every Python file in `code/` was invoked with `--help` under `graphgpu` on 2026-11-19 without reported failure. Python syntax compilation succeeded. These checks only cover imports and argument parsing; they do not reproduce training or benchmark metrics.

