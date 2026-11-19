# Data and software notices

This release does not include raw third-party data or model weights. Before using any external resource, consult its provider for current access conditions, license, citation, and redistribution rules.

## External resources referenced by the workflows

The manuscript and scripts refer to resources including AlphaFold DB/Swiss-Prot structures, UniProt annotations, RCSB PDB structures, ProteinGym assays, Oxer11/ESM-S benchmark resources, and pretrained ESM2/ESM-S models. This repository does not redistribute these source databases or model weights. Dataset versions, exact processed inputs, and access conditions may differ across workflows.

For a full reproduction, authors and users should document the source URL or accession, retrieval date/version, applicable terms, preprocessing steps, and checksums for each required external file. Do not redistribute source or derived files unless the applicable provider terms allow it.

## Software dependencies and attribution

The project uses PyTorch, fair-esm/ESM, NumPy, pandas, scikit-learn and other scientific Python packages. Individual scripts may need Biopython, requests, Matplotlib, Seaborn, Hydra/OmegaConf, einops, MMseqs2, or other tools. Consult the package and tool providers for their current license and citation requirements.

Three helper modules from the local project are included because candidate scripts import them. Their authorship and license compatibility must be confirmed by the project authors. No LC-SEPLM license is currently assigned.

## Manuscript availability statements

The manuscript's code/data availability wording must match the final repository contents and persistent identifier. Do not claim that all code, data, weights, or exact splits are available unless they have actually been deposited and the access terms have been checked.
