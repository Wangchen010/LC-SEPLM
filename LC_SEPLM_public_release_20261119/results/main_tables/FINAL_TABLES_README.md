# Final-table package (2026-07-21)

- **Our model:** ESM2 t30 (150M) backbone with Tail8 LoRA (layers 22–29, q/v projections), trained for 500k steps with MLM weight 0.6; the main candidate column is epoch 10 unless explicitly marked as a gate row.
- The eight protein-level rows use the repaired no-exact-sequence-overlap task package. ESM2/ESM-S values come from the archived five-seed comparison; Tail8+MLM0.6 epoch10 values are aggregated from the per-seed outputs where available.
- The GO row for Tail8+MLM0.6 is a three-seed validation gate (seeds 7, 17, 23), not a five-seed final test.
- ProteinGym in the main table is the 93-assay development gate. The ESM-S cell is explicitly marked as a legacy 184-assay value and must not be used for a direct three-model comparison.
- PTM averages are means over three site types per split seed, with five split seeds (7, 11, 13, 17, 19).
- ClinVar is excluded because the archived 1,779-assay output is identically zero for every model. Metal ion binding is diagnostic only because the archived binary metrics are near zero for every model and require an evaluator/data audit.
- `epoch10_11_14_gate_comparison.*` is the correct source for epoch checkpoint discussion; its EC/GO rows use three gate seeds and its ProteinGym rows use 93 assays.
