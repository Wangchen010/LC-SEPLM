# External low-homology PDB contact benchmark

## Purpose of this document

This file is an insertion-ready supplement for the LC-SEPLM manuscript. It records the external low-homology PDB long-range-contact experiment, the numbers currently available in the project record, manuscript-ready English text, a Chinese explanation, and the limits that should remain explicit. It does not modify the original manuscript.

## One-sentence result

On an external, low-homology RCSB PDB benchmark, a frozen LC-SEPLM representation produced higher long-range-contact ranking and pair-classification scores than frozen ESM2 and the matched MLM-only control when the probe received sequence-derived representations only.

## What task was evaluated?

This is a residue-pair contact task, not a protein-level EC/GO classification task. For a protein sequence, the probe ranks candidate residue pairs according to their predicted probability of forming a spatial contact after folding. A pair is a positive long-range contact when

\[
|i-j|\geq 12 \quad\text{and}\quad d_{C_\alpha}(i,j)\leq 8\,\text{\AA}.
\]

The sequence is the only model input. PDB coordinates are used only after sequence encoding to construct reference labels and calculate the evaluation metrics.

## Dataset construction and evaluation protocol

- **External source:** RCSB Protein Data Bank chains, independent of the AlphaFold-Swiss-Prot adaptation set.
- **Length filter:** 64–512 residues.
- **Coordinate filter:** at least 90% coordinate coverage.
- **Homology filter:** remove candidates with a qualifying MMseqs2 hit against the 500,000-protein adaptation set at ≥30% sequence identity and ≥80% query coverage.
- **Final size:** 187 PDB chains after length, coordinate-coverage and homology filtering.
- **Contact label:** sequence separation \(|i-j|\geq12\) and Cα–Cα distance ≤8 Å.
- **Encoders:** ESM2-150M, MLM-only 500k Tail-8 control, and LC-SEPLM epoch 10.
- **Inference:** encoders frozen; amino-acid sequence only; no PDB coordinates, contact matrix or contact labels supplied to the encoder.
- **Pair probe:** pair features formed from residue embeddings using the element-wise absolute difference and element-wise product; a standardized class-balanced logistic SGD probe was fitted using 500 proteins and at most 50,000 training pairs.
- **Probe seeds:** 7, 17 and 23.
- **Metrics:** P@L, P@L/2, AUPRC and AUROC. Here, P@L selects the top \(L\) predicted pairs for a protein of length \(L\); P@L/2 uses the top \(L/2\) pairs.

## Current numerical results

The project record reports the following means across the three probe seeds. Standard deviations are currently recorded only for P@L; do not add standard deviations for the other metrics until the seed-level raw table is recovered or recomputed.

| Metric | ESM2 | MLM-only 500k | LC-SEPLM epoch 10 | LC-SEPLM − ESM2 | LC-SEPLM − MLM-only |
|---|---:|---:|---:|---:|---:|
| P@L | 0.0655 ± 0.0036 | 0.0696 ± 0.0020 | **0.0814 ± 0.0057** | **+0.0159** | **+0.0118** |
| P@L/2 | 0.0766 | 0.0809 | **0.0934** | **+0.0168** | **+0.0125** |
| AUPRC | 0.0358 | 0.0368 | **0.0446** | **+0.0088** | **+0.0078** |
| AUROC | 0.6717 | 0.6741 | **0.7091** | **+0.0374** | **+0.0350** |

### Interpretation supported by these numbers

LC-SEPLM ranked true long-range contacts above non-contacts more effectively than both ESM2 and MLM-only on this external benchmark. The ordering was consistent across all four reported metrics: LC-SEPLM > MLM-only and LC-SEPLM > ESM2. Because the benchmark is external and filtered below 30% sequence identity to the adaptation set, the result is evidence that contact-related information in the LC-SEPLM representation can transfer beyond the AlphaFold-Swiss-Prot adaptation distribution.

### Claim boundary

This experiment supports transfer of contact-related representation information. It does not establish universal improvement on functional downstream tasks, does not show that the contact head itself generalizes end-to-end, and does not make SaProt or ESM-S a matched comparator in this experiment. The benchmark contains 187 chains and the current record does not include paired bootstrap confidence intervals. The results should therefore be described as supportive external generalization evidence, not as a universal superiority claim.

## Manuscript-ready Results paragraph (English)

### External low-homology PDB contact generalization

To test whether contact-related information transferred beyond the AlphaFold-Swiss-Prot adaptation distribution, we constructed an external benchmark from RCSB Protein Data Bank chains. We retained chains of 64–512 residues with at least 90% coordinate coverage and removed candidates with MMseqs2 hits to the 500,000-protein adaptation set at ≥30% sequence identity and ≥80% query coverage, yielding 187 chains. Contacts were defined by a Cα–Cα distance of at most 8 Å and a sequence separation of at least 12 residues. The ESM2, MLM-only and LC-SEPLM encoders were frozen, and the probe received amino-acid sequences only; PDB coordinates were used only to derive reference contact labels. Pair features were constructed from residue embeddings using the element-wise absolute difference and product, and a standardized class-balanced logistic probe was fitted with probe seeds 7, 17 and 23. LC-SEPLM achieved P@L of 0.0814 ± 0.0057, compared with 0.0696 ± 0.0020 for MLM-only and 0.0655 ± 0.0036 for ESM2. The same ordering was observed for P@L/2 (0.0934, 0.0809 and 0.0766), AUPRC (0.0446, 0.0368 and 0.0358) and AUROC (0.7091, 0.6741 and 0.6717), respectively. These results support the transferability of contact-related information in LC-SEPLM representations to low-homology proteins from an external structure resource, while not implying universal improvement across functional or residue-level tasks.

## Manuscript-ready Methods paragraph (English)

### External PDB contact benchmark and frozen probe

RCSB PDB chains were converted into amino-acid sequences and Cα coordinate arrays. We retained chains of 64–512 residues with at least 90% coordinate coverage. Each candidate sequence was searched against the 500,000 adaptation sequences using MMseqs2, and candidates with a hit at ≥30% sequence identity and ≥80% query coverage were removed. For each remaining chain, a residue pair was labeled as a long-range contact when \(|i-j|\geq12\) and the Cα–Cα distance was ≤8 Å. ESM2, MLM-only and LC-SEPLM encoders were kept frozen during evaluation and received amino-acid sequences only. Pair features were formed from the element-wise absolute difference and element-wise product of the two residue embeddings. A standardized, class-balanced logistic SGD probe was fitted using 500 proteins and at most 50,000 training pairs, and evaluated with probe seeds 7, 17 and 23. P@L, P@L/2, AUPRC and AUROC were calculated from the ranked pair probabilities. No PDB coordinate or contact label was provided to the encoder during inference.

## Manuscript-ready Figure 9 caption (English)

**Figure 9 | External low-homology PDB contact generalization.** Frozen-probe performance of ESM2, MLM-only 500k and LC-SEPLM epoch 10 on 187 low-homology RCSB PDB chains. Bars show means across probe seeds 7, 17 and 23; error bars should be shown only for metrics for which seed-level standard deviations are available. The model received amino-acid sequences only, whereas PDB coordinates were used only to derive reference contacts. Contacts were defined by a Cα distance threshold of 8 Å and a sequence-separation threshold of 12 residues.

## 可直接理解的中文说明

这不是 EC、GO 或膜蛋白分类，而是一个残基对接触预测任务。给模型一条蛋白质序列，模型需要判断序列上相距较远的两个残基，折叠后是否会在三维空间中靠近。测试集来自 RCSB PDB，而不是训练 LC-SEPLM 时使用的 AlphaFold-Swiss-Prot 结构。经过长度、坐标覆盖率和同源性过滤后，保留 187 条蛋白质链；与 500,000 条适应序列的序列一致性达到 30% 或以上、且 query coverage 达到 80% 或以上的候选链被排除。

接触标签由 PDB 坐标生成：两个残基在序列上至少相隔 12 个位置，并且 Cα–Cα 距离不超过 8 Å，就标记为正接触。模型推理时不输入 PDB 坐标，只输入氨基酸序列。ESM2、MLM-only 和 LC-SEPLM 的编码器都被冻结，再从残基 embedding 构造残基对特征，并训练一个简单的类别平衡逻辑回归 probe。使用 7、17、23 三个 probe seed，报告 P@L、P@L/2、AUPRC 和 AUROC。

结果为：LC-SEPLM 的 P@L 为 0.0814 ± 0.0057，高于 MLM-only 的 0.0696 ± 0.0020 和 ESM2 的 0.0655 ± 0.0036。P@L/2、AUPRC 和 AUROC 也都是 LC-SEPLM 最高。这个结果说明 LC-SEPLM 表征中的接触相关信息可以迁移到低同源的外部 PDB 蛋白质，但不能据此声称模型在所有功能任务上都更好。

## Source and audit notes

- Main numerical source currently available: `lc_seplm_revision_v2/manuscript_additions_bilingual.md`, Results paragraph “external low-homology PDB generalization”.
- Figure assets: `lc_seplm_revision_v2/figures/Figure_9_external_PDB_generalization.*`.
- The current project note gives standard deviations for P@L only. The other three metrics are reported as three-seed means without seed-level standard deviations in the available record.
- Before final submission, recover or recompute the per-seed raw table and add 95% confidence intervals if possible.
- This document is supplementary insertion material; the original manuscript remains unchanged.
