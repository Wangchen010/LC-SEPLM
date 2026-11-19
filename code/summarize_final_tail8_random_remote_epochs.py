#!/usr/bin/env python3
"""Summarize random/fixed-split Remote-homology results for Tail-8 epochs 1-14."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASELINE = "ESM2_pretrained_esm2_t30_150M"
SEEDS = (7, 17, 23, 42, 101)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []
    for epoch in range(1, 15):
        tag = f"final_tail8_mlm0p6_random_remote_epoch{epoch}"
        candidate = f"LC_SEPLM_Final_Tail8_MLM0p6_Epoch{epoch}"
        for seed in SEEDS:
            path = (
                args.eval_root
                / f"pfam_clan_remote_homology_{tag}_evalseed{seed}"
                / "protein_task_summary.csv"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            data = pd.read_csv(path).set_index("feature_set")
            if BASELINE not in data.index or candidate not in data.index:
                raise KeyError(f"Missing feature set in {path}")
            baseline = float(data.loc[BASELINE, "macro_f1_mean"])
            score = float(data.loc[candidate, "macro_f1_mean"])
            raw_rows.append(
                {
                    "epoch": epoch,
                    "classifier_seed": seed,
                    "ESM2_macro_F1": baseline,
                    "LC_SEPLM_macro_F1": score,
                    "delta": score - baseline,
                    "source": str(path),
                }
            )

    raw = pd.DataFrame(raw_rows)
    summary_rows: list[dict[str, object]] = []
    for epoch, group in raw.groupby("epoch", sort=True):
        baseline = float(group["ESM2_macro_F1"].mean())
        candidate = float(group["LC_SEPLM_macro_F1"].mean())
        delta = candidate - baseline
        summary_rows.append(
            {
                "epoch": int(epoch),
                "n_classifier_seeds": int(group.shape[0]),
                "ESM2_macro_F1_mean": baseline,
                "ESM2_macro_F1_std": float(group["ESM2_macro_F1"].std(ddof=1)),
                "LC_SEPLM_macro_F1_mean": candidate,
                "LC_SEPLM_macro_F1_std": float(group["LC_SEPLM_macro_F1"].std(ddof=1)),
                "absolute_delta": delta,
                "percentage_point_gain": 100.0 * delta,
                "relative_gain_percent": 100.0 * delta / baseline,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("epoch")
    best = summary.loc[summary["LC_SEPLM_macro_F1_mean"].idxmax()].to_dict()

    raw.to_csv(args.out_dir / "remote_random_split_epoch1_14_raw.csv", index=False)
    summary.to_csv(args.out_dir / "remote_random_split_epoch1_14_summary.csv", index=False)
    (args.out_dir / "remote_random_split_epoch1_14_best.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False) + "\n"
    )
    md = [
        "# Final Tail-8 + MLM 0.6: random/fixed-split Remote homology, epochs 1-14",
        "",
        "Protocol: original fixed split, legacy `val_test` evaluation, classifier seeds 7/17/23/42/101.",
        "The maximum below is a best-observed test result; it must not be described as validation-selected.",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Best observed epoch",
        "",
        f"- Epoch: {int(best['epoch'])}",
        f"- ESM2 macro-F1: {best['ESM2_macro_F1_mean']:.6f}",
        f"- LC-SEPLM macro-F1: {best['LC_SEPLM_macro_F1_mean']:.6f}",
        f"- Absolute delta: {best['absolute_delta']:.6f}",
        f"- Percentage-point gain: {best['percentage_point_gain']:.3f}",
        f"- Relative gain: {best['relative_gain_percent']:.2f}%",
        "",
    ]
    (args.out_dir / "remote_random_split_epoch1_14_summary.md").write_text("\n".join(md))
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
