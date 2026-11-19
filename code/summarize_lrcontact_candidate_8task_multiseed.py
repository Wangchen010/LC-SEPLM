#!/usr/bin/env python3
"""Summarize one candidate across the eight protein-level downstream tasks."""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


TASKS = {
    "ec_level1": ("EC", "macro_f1_mean"),
    "subcellular_location": ("Subcellular", "macro_f1_mean"),
    "membrane_binary": ("Membrane", "macro_f1_mean"),
    "strict_membrane_binary_seed23": ("Strict membrane", "macro_f1_mean"),
    "go_molecular_function": ("GO MF", "micro_f1_mean"),
    "pfam_family": ("Pfam family", "macro_f1_mean"),
    "pfam_clan_remote_homology": ("Remote homology", "macro_f1_mean"),
    "hou_subcell_level3": ("Hou subcell L3", "micro_f1_mean"),
}


def read_score(path: Path, feature: str, metric: str) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["feature_set"] == feature:
                return float(row[metric])
    raise KeyError(f"{feature} not found in {path}")


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    eval_root = Path(args.eval_root) if args.eval_root is not None else (
        root / "sequence_graph_pretraining/formal_downstream_tasks/evaluations"
    )
    branch = root / "structure_topology_branch"
    seeds = [int(seed) for seed in args.eval_seeds.split(",") if seed.strip()]
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    rows: list[dict[str, object]] = []
    raw: list[dict[str, object]] = []
    for task in tasks:
        label, metric = TASKS[task]
        base_values: list[float] = []
        model_values: list[float] = []
        for seed in seeds:
            path = eval_root / f"{task}_{args.tag}_evalseed{seed}/protein_task_summary.csv"
            base = read_score(path, args.esm2_feature_name, metric)
            model = read_score(path, args.lora_feature_name, metric)
            base_values.append(base)
            model_values.append(model)
            raw.append(
                {
                    "task": task,
                    "task_label": label,
                    "metric": metric,
                    "eval_seed": seed,
                    "esm2": base,
                    "lr_contact_xattn_af2": model,
                    "delta_vs_esm2": model - base,
                }
            )
        deltas = [model - base for model, base in zip(model_values, base_values)]
        base_mean, base_std = mean_std(base_values)
        model_mean, model_std = mean_std(model_values)
        delta_mean, delta_std = mean_std(deltas)
        rows.append(
            {
                "task": task,
                "task_label": label,
                "metric": metric,
                "eval_seeds": ",".join(map(str, seeds)),
                "esm2_mean": base_mean,
                "esm2_std": base_std,
                "lr_contact_xattn_af2_mean": model_mean,
                "lr_contact_xattn_af2_std": model_std,
                "delta_vs_esm2_mean": delta_mean,
                "delta_vs_esm2_std": delta_std,
            }
        )

    raw_path = branch / f"{args.tag}_full8_multiseed_raw.csv"
    summary_path = branch / f"{args.tag}_full8_multiseed_summary.csv"
    md_path = branch / f"{args.tag}_full8_multiseed_summary.md"
    for path, data in ((raw_path, raw), (summary_path, rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    positive = sum(float(row["delta_vs_esm2_mean"]) > 0 for row in rows)
    lines = [
        "# LRContact candidate: eight protein tasks",
        "",
        f"- Tag: `{args.tag}`",
        f"- Positive tasks: `{positive}/{len(rows)}`",
        "",
        "| Task | Metric | ESM2 | Candidate | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task_label']} | `{row['metric']}` | {row['esm2_mean']:.6f} | "
            f"{row['lr_contact_xattn_af2_mean']:.6f} | {row['delta_vs_esm2_mean']:+.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", required=True, help="Project root containing evaluation results.")
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--eval-seeds", default="7,17,23,42,101")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--esm2-feature-name", default="ESM2_pretrained_esm2_t30_150M")
    parser.add_argument("--lora-feature-name", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
