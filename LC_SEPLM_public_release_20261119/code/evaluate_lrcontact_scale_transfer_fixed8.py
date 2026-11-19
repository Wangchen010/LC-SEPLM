#!/usr/bin/env python3
"""Evaluate scale-transfer LRContact and MLM-only checkpoints on fixed 8 tasks."""
from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path


TASKS = {
    "ec_level1": ("EC", "classification", "01_ec_level1.csv", "macro_f1_mean", None),
    "subcellular_location": ("Subcellular", "classification", "02_subcellular_location.csv", "macro_f1_mean", None),
    "membrane_binary": ("Membrane", "classification", "03_membrane_binary.csv", "macro_f1_mean", None),
    "strict_membrane_binary_seed23": (
        "Strict membrane",
        "classification",
        "04_strict_membrane_binary_seed23.csv",
        "macro_f1_mean",
        None,
    ),
    "go_molecular_function": ("GO MF", "multilabel", "05_go_molecular_function.csv", "micro_f1_mean", ";"),
    "pfam_family": ("Pfam family", "classification", "06_pfam_family.csv", "macro_f1_mean", None),
    "pfam_clan_remote_homology": (
        "Remote homology",
        "classification",
        "07_pfam_clan_remote_homology.csv",
        "macro_f1_mean",
        None,
    ),
    "hou_subcell_level3": ("Hou subcell L3", "multilabel", "08_hou_subcell_level3.csv", "micro_f1_mean", ";"),
}


def read_score(path: Path, feature: str, metric: str) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["feature_set"] == feature:
                return float(row[metric])
    raise KeyError(f"{feature} not found in {path}")


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def parse_extra_full_state_models(values: list[str]) -> dict[str, str]:
    models: dict[str, str] = {}
    for value in values:
        if "=" not in value or "," not in value.split("=", 1)[1]:
            raise ValueError(
                f"Invalid --extra-full-state-model {value!r}; expected LABEL=MODEL_NAME,CHECKPOINT"
            )
        label, payload = value.split("=", 1)
        models[label.strip()] = payload.strip()
    return models


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    eval_root = out_dir / "evaluations"
    cache_root = args.feature_cache_root or (out_dir / "feature_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed) for seed in args.eval_seeds.split(",") if seed.strip()]
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")

    pretrained_name = f"ESM2_pretrained_{args.backbone_tag}"
    mlm_name = args.mlm_feature_name or f"MLMOnly_{args.backbone_tag}_full500k_epoch10"
    lrcontact_name = (
        args.lrcontact_feature_name
        or f"LRContact_{args.backbone_tag}_tail8_mlm0p6_full500k_epoch10"
    )
    extra_full_state_models = parse_extra_full_state_models(args.extra_full_state_model)

    for task_name in tasks:
        _, task_type, csv_name, _, delimiter = TASKS[task_name]
        task_csv = dataset_dir / csv_name
        if not task_csv.is_file():
            raise FileNotFoundError(task_csv)
        for seed in seeds:
            task_out = eval_root / f"{task_name}_seed{seed}"
            summary = task_out / "protein_task_summary.csv"
            if summary.is_file() and not args.force:
                print(f"Skip existing {summary}", flush=True)
                continue
            cmd = [
                sys.executable,
                str(root / "code/evaluate_protein_task_embeddings.py"),
                "--task-csv",
                str(task_csv),
                "--out-dir",
                str(task_out),
                "--task-type",
                task_type,
                "--sequence-col",
                "sequence",
                "--label-col",
                "label",
                "--seed",
                str(seed),
                "--n-splits",
                "5",
                "--eval-split-mode",
                "test",
                "--device",
                args.device,
                "--classifier",
                "sgd_log_loss",
                "--classifier-max-iter",
                str(args.classifier_max_iter),
                "--skip-default-baselines",
                "--max-len",
                "256",
                "--batch-size",
                "256",
                "--esm2-max-residues",
                "512",
                "--esm2-batch-size",
                str(args.esm2_batch_size),
                "--esm2-amp",
                "--feature-cache-dir",
                str(cache_root / task_name),
                "--esm2-pretrained-model",
                f"{pretrained_name}={args.model_name}",
                "--esm2-lora-model",
                f"{mlm_name}={args.mlm_checkpoint}",
                "--esm2-lora-model",
                f"{lrcontact_name}={args.lrcontact_checkpoint}",
                "--include-feature-set",
                pretrained_name,
                "--include-feature-set",
                mlm_name,
                "--include-feature-set",
                lrcontact_name,
            ]
            for feature_name, model_spec in extra_full_state_models.items():
                cmd.extend(["--esm2-full-state-model", f"{feature_name}={model_spec}"])
                cmd.extend(["--include-feature-set", feature_name])
            if delimiter is not None:
                cmd.extend(["--label-delimiter", delimiter])
            print("Run " + " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)

    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for task_name in tasks:
        label, _, _, metric, _ = TASKS[task_name]
        pretrained_values: list[float] = []
        mlm_values: list[float] = []
        lrcontact_values: list[float] = []
        extra_values: dict[str, list[float]] = {name: [] for name in extra_full_state_models}
        for seed in seeds:
            summary = eval_root / f"{task_name}_seed{seed}" / "protein_task_summary.csv"
            pretrained = read_score(summary, pretrained_name, metric)
            mlm = read_score(summary, mlm_name, metric)
            lrcontact = read_score(summary, lrcontact_name, metric)
            pretrained_values.append(pretrained)
            mlm_values.append(mlm)
            lrcontact_values.append(lrcontact)
            raw_row = {
                "backbone": args.backbone_tag,
                "task": task_name,
                "task_label": label,
                "metric": metric,
                "eval_seed": seed,
                "pretrained": pretrained,
                "mlm_only": mlm,
                "lrcontact": lrcontact,
                "lrcontact_delta_vs_pretrained": lrcontact - pretrained,
                "lrcontact_delta_vs_mlm_only": lrcontact - mlm,
            }
            for feature_name in extra_full_state_models:
                value = read_score(summary, feature_name, metric)
                extra_values[feature_name].append(value)
                raw_row[feature_name] = value
                raw_row[f"lrcontact_delta_vs_{feature_name}"] = lrcontact - value
            raw_rows.append(raw_row)
        pretrained_mean, pretrained_std = mean_std(pretrained_values)
        mlm_mean, mlm_std = mean_std(mlm_values)
        lrcontact_mean, lrcontact_std = mean_std(lrcontact_values)
        delta_pre = [lr - base for lr, base in zip(lrcontact_values, pretrained_values)]
        delta_mlm = [lr - mlm for lr, mlm in zip(lrcontact_values, mlm_values)]
        delta_pre_mean, delta_pre_std = mean_std(delta_pre)
        delta_mlm_mean, delta_mlm_std = mean_std(delta_mlm)
        summary_row = {
            "backbone": args.backbone_tag,
            "task": task_name,
            "task_label": label,
            "metric": metric,
            "eval_seeds": args.eval_seeds,
            "pretrained_mean": pretrained_mean,
            "pretrained_std": pretrained_std,
            "mlm_only_mean": mlm_mean,
            "mlm_only_std": mlm_std,
            "lrcontact_mean": lrcontact_mean,
            "lrcontact_std": lrcontact_std,
            "lrcontact_delta_vs_pretrained_mean": delta_pre_mean,
            "lrcontact_delta_vs_pretrained_std": delta_pre_std,
            "lrcontact_delta_vs_mlm_only_mean": delta_mlm_mean,
            "lrcontact_delta_vs_mlm_only_std": delta_mlm_std,
        }
        for feature_name, values in extra_values.items():
            value_mean, value_std = mean_std(values)
            deltas = [lr - value for lr, value in zip(lrcontact_values, values)]
            delta_mean, delta_std = mean_std(deltas)
            summary_row[f"{feature_name}_mean"] = value_mean
            summary_row[f"{feature_name}_std"] = value_std
            summary_row[f"lrcontact_delta_vs_{feature_name}_mean"] = delta_mean
            summary_row[f"lrcontact_delta_vs_{feature_name}_std"] = delta_std
        summary_rows.append(summary_row)

    for name, rows in (("raw", raw_rows), ("summary", summary_rows)):
        path = out_dir / f"{args.backbone_tag}_fixed8_{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    win_pre = sum(float(row["lrcontact_delta_vs_pretrained_mean"]) > 0 for row in summary_rows)
    win_mlm = sum(float(row["lrcontact_delta_vs_mlm_only_mean"]) > 0 for row in summary_rows)
    mean_pre = statistics.mean(float(row["lrcontact_delta_vs_pretrained_mean"]) for row in summary_rows)
    mean_mlm = statistics.mean(float(row["lrcontact_delta_vs_mlm_only_mean"]) for row in summary_rows)
    lines = [
        f"# LRContact Scale Transfer: {args.backbone_tag}",
        "",
        f"- Model: `{args.model_name}`",
        f"- Eval seeds: `{args.eval_seeds}`",
        f"- LRContact wins vs pretrained: `{win_pre}/{len(summary_rows)}`; mean delta `{mean_pre:+.6f}`",
        f"- LRContact wins vs MLM-only: `{win_mlm}/{len(summary_rows)}`; mean delta `{mean_mlm:+.6f}`",
        "",
        "| Task | Metric | Pretrained | MLM-only | LRContact | Delta vs pretrained | Delta vs MLM-only |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    if extra_full_state_models:
        extra_headers = "".join(f" {name} | Delta vs {name} |" for name in extra_full_state_models)
        lines[-2] += extra_headers
        lines[-1] += "".join("---:|---:|" for _ in extra_full_state_models)
    for row in summary_rows:
        line = (
            f"| {row['task_label']} | `{row['metric']}` | {row['pretrained_mean']:.6f} | "
            f"{row['mlm_only_mean']:.6f} | {row['lrcontact_mean']:.6f} | "
            f"{row['lrcontact_delta_vs_pretrained_mean']:+.6f} | "
            f"{row['lrcontact_delta_vs_mlm_only_mean']:+.6f} |"
        )
        for feature_name in extra_full_state_models:
            line += (
                f" {row[f'{feature_name}_mean']:.6f} | "
                f"{row[f'lrcontact_delta_vs_{feature_name}_mean']:+.6f} |"
            )
        lines.append(line)
    md_path = out_dir / f"{args.backbone_tag}_fixed8_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Project root containing model and evaluation outputs.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backbone-tag", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mlm-checkpoint", type=Path, required=True)
    parser.add_argument("--lrcontact-checkpoint", type=Path, required=True)
    parser.add_argument("--mlm-feature-name")
    parser.add_argument("--lrcontact-feature-name")
    parser.add_argument(
        "--extra-full-state-model",
        action="append",
        default=[],
        metavar="LABEL=MODEL_NAME,CHECKPOINT",
        help="Add a fair-esm-compatible full-state baseline to every task.",
    )
    parser.add_argument(
        "--feature-cache-root",
        type=Path,
        help="Reuse an existing feature cache while writing evaluations elsewhere.",
    )
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--eval-seeds", default="7,17,23")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--esm2-batch-size", type=int, default=16)
    parser.add_argument("--classifier-max-iter", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
