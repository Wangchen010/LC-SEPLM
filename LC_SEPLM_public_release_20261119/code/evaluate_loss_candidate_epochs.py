#!/usr/bin/env python3
"""Evaluate every loss-search epoch on EC, GO-MF, and ProteinGym development sets.

The three tasks are treated as separate validation objectives. Raw F1 and
Spearman deltas are not averaged directly; checkpoints are selected by positive
task count and per-task percentile ranks. The selected epoch is copied to
``best_downstream_model.pt`` while all ``epoch_N.pt`` checkpoints are retained.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--base-tag", required=True)
    parser.add_argument("--feature-prefix", required=True)
    parser.add_argument("--epochs", default="1,2")
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--proteingym-assay-id-file", type=Path, required=True)
    parser.add_argument("--eval-seeds", default="7,17,23")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-tag-prefix", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND\n" + " ".join(command) + "\n\n")
        handle.flush()
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


def read_feature_row(path: Path, feature: str) -> pd.Series:
    frame = pd.read_csv(path)
    selected = frame[frame["feature_set"] == feature]
    if len(selected) != 1:
        raise ValueError(f"Expected one {feature!r} row in {path}; got {len(selected)}")
    return selected.iloc[0]


def training_metrics(checkpoint: Path) -> dict[str, object]:
    payload = torch.load(checkpoint, map_location="cpu")
    metrics = payload.get("val_metrics", {})
    return {
        "training_epoch": int(payload.get("epoch", 0)),
        "training_selection_score": float(payload.get("val_selection_score", float("nan"))),
        "val_total_loss": float(metrics.get("total_loss", float("nan"))),
        "val_contact_loss": float(metrics.get("contact_loss", float("nan"))),
        "val_mlm_loss": float(metrics.get("mlm_loss", float("nan"))),
        "val_contact_balanced_accuracy": float(metrics.get("contact_balanced_accuracy", float("nan"))),
        "val_mlm_accuracy": float(metrics.get("mlm_accuracy", float("nan"))),
    }


def evaluate_epoch(args: argparse.Namespace, epoch: int) -> dict[str, object]:
    checkpoint = args.checkpoint_dir / f"epoch_{epoch}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    epoch_tag = f"{args.base_tag}_epoch{epoch}_gate"
    feature = f"{args.feature_prefix}_Epoch{epoch}"
    log_path = args.result_root / "epoch_gate_logs" / f"{args.candidate}_epoch{epoch}.log"
    protein_summary = args.root / "structure_topology_branch" / f"{epoch_tag}_full8_multiseed_summary.csv"

    if args.force or not protein_summary.is_file():
        command = [
            sys.executable,
            str(args.root / "code/run_lrcontact_candidate_8task_multiseed.py"),
            "--root",
            str(args.root),
            "--checkpoint",
            str(checkpoint),
            "--tag",
            epoch_tag,
            "--lora-feature-name",
            feature,
            "--tasks",
            "ec_level1,go_molecular_function",
            "--eval-seeds",
            args.eval_seeds,
            "--eval-split-mode",
            "val",
            "--task-root",
            str(args.task_root),
            "--eval-root",
            str(args.eval_root),
            "--device",
            args.device,
            "--cache-tag",
            f"{args.cache_tag_prefix}_epoch{epoch}",
        ]
        run_logged(command, log_path)

    pg_out = args.result_root / "proteingym_epoch_gate" / args.candidate / f"epoch_{epoch}"
    pg_summary = pg_out / "proteingym_wt_marginals_summary.csv"
    if args.force or not pg_summary.is_file():
        command = [
            sys.executable,
            str(args.root / "code/evaluate_proteingym_wt_marginals.py"),
            "--assay-id-file",
            str(args.proteingym_assay_id_file),
            "--out-dir",
            str(pg_out),
            "--device",
            args.device,
            "--amp",
            "--model-spec",
            "ESM2_pretrained=pretrained:esm2_t30_150M_UR50D",
            "--model-spec",
            f"{feature}=lora:{checkpoint}",
        ]
        run_logged(command, log_path)

    protein = pd.read_csv(protein_summary).set_index("task")
    for task in ("ec_level1", "go_molecular_function"):
        if task not in protein.index:
            raise ValueError(f"{task} absent from {protein_summary}")
    pg = pd.read_csv(pg_summary)
    pg_model = pg[pg["model"] == feature]
    pg_base = pg[pg["model"] == "ESM2_pretrained"]
    if len(pg_model) != 1 or len(pg_base) != 1:
        raise ValueError(f"Missing model/baseline row in {pg_summary}")
    ec = protein.loc["ec_level1"]
    go = protein.loc["go_molecular_function"]
    pg_model_row = pg_model.iloc[0]
    pg_base_row = pg_base.iloc[0]
    return {
        "candidate": args.candidate,
        "epoch": epoch,
        "checkpoint": str(checkpoint),
        "ec_metric": "macro_f1",
        "ec_baseline": float(ec["esm2_mean"]),
        "ec_score": float(ec["lr_contact_xattn_af2_mean"]),
        "ec_delta": float(ec["delta_vs_esm2_mean"]),
        "go_metric": "micro_f1",
        "go_baseline": float(go["esm2_mean"]),
        "go_score": float(go["lr_contact_xattn_af2_mean"]),
        "go_delta": float(go["delta_vs_esm2_mean"]),
        "proteingym_metric": "mean_assay_spearman",
        "proteingym_assays": int(pg_model_row["assays"]),
        "proteingym_baseline": float(pg_base_row["mean_spearman"]),
        "proteingym_score": float(pg_model_row["mean_spearman"]),
        "proteingym_delta": float(pg_model_row["mean_delta"]),
        "protein_summary": str(protein_summary),
        "proteingym_summary": str(pg_summary),
        "epoch_log": str(log_path),
        **training_metrics(checkpoint),
    }


def rank_and_select(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    delta_columns = ["ec_delta", "go_delta", "proteingym_delta"]
    frame = frame.copy()
    frame["positive_downstream_tasks"] = (frame[delta_columns] > 0).sum(axis=1)
    rank_columns: list[str] = []
    for column in delta_columns:
        rank_column = column.replace("_delta", "_rank_percentile")
        frame[rank_column] = frame[column].rank(method="average", pct=True, ascending=True)
        rank_columns.append(rank_column)
    frame["worst_task_rank_percentile"] = frame[rank_columns].min(axis=1)
    frame["mean_task_rank_percentile"] = frame[rank_columns].mean(axis=1)
    frame["pareto_dominated"] = False
    for index, row in frame.iterrows():
        for other_index, other in frame.iterrows():
            if other_index == index:
                continue
            if bool((other[delta_columns] >= row[delta_columns]).all() and (other[delta_columns] > row[delta_columns]).any()):
                frame.loc[index, "pareto_dominated"] = True
                break
    ranked = frame.sort_values(
        [
            "positive_downstream_tasks",
            "pareto_dominated",
            "worst_task_rank_percentile",
            "mean_task_rank_percentile",
            "epoch",
        ],
        ascending=[False, True, False, False, True],
    ).reset_index(drop=True)
    ranked["selected_best_downstream"] = False
    ranked.loc[0, "selected_best_downstream"] = True
    return ranked, ranked.iloc[0]


def write_report(frame: pd.DataFrame, selected: pd.Series, path: Path) -> None:
    columns = [
        "epoch",
        "ec_score",
        "ec_delta",
        "go_score",
        "go_delta",
        "proteingym_score",
        "proteingym_delta",
        "positive_downstream_tasks",
        "pareto_dominated",
        "selected_best_downstream",
    ]
    lines = [
        f"# Epoch downstream gate: {selected['candidate']}",
        "",
        "Each epoch is evaluated on no-exact-leak EC validation, GO-MF validation, and the fixed "
        "ProteinGym development assay set. Selection uses positive-task count and per-task ranks; "
        "raw F1 and Spearman deltas are not directly averaged.",
        "",
        frame[columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Selected epoch: **{int(selected['epoch'])}**",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    epochs = [int(item) for item in args.epochs.split(",") if item.strip()]
    if not epochs:
        raise ValueError("No epochs selected")
    args.result_root.mkdir(parents=True, exist_ok=True)
    rows = [evaluate_epoch(args, epoch) for epoch in epochs]
    ranked, selected = rank_and_select(pd.DataFrame(rows))

    candidate_result_dir = args.result_root / "epoch_gate" / args.candidate
    candidate_result_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = candidate_result_dir / "epoch_downstream_metrics.csv"
    report_path = candidate_result_dir / "epoch_downstream_metrics.md"
    ranked.to_csv(metrics_path, index=False)
    write_report(ranked, selected, report_path)

    source = Path(str(selected["checkpoint"]))
    destination = args.checkpoint_dir / "best_downstream_model.pt"
    shutil.copy2(source, destination)
    metadata = {
        "candidate": args.candidate,
        "selected_epoch": int(selected["epoch"]),
        "source_checkpoint": str(source),
        "best_downstream_checkpoint": str(destination),
        "selection_tasks": ["EC level 1 validation", "GO molecular function validation", "ProteinGym development assays"],
        "selection_policy": (
            "positive task count, non-dominated status, worst per-task percentile rank, "
            "mean per-task percentile rank, then earlier epoch"
        ),
        "metrics_csv": str(metrics_path),
        "report": str(report_path),
    }
    metadata_path = args.checkpoint_dir / "best_downstream_model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(ranked.to_string(index=False), flush=True)
    print(f"Selected epoch {int(selected['epoch'])}: {destination}", flush=True)
    print(f"Wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
