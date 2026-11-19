#!/usr/bin/env python3
"""Strict holdout checks for LRContact ESM2-LoRA downstream evaluations.

The script reuses cached ESM2/LoRA embeddings and evaluates fixed downstream
train splits against stricter held-out subsets:

* exact_accession: downstream val/test rows whose record_id did not appear in
  the AF2 SwissProt pretraining train subset.
* identity: downstream val/test rows not listed as similar to pretraining train
  by an externally generated hit table.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


TASKS = {
    "ec_level1": ("EC", "classification", "prepared_full/uniprot_ec_level1_task.csv", None, "macro_f1"),
    "subcellular_location": (
        "Subcellular",
        "classification",
        "prepared_full/uniprot_subcellular_location_task.csv",
        None,
        "macro_f1",
    ),
    "membrane_binary": (
        "Membrane",
        "classification",
        "prepared_membrane_binary/uniprot_membrane_binary_task.csv",
        None,
        "macro_f1",
    ),
    "strict_membrane_binary_seed23": (
        "Strict membrane",
        "classification",
        "prepared_strict_membrane_binary_seed23/uniprot_strict_membrane_binary_task.csv",
        None,
        "macro_f1",
    ),
    "go_molecular_function": (
        "GO MF",
        "multilabel",
        "prepared_go_pfam/uniprot_go_molecular_function_task.csv",
        ";",
        "micro_f1",
    ),
    "pfam_family": (
        "Pfam family",
        "classification",
        "prepared_go_pfam/uniprot_pfam_family_task.csv",
        None,
        "macro_f1",
    ),
    "pfam_clan_remote_homology": (
        "Remote homology",
        "classification",
        "prepared_go_pfam/uniprot_pfam_clan_remote_homology_task.csv",
        None,
        "macro_f1",
    ),
}


FEATURES = {
    "esm2": "ESM2_pretrained_esm2_t30_150M",
    "epoch1": "LRContact_XAttn_AF2_500k",
    "epoch2": "LRContact_XAttn_AF2_500k_epoch2",
    "epoch3": "LRContact_XAttn_AF2_500k_epoch3",
}


def read_pretrain_accessions(manifest: Path, max_train: int) -> set[str]:
    accessions: list[str] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "train":
                accessions.append(row["accession"].strip())
                if len(accessions) >= max_train:
                    break
    return set(accessions)


def read_excluded_ids(path: Path | None) -> dict[str, set[str]]:
    if path is None:
        return {}
    excluded: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"task", "record_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Excluded-id table missing columns: {sorted(missing)}")
        for row in reader:
            excluded.setdefault(row["task"], set()).add(row["record_id"].strip())
    return excluded


def make_classifier(seed: int, max_iter: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        random_state=seed,
        max_iter=max_iter,
        tol=1e-3,
        n_jobs=-1,
    )


def evaluate_feature(
    x: np.ndarray,
    labels: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    task_type: str,
    delimiter: str | None,
    seed: int,
    max_iter: int,
) -> dict[str, float]:
    if task_type == "classification":
        label_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        y = np.asarray([label_to_id[label] for label in labels], dtype=np.int64)
        clf = make_pipeline(StandardScaler(), make_classifier(seed, max_iter))
        clf.fit(x[train_idx], y[train_idx])
        pred = clf.predict(x[test_idx])
        return {
            "accuracy": float(accuracy_score(y[test_idx], pred)),
            "macro_f1": float(f1_score(y[test_idx], pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y[test_idx], pred, average="weighted", zero_division=0)),
        }
    if task_type != "multilabel":
        raise ValueError(f"Unsupported task type: {task_type}")
    label_lists = [[part.strip() for part in label.split(delimiter or ";") if part.strip()] for label in labels]
    y = MultiLabelBinarizer().fit_transform(label_lists)
    clf = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(make_classifier(seed, max_iter), n_jobs=-1),
    )
    clf.fit(x[train_idx], y[train_idx])
    pred = clf.predict(x[test_idx])
    return {
        "micro_f1": float(f1_score(y[test_idx], pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y[test_idx], pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y[test_idx], pred, average="samples", zero_division=0)),
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) <= 1:
        return (float(values[0]) if values else float("nan"), 0.0)
    return statistics.mean(values), statistics.stdev(values)


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    task_root = root / "sequence_graph_pretraining" / "formal_downstream_tasks"
    cache_root = Path(args.feature_cache_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pretrain_accessions = read_pretrain_accessions(Path(args.pretrain_manifest), args.max_pretrain_train)
    identity_excluded = read_excluded_ids(args.identity_excluded_csv)
    seeds = [int(seed) for seed in args.eval_seeds.split(",") if seed.strip()]
    task_keys = [task.strip() for task in args.tasks.split(",") if task.strip()]

    raw_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for task in task_keys:
        task_label, task_type, rel_path, delimiter, primary_metric = TASKS[task]
        df = pd.read_csv(task_root / rel_path)
        record_ids = df["record_id"].astype(str).str.strip()
        split = df["split"].astype(str).str.lower()
        labels = df["label"].astype(str).tolist()
        train_idx = np.flatnonzero(split.eq("train").to_numpy())
        eval_mask = split.isin(["val", "test"]).to_numpy()
        if args.holdout_mode == "exact_accession":
            excluded_mask = record_ids.isin(pretrain_accessions).to_numpy()
        elif args.holdout_mode == "identity":
            excluded_mask = record_ids.isin(identity_excluded.get(task, set())).to_numpy()
        else:
            raise ValueError(f"Unknown holdout mode: {args.holdout_mode}")
        test_idx = np.flatnonzero(eval_mask & ~excluded_mask)
        if len(test_idx) == 0:
            raise ValueError(f"No held-out examples left for {task}")
        split_rows.append(
            {
                "task": task,
                "task_label": task_label,
                "train_n": int(len(train_idx)),
                "eval_n_original": int(eval_mask.sum()),
                "eval_n_excluded": int((eval_mask & excluded_mask).sum()),
                "eval_n_kept": int(len(test_idx)),
                "eval_excluded_fraction": float((eval_mask & excluded_mask).sum() / max(int(eval_mask.sum()), 1)),
            }
        )
        feature_arrays = {}
        for feature_key, feature_name in FEATURES.items():
            path = cache_root / task / f"{feature_name}_n{len(df)}_maxlen{args.max_residues}.npy"
            if not path.exists():
                raise FileNotFoundError(path)
            feature_arrays[feature_key] = np.load(path, mmap_mode="r")

        for seed in seeds:
            for feature_key, x in feature_arrays.items():
                metrics = evaluate_feature(
                    np.asarray(x),
                    labels,
                    train_idx,
                    test_idx,
                    task_type,
                    delimiter,
                    seed,
                    args.classifier_max_iter,
                )
                raw_rows.append(
                    {
                        "holdout_mode": args.holdout_mode,
                        "task": task,
                        "task_label": task_label,
                        "feature": feature_key,
                        "feature_name": FEATURES[feature_key],
                        "eval_seed": seed,
                        "primary_metric": primary_metric,
                        "primary_score": metrics[primary_metric],
                        **metrics,
                    }
                )
                print(
                    f"{args.holdout_mode} {task} seed={seed} {feature_key} "
                    f"{primary_metric}={metrics[primary_metric]:.6f}",
                    flush=True,
                )

    raw = pd.DataFrame(raw_rows)
    summary_rows: list[dict[str, object]] = []
    for task in task_keys:
        task_label = TASKS[task][0]
        metric = TASKS[task][4]
        esm2 = raw[(raw["task"] == task) & (raw["feature"] == "esm2")]["primary_score"].astype(float).tolist()
        esm2_mean, esm2_std = mean_std(esm2)
        row: dict[str, object] = {
            "holdout_mode": args.holdout_mode,
            "task": task,
            "task_label": task_label,
            "metric": metric,
            "eval_seeds": ",".join(map(str, seeds)),
            "esm2_mean": esm2_mean,
            "esm2_std": esm2_std,
        }
        for feature in ["epoch1", "epoch2", "epoch3"]:
            values = raw[(raw["task"] == task) & (raw["feature"] == feature)]["primary_score"].astype(float).tolist()
            mean, std = mean_std(values)
            deltas = [value - baseline for value, baseline in zip(values, esm2)]
            delta_mean, delta_std = mean_std(deltas)
            row[f"{feature}_mean"] = mean
            row[f"{feature}_std"] = std
            row[f"{feature}_delta"] = delta_mean
            row[f"{feature}_delta_std"] = delta_std
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    split_summary = pd.DataFrame(split_rows)

    prefix = args.output_prefix
    raw_path = out_dir / f"{prefix}_raw.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"
    split_path = out_dir / f"{prefix}_split_stats.csv"
    metadata_path = out_dir / f"{prefix}_metadata.json"
    md_path = out_dir / f"{prefix}_summary.md"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    split_summary.to_csv(split_path, index=False)
    metadata = {
        "holdout_mode": args.holdout_mode,
        "tasks": task_keys,
        "eval_seeds": seeds,
        "features": FEATURES,
        "pretrain_manifest": str(args.pretrain_manifest),
        "max_pretrain_train": args.max_pretrain_train,
        "feature_cache_root": str(cache_root),
        "raw_path": str(raw_path),
        "summary_path": str(summary_path),
        "split_path": str(split_path),
    }
    if args.identity_excluded_csv is not None:
        metadata["identity_excluded_csv"] = str(args.identity_excluded_csv)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# LRContact AF2 500k strict holdout summary ({args.holdout_mode})",
        "",
        f"- Seeds: `{','.join(map(str, seeds))}`",
        f"- Feature cache: `{cache_root}`",
        f"- Summary CSV: `{summary_path}`",
        f"- Raw CSV: `{raw_path}`",
        f"- Split stats CSV: `{split_path}`",
        "",
        "| Task | Metric | ESM2 | Epoch1 delta | Epoch2 delta | Epoch3 delta | Kept eval |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    split_by_task = split_summary.set_index("task").to_dict("index")
    for row in summary_rows:
        kept = split_by_task[row["task"]]["eval_n_kept"]
        lines.append(
            "| {task_label} | `{metric}` | {esm2:.6f} +/- {esm2_std:.6f} | "
            "{e1:+.6f} +/- {e1s:.6f} | {e2:+.6f} +/- {e2s:.6f} | "
            "{e3:+.6f} +/- {e3s:.6f} | {kept} |".format(
                task_label=row["task_label"],
                metric=row["metric"],
                esm2=float(row["esm2_mean"]),
                esm2_std=float(row["esm2_std"]),
                e1=float(row["epoch1_delta"]),
                e1s=float(row["epoch1_delta_std"]),
                e2=float(row["epoch2_delta"]),
                e2s=float(row["epoch2_delta_std"]),
                e3=float(row["epoch3_delta"]),
                e3s=float(row["epoch3_delta_std"]),
                kept=kept,
            )
        )
    lines.extend(["", "Mean delta across tasks:"])
    for feature in ["epoch1", "epoch2", "epoch3"]:
        mean_delta = float(summary[f"{feature}_delta"].mean())
        positives = int((summary[f"{feature}_delta"] > 0).sum())
        lines.append(f"- `{feature}`: `{mean_delta:+.6f}`; positive tasks `{positives}/{len(summary)}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", required=True, help="Project root containing prepared task splits.")
    parser.add_argument(
        "--feature-cache-root",
        required=True,
    )
    parser.add_argument(
        "--pretrain-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        required=True,
    )
    parser.add_argument("--output-prefix", default="lrcontact_af2_500k_exact_accession_holdout")
    parser.add_argument("--holdout-mode", choices=["exact_accession", "identity"], default="exact_accession")
    parser.add_argument("--identity-excluded-csv", type=Path, default=None)
    parser.add_argument("--max-pretrain-train", type=int, default=500000)
    parser.add_argument("--max-residues", type=int, default=512)
    parser.add_argument("--eval-seeds", default="7,17,23,42,101")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--classifier-max-iter", type=int, default=1000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
