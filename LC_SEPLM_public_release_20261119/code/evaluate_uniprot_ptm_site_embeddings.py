#!/usr/bin/env python3
"""Evaluate ESM2/LoRA residue embeddings on UniProt PTM site prediction.

This is a lightweight residue-level probe:

* Inputs are the protein/site CSVs produced by build_uniprot_ptm_site_dataset.py.
* For each PTM task, select proteins with at least one positive site.
* Candidate negatives are sampled from unannotated residue-type-matched sites on
  the same selected proteins.
* Splits are by protein accession, not by individual sites.
* The classifier is a standard scaled logistic regression probe.

The goal is relative comparison of frozen residue embeddings, not a new
state-of-the-art PTM predictor.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train_esm2_lora_structure_topology import (  # noqa: E402
    load_esm2_full_state_for_inference,
    load_esm2_lora_for_inference,
    load_esm2_pretrained_for_inference,
)


DEFAULT_BASE = Path("data/ptm_uniprot")
DEFAULT_OUT = Path("results/ptm_uniprot")
MODEL_DIR = Path("models")


DEFAULT_MODEL_SPECS = [
    {
        "label": "ESM2_pretrained",
        "kind": "pretrained",
        "model_name": "esm2_t30_150M_UR50D",
        "checkpoint": "",
    },
    {
        "label": "ESM_S_150M",
        "kind": "full_state",
        "model_name": "esm2_t30_150M_UR50D",
        "checkpoint": str(MODEL_DIR / "esm_s_150m.pth"),
    },
    {
        "label": "LRContact_epoch1",
        "kind": "lora",
        "model_name": "",
        "checkpoint": str(MODEL_DIR / "lc_seplm_epoch1.pt"),
    },
    {
        "label": "LRContact_epoch2",
        "kind": "lora",
        "model_name": "",
        "checkpoint": str(
            MODEL_DIR / "lc_seplm_epoch2.pt"
        ),
    },
    {
        "label": "LRContact_epoch3",
        "kind": "lora",
        "model_name": "",
        "checkpoint": str(
            MODEL_DIR / "lc_seplm_epoch3.pt"
        ),
    },
]


def stable_seed(text: str, base: int = 0) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int((int(digest[:8], 16) + base) % (2**31 - 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein-csv", type=Path, default=DEFAULT_BASE / "uniprot_human_reviewed_len80_512_ptm_proteins.csv")
    parser.add_argument("--site-csv", type=Path, default=DEFAULT_BASE / "uniprot_human_reviewed_len80_512_ptm_sites.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--tasks",
        default="phosphorylation_sty,acetylation_k,glycosylation_n",
        help="Comma-separated PTM tasks from the site CSV.",
    )
    parser.add_argument("--max-positive-proteins-per-task", type=int, default=250)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--selection-seed", type=int, default=20260713)
    parser.add_argument("--eval-seeds", default="7,11,13,17,19")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        metavar="LABEL=KIND:VALUE",
        help=(
            "Model to evaluate. KIND is 'pretrained' (VALUE is a fair-esm model name) "
            "or 'lora' (VALUE is a checkpoint path). Supplying this option replaces "
            "the built-in model list; repeat it for multiple models."
        ),
    )
    parser.add_argument(
        "--baseline-label",
        default="ESM2_pretrained",
        help="Model label used as the delta baseline.",
    )
    return parser.parse_args()


def parse_model_specs(values: list[str]) -> list[dict[str, str]]:
    if not values:
        return [dict(spec) for spec in DEFAULT_MODEL_SPECS]
    specs: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value or ":" not in value.split("=", 1)[1]:
            raise ValueError(f"Invalid --model-spec {value!r}; expected LABEL=KIND:VALUE")
        label, payload = value.split("=", 1)
        kind, target = payload.split(":", 1)
        label, kind, target = label.strip(), kind.strip(), target.strip()
        if not label or not target or kind not in {"pretrained", "lora"}:
            raise ValueError(f"Invalid --model-spec {value!r}; KIND must be pretrained or lora")
        if label in seen:
            raise ValueError(f"Duplicate model label: {label}")
        seen.add(label)
        specs.append(
            {
                "label": label,
                "kind": kind,
                "model_name": target if kind == "pretrained" else "",
                "checkpoint": target if kind == "lora" else "",
            }
        )
    return specs


def load_model(spec: dict[str, str], device: torch.device):
    if spec["kind"] == "pretrained":
        return load_esm2_pretrained_for_inference(spec["model_name"], device)
    if spec["kind"] == "full_state":
        return load_esm2_full_state_for_inference(Path(spec["checkpoint"]), spec["model_name"], device)
    return load_esm2_lora_for_inference(Path(spec["checkpoint"]), device)


def encode_sequence(alphabet, sequence: str, device: torch.device) -> torch.Tensor:
    converter = alphabet.get_batch_converter()
    _, _, tokens = converter([("protein", sequence)])
    return tokens.to(device)


def residue_representations(model, alphabet, sequence: str, device: torch.device) -> np.ndarray:
    tokens = encode_sequence(alphabet, sequence, device)
    with torch.no_grad():
        if hasattr(model, "esm_forward"):
            _, reps = model.esm_forward(tokens)
        else:
            outputs = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
            reps = outputs["representations"][model.num_layers]
    offset = 1 if alphabet.prepend_bos else 0
    arr = reps[0, offset : offset + len(sequence)].detach().cpu().numpy().astype(np.float32)
    if arr.shape[0] != len(sequence):
        raise RuntimeError(f"Representation length mismatch: {arr.shape[0]} vs {len(sequence)}")
    return arr


def select_benchmark_sites(
    sites: pd.DataFrame,
    tasks: list[str],
    max_positive_proteins: int,
    negative_ratio: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected_frames = []
    for task in tasks:
        task_df = sites[sites["task"] == task].copy()
        positives = task_df[task_df["label"] == 1]
        positive_accessions = np.array(sorted(positives["accession"].unique()))
        if len(positive_accessions) == 0:
            continue
        if len(positive_accessions) > max_positive_proteins:
            positive_accessions = rng.choice(positive_accessions, size=max_positive_proteins, replace=False)
        task_df = task_df[task_df["accession"].isin(set(positive_accessions))].copy()
        pos_rows = task_df[task_df["label"] == 1].copy()
        neg_rows = task_df[task_df["label"] == 0].copy()
        n_neg = min(len(neg_rows), max(1, negative_ratio * len(pos_rows)))
        neg_rows = neg_rows.sample(n=n_neg, random_state=stable_seed(task, seed))
        selected = pd.concat([pos_rows, neg_rows], ignore_index=True)
        selected["site_uid"] = (
            selected["task"].astype(str)
            + "|"
            + selected["accession"].astype(str)
            + "|"
            + selected["position"].astype(str)
        )
        selected_frames.append(selected)
    if not selected_frames:
        raise ValueError("No selected PTM rows; check task names and labels.")
    out = pd.concat(selected_frames, ignore_index=True)
    return out.sort_values(["task", "accession", "position"]).reset_index(drop=True)


def build_task_features(
    model,
    alphabet,
    proteins: pd.DataFrame,
    selected_sites: pd.DataFrame,
    device: torch.device,
    model_label: str,
) -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
    protein_by_acc = proteins.set_index("accession")
    site_groups = {acc: frame.copy() for acc, frame in selected_sites.groupby("accession")}
    task_features: dict[str, list[np.ndarray]] = {}
    task_meta: dict[str, list[dict]] = {}
    accessions = sorted(site_groups)
    t0 = time.time()
    for idx, accession in enumerate(accessions, start=1):
        if accession not in protein_by_acc.index:
            continue
        sequence = str(protein_by_acc.loc[accession, "sequence"])
        reps = residue_representations(model, alphabet, sequence, device)
        frame = site_groups[accession]
        for _, row in frame.iterrows():
            pos = int(row["position"]) - 1
            if pos < 0 or pos >= reps.shape[0]:
                continue
            task = str(row["task"])
            task_features.setdefault(task, []).append(reps[pos].copy())
            task_meta.setdefault(task, []).append(row.to_dict())
        if idx % 50 == 0 or idx == len(accessions):
            print(
                f"{model_label}: embedded {idx}/{len(accessions)} selected proteins "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    out: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for task, feats in task_features.items():
        out[task] = (np.vstack(feats), pd.DataFrame(task_meta[task]).reset_index(drop=True))
    return out


def split_accessions(meta: pd.DataFrame, seed: int, test_size: float) -> tuple[set[str], set[str]]:
    acc = np.array(sorted(meta["accession"].unique()))
    # All selected proteins have positive sites by construction, but keep a coarse
    # stratification by per-protein positive-site count when possible.
    counts = meta.groupby("accession")["label"].sum().reindex(acc).to_numpy()
    bins = pd.qcut(counts, q=min(4, len(np.unique(counts))), labels=False, duplicates="drop")
    stratify = bins if len(set(bins)) > 1 and min(pd.Series(bins).value_counts()) >= 2 else None
    train_acc, test_acc = train_test_split(
        acc,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return set(train_acc), set(test_acc)


def evaluate_task(
    X: np.ndarray,
    meta: pd.DataFrame,
    model_label: str,
    seed: int,
    test_size: float,
    max_iter: int,
) -> dict[str, object]:
    y = meta["label"].astype(int).to_numpy()
    train_acc, test_acc = split_accessions(meta, seed=seed, test_size=test_size)
    train_mask = meta["accession"].isin(train_acc).to_numpy()
    test_mask = meta["accession"].isin(test_acc).to_numpy()
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError(f"Degenerate labels for {model_label} seed {seed}")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        ),
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "model": model_label,
        "seed": seed,
        "task": str(meta["task"].iloc[0]),
        "train_proteins": len(train_acc),
        "test_proteins": len(test_acc),
        "train_sites": int(train_mask.sum()),
        "test_sites": int(test_mask.sum()),
        "train_positive": int(y_train.sum()),
        "test_positive": int(y_test.sum()),
        "auroc": float(roc_auc_score(y_test, prob)),
        "auprc": float(average_precision_score(y_test, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "mcc": float(matthews_corrcoef(y_test, pred)),
    }


def summarize(results: pd.DataFrame, baseline_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if baseline_label not in set(results["model"]):
        raise ValueError(f"Baseline label {baseline_label!r} is absent from evaluation results")
    base = results[results["model"] == baseline_label][
        ["task", "seed", "auroc", "auprc", "balanced_accuracy", "mcc"]
    ].rename(
        columns={
            "auroc": "base_auroc",
            "auprc": "base_auprc",
            "balanced_accuracy": "base_balanced_accuracy",
            "mcc": "base_mcc",
        }
    )
    merged = results.merge(base, on=["task", "seed"], how="left")
    for metric in ["auroc", "auprc", "balanced_accuracy", "mcc"]:
        merged[f"delta_{metric}"] = merged[metric] - merged[f"base_{metric}"]
    summary = (
        merged.groupby(["task", "model"])
        .agg(
            seeds=("seed", "count"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            balacc_mean=("balanced_accuracy", "mean"),
            mcc_mean=("mcc", "mean"),
            delta_auroc_mean=("delta_auroc", "mean"),
            delta_auprc_mean=("delta_auprc", "mean"),
            positive_delta_auroc=("delta_auroc", lambda x: int((x > 0).sum())),
            positive_delta_auprc=("delta_auprc", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )
    return merged, summary


def main() -> None:
    args = parse_args()
    model_specs = parse_model_specs(args.model_spec)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    eval_seeds = [int(seed.strip()) for seed in args.eval_seeds.split(",") if seed.strip()]
    proteins = pd.read_csv(args.protein_csv)
    sites = pd.read_csv(args.site_csv)
    selected_sites = select_benchmark_sites(
        sites,
        tasks=tasks,
        max_positive_proteins=args.max_positive_proteins_per_task,
        negative_ratio=args.negative_ratio,
        seed=args.selection_seed,
    )
    selected_csv = args.out_dir / "selected_ptm_site_probe_rows.csv"
    selected_sites.to_csv(selected_csv, index=False)

    selected_summary = (
        selected_sites.groupby("task")
        .agg(
            proteins=("accession", "nunique"),
            sites=("label", "count"),
            positives=("label", "sum"),
            negatives=("label", lambda x: int((x == 0).sum())),
            positive_rate=("label", "mean"),
        )
        .reset_index()
    )
    selected_summary.to_csv(args.out_dir / "selected_ptm_site_probe_summary.csv", index=False)
    print(selected_summary.to_string(index=False), flush=True)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    result_rows = []
    for spec in model_specs:
        label = spec["label"]
        print(f"Loading {label} on {device}...", flush=True)
        model, alphabet = load_model(spec, device)
        model.eval()
        task_features = build_task_features(model, alphabet, proteins, selected_sites, device, label)
        for task, (X, meta) in task_features.items():
            for seed in eval_seeds:
                result = evaluate_task(
                    X,
                    meta,
                    model_label=label,
                    seed=seed,
                    test_size=args.test_size,
                    max_iter=args.max_iter,
                )
                result_rows.append(result)
                print(
                    f"{label} {task} seed={seed} "
                    f"AUROC={result['auroc']:.4f} AUPRC={result['auprc']:.4f}",
                    flush=True,
                )
        del model, alphabet, task_features
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = pd.DataFrame(result_rows)
    per_seed_csv = args.out_dir / "ptm_site_probe_per_seed.csv"
    results.to_csv(per_seed_csv, index=False)
    merged, summary = summarize(results, baseline_label=args.baseline_label)
    merged_csv = args.out_dir / "ptm_site_probe_per_seed_with_deltas.csv"
    summary_csv = args.out_dir / "ptm_site_probe_summary.csv"
    merged.to_csv(merged_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    metadata = {
        "protein_csv": str(args.protein_csv),
        "site_csv": str(args.site_csv),
        "selected_sites_csv": str(selected_csv),
        "tasks": tasks,
        "selection_seed": args.selection_seed,
        "eval_seeds": eval_seeds,
        "max_positive_proteins_per_task": args.max_positive_proteins_per_task,
        "negative_ratio": args.negative_ratio,
        "test_size": args.test_size,
        "models": model_specs,
        "baseline_label": args.baseline_label,
        "outputs": {
            "per_seed": str(per_seed_csv),
            "per_seed_with_deltas": str(merged_csv),
            "summary": str(summary_csv),
        },
    }
    (args.out_dir / "ptm_site_probe_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    md = args.out_dir / "ptm_site_probe_summary.md"
    with md.open("w", encoding="utf-8") as handle:
        handle.write("# UniProt PTM site embedding probe\n\n")
        handle.write("## Selected benchmark rows\n\n")
        handle.write(selected_summary.to_markdown(index=False, floatfmt=".6f"))
        handle.write("\n\n## Mean metrics across seeds\n\n")
        handle.write(summary.to_markdown(index=False, floatfmt=".6f"))
        handle.write("\n")
    print(f"Wrote {per_seed_csv}")
    print(f"Wrote {merged_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
