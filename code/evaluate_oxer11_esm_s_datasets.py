#!/usr/bin/env python3
"""Evaluate ESM2 / ESM-S / ESM2-LoRA on Oxer11 ESM-S official datasets.

The Oxer11/Protein-Function-Annotation bundle is the dataset used by the
ESM-S repository for fold classification and EC/GO function annotation.
This script keeps the encoder frozen, extracts mean-pooled ESM embeddings,
and trains the same lightweight MLP probe for every encoder.

It intentionally does not use the provided coordinates, because the goal is
to compare sequence-only representations fairly.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "code"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_esm2_lora_structure_topology import (  # noqa: E402
    esm2_sequence_embeddings,
    load_esm2_full_state_for_inference,
    load_esm2_lora_for_inference,
    load_esm2_pretrained_for_inference,
)


BRANCH_ROOT = ROOT / "results"


@dataclass
class SplitData:
    ids: list[str]
    sequences: list[str]
    labels: np.ndarray


@dataclass
class MultilabelTask:
    name: str
    train: SplitData
    valid: SplitData
    tests: dict[str, SplitData]
    class_weights: np.ndarray


@dataclass
class ClassificationTask:
    name: str
    train: SplitData
    valid: SplitData
    tests: dict[str, SplitData]
    num_classes: int


class ProbeMLP(nn.Module):
    """MLP head matching the ESM-S / TorchDrug predictor shape."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2, batch_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        if batch_norm:
            layers.append(nn.BatchNorm1d(in_dim, momentum=0.2))
        layers.extend(
            [
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
                nn.Linear(in_dim, in_dim),
            ]
        )
        if batch_norm:
            layers.append(nn.BatchNorm1d(in_dim, momentum=0.2))
        layers.extend([nn.LeakyReLU(0.2), nn.Linear(in_dim, out_dim)])
        self.module = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].strip()
                records[current] = []
            elif current is not None:
                records[current].append(line)
            else:
                raise ValueError(f"FASTA sequence before header in {path}")
    return {key: "".join(parts).upper() for key, parts in records.items()}


def read_fold_labels(path: Path, class_map: dict[str, int]) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            parts = raw.rstrip("\n").split("\t")
            labels[parts[0]] = class_map[parts[-1]]
    return labels


def make_classification_split(
    ids: Iterable[str],
    seqs: dict[str, str],
    labels: dict[str, int],
) -> SplitData:
    kept_ids: list[str] = []
    kept_seqs: list[str] = []
    kept_labels: list[int] = []
    for protein_id in ids:
        if protein_id not in seqs or protein_id not in labels:
            continue
        kept_ids.append(protein_id)
        kept_seqs.append(seqs[protein_id])
        kept_labels.append(labels[protein_id])
    return SplitData(kept_ids, kept_seqs, np.asarray(kept_labels, dtype=np.int64))


def load_fold_task(data_root: Path) -> ClassificationTask:
    path = data_root / "fold"
    class_map: dict[str, int] = {}
    with (path / "class_map.txt").open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            fold_id, idx = raw.rstrip("\n").split("\t")
            class_map[fold_id] = int(idx)

    seqs_by_split = {
        split: read_fasta(path / f"{split}.fasta")
        for split in ["training", "validation", "test_family", "test_superfamily", "test_fold"]
    }
    labels_by_split = {
        split: read_fold_labels(path / f"{split}.txt", class_map)
        for split in ["training", "validation", "test_family", "test_superfamily", "test_fold"]
    }
    return ClassificationTask(
        name="fold",
        train=make_classification_split(seqs_by_split["training"].keys(), seqs_by_split["training"], labels_by_split["training"]),
        valid=make_classification_split(
            seqs_by_split["validation"].keys(), seqs_by_split["validation"], labels_by_split["validation"]
        ),
        tests={
            split: make_classification_split(seqs_by_split[split].keys(), seqs_by_split[split], labels_by_split[split])
            for split in ["test_fold", "test_superfamily", "test_family"]
        },
        num_classes=max(class_map.values()) + 1,
    )


def read_percent_test_ids(path: Path, percents: list[int]) -> dict[int, set[str]]:
    columns = {30: "<30%", 40: "<40%", 50: "<50%", 70: "<70%", 95: "<95%"}
    out = {percent: set() for percent in percents}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            protein_id = row["PDB-chain"].strip()
            for percent in percents:
                if row[columns[percent]].strip() == "1":
                    out[percent].add(protein_id)
    return out


def labels_to_matrix(ids: Iterable[str], label_map: dict[str, list[int]], num_classes: int) -> np.ndarray:
    ids_list = list(ids)
    y = np.zeros((len(ids_list), num_classes), dtype=np.float32)
    for row_idx, protein_id in enumerate(ids_list):
        for class_idx in label_map.get(protein_id, []):
            y[row_idx, class_idx] = 1.0
    return y


def make_multilabel_split(
    ids: Iterable[str],
    seqs: dict[str, str],
    label_map: dict[str, list[int]],
    num_classes: int,
) -> SplitData:
    kept_ids: list[str] = []
    kept_seqs: list[str] = []
    for protein_id in ids:
        if protein_id not in seqs:
            continue
        kept_ids.append(protein_id)
        kept_seqs.append(seqs[protein_id])
    return SplitData(kept_ids, kept_seqs, labels_to_matrix(kept_ids, label_map, num_classes))


def read_ec_annotations(path: Path) -> tuple[list[str], dict[str, list[int]], np.ndarray]:
    terms: list[str] = []
    label_map: dict[str, list[int]] = {}
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for idx, raw in enumerate(handle):
            line = raw.rstrip("\n")
            if idx == 1:
                terms = [part for part in line.split("\t") if part]
                counts = {term: 0 for term in terms}
                term_to_idx = {term: term_idx for term_idx, term in enumerate(terms)}
            elif idx > 2 and line:
                parts = line.split("\t")
                protein_id = parts[0]
                labels: list[int] = []
                if len(parts) > 1:
                    for term in parts[1].split(","):
                        if term:
                            labels.append(term_to_idx[term])
                            counts[term] += 1
                label_map[protein_id] = labels
    weights = np.ones((len(terms),), dtype=np.float32)
    denom_n = max(len(label_map), 1)
    for term, term_idx in {term: idx for idx, term in enumerate(terms)}.items():
        weights[term_idx] = float(denom_n) / float(max(counts.get(term, 0), 1))
    return terms, label_map, weights


def load_ec_task(data_root: Path, percents: list[int]) -> MultilabelTask:
    path = data_root / "ec"
    train_seqs = read_fasta(path / "train.fasta")
    valid_seqs = read_fasta(path / "valid.fasta")
    test_seqs = read_fasta(path / "test.fasta")
    terms, label_map, weights = read_ec_annotations(path / "nrPDB-EC_annot.tsv")
    test_ids = read_percent_test_ids(path / "nrPDB-EC_test.csv", percents)
    return MultilabelTask(
        name="ec",
        train=make_multilabel_split(train_seqs.keys(), train_seqs, label_map, len(terms)),
        valid=make_multilabel_split(valid_seqs.keys(), valid_seqs, label_map, len(terms)),
        tests={
            f"test_{percent}": make_multilabel_split(
                [protein_id for protein_id in test_seqs if protein_id in test_ids[percent]],
                test_seqs,
                label_map,
                len(terms),
            )
            for percent in percents
        },
        class_weights=weights,
    )


def read_go_annotations(path: Path, level: str) -> tuple[list[str], dict[str, list[int]], np.ndarray]:
    level_to_header_idx = {"mf": 1, "bp": 5, "cc": 9}
    level_to_label_idx = {"mf": 1, "bp": 2, "cc": 3}
    header_idx = level_to_header_idx[level]
    label_idx = level_to_label_idx[level]
    terms: list[str] = []
    label_map: dict[str, list[int]] = {}
    counts: dict[str, int] = {}
    term_to_idx: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for idx, raw in enumerate(handle):
            line = raw.rstrip("\n")
            if idx == header_idx:
                terms = [part for part in line.split("\t") if part]
                counts = {term: 0 for term in terms}
                term_to_idx = {term: term_idx for term_idx, term in enumerate(terms)}
            elif idx > 12 and line:
                parts = line.split("\t")
                protein_id = parts[0]
                labels: list[int] = []
                if len(parts) > label_idx:
                    for term in parts[label_idx].split(","):
                        if term:
                            labels.append(term_to_idx[term])
                            counts[term] += 1
                label_map[protein_id] = labels
    weights = np.ones((len(terms),), dtype=np.float32)
    denom_n = max(len(label_map), 1)
    for term, term_idx in term_to_idx.items():
        weights[term_idx] = float(denom_n) / float(max(counts.get(term, 0), 1))
    return terms, label_map, weights


def load_go_task(data_root: Path, level: str, percents: list[int]) -> MultilabelTask:
    path = data_root / "go"
    train_seqs = read_fasta(path / "train.fasta")
    valid_seqs = read_fasta(path / "valid.fasta")
    test_seqs = read_fasta(path / "test.fasta")
    terms, label_map, weights = read_go_annotations(path / "nrPDB-GO_2019.06.18_annot.tsv", level)
    test_ids = read_percent_test_ids(path / "nrPDB-GO_2019.06.18_test.csv", percents)
    return MultilabelTask(
        name=f"go_{level}",
        train=make_multilabel_split(train_seqs.keys(), train_seqs, label_map, len(terms)),
        valid=make_multilabel_split(valid_seqs.keys(), valid_seqs, label_map, len(terms)),
        tests={
            f"test_{percent}": make_multilabel_split(
                [protein_id for protein_id in test_seqs if protein_id in test_ids[percent]],
                test_seqs,
                label_map,
                len(terms),
            )
            for percent in percents
        },
        class_weights=weights,
    )


def combine_splits(task: ClassificationTask | MultilabelTask) -> tuple[list[str], list[str], dict[str, np.ndarray]]:
    ids: list[str] = []
    seqs: list[str] = []
    index: dict[str, int] = {}

    def add(split: SplitData) -> None:
        for protein_id, sequence in zip(split.ids, split.sequences):
            if protein_id in index:
                continue
            index[protein_id] = len(ids)
            ids.append(protein_id)
            seqs.append(sequence)

    add(task.train)
    add(task.valid)
    for split in task.tests.values():
        add(split)

    split_indices: dict[str, np.ndarray] = {
        "train": np.asarray([index[protein_id] for protein_id in task.train.ids], dtype=np.int64),
        "valid": np.asarray([index[protein_id] for protein_id in task.valid.ids], dtype=np.int64),
    }
    for split_name, split in task.tests.items():
        split_indices[split_name] = np.asarray([index[protein_id] for protein_id in split.ids], dtype=np.int64)
    return ids, seqs, split_indices


def parse_model_specs(values: list[str]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for value in values:
        if "=" not in value or ":" not in value:
            raise ValueError(
                "--model-spec must look like NAME=pretrained:MODEL, NAME=full_state:MODEL,CHECKPOINT, "
                "or NAME=lora:CHECKPOINT"
            )
        name, payload = value.split("=", 1)
        kind, target = payload.split(":", 1)
        if kind == "pretrained":
            specs.append({"name": name, "kind": kind, "model_name": target})
        elif kind == "full_state":
            if "," not in target:
                raise ValueError(f"full_state spec needs MODEL_NAME,CHECKPOINT: {value}")
            model_name, checkpoint = target.split(",", 1)
            specs.append({"name": name, "kind": kind, "model_name": model_name, "checkpoint": checkpoint})
        elif kind == "lora":
            specs.append({"name": name, "kind": kind, "checkpoint": target})
        else:
            raise ValueError(f"Unknown model kind in {value}")
    return specs


def default_model_specs() -> list[str]:
    return [
        "ESM2_pretrained=pretrained:esm2_t30_150M_UR50D",
        f"ESM_S_150M=full_state:esm2_t30_150M_UR50D,models/esm_s_150m.pth",
        f"LC_SEPLM_epoch10=lora:models/lc_seplm_epoch10.pt",
    ]


def load_encoder(spec: dict[str, str], device: torch.device):
    if spec["kind"] == "pretrained":
        return load_esm2_pretrained_for_inference(spec["model_name"], device)
    if spec["kind"] == "full_state":
        return load_esm2_full_state_for_inference(Path(spec["checkpoint"]), spec["model_name"], device)
    if spec["kind"] == "lora":
        return load_esm2_lora_for_inference(Path(spec["checkpoint"]), device)
    raise ValueError(spec)


def feature_cache_path(cache_dir: Path, task_name: str, model_name: str, n: int, max_residues: int) -> Path:
    return cache_dir / safe_name(task_name) / f"{safe_name(model_name)}_n{n}_maxres{max_residues}.npy"


def embedding_cache_task_name(task_name: str) -> str:
    """Share GO embeddings across MF/BP/CC because their sequences are identical."""
    if task_name.startswith("go_"):
        return "go"
    return task_name


def get_embeddings(
    spec: dict[str, str],
    task_name: str,
    sequences: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    cache_task_name = embedding_cache_task_name(task_name)
    cache_path = feature_cache_path(args.feature_cache_dir, cache_task_name, spec["name"], len(sequences), args.max_residues)
    if cache_path.exists():
        return np.load(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "extract_embeddings_start",
                "model": spec["name"],
                "task": task_name,
                "n": len(sequences),
                "max_residues": args.max_residues,
                "time": time.strftime("%F %T"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    model, alphabet = load_encoder(spec, device)
    chunks: list[np.ndarray] = []
    for start in range(0, len(sequences), args.embedding_log_every):
        end = min(start + args.embedding_log_every, len(sequences))
        chunks.append(
            esm2_sequence_embeddings(
                model,
                alphabet,
                sequences[start:end],
                args.max_residues,
                args.embedding_batch_size,
                device,
                use_amp=args.amp,
            )
        )
        print(
            json.dumps(
                {
                    "event": "extract_embeddings_progress",
                    "model": spec["name"],
                    "task": task_name,
                    "done": end,
                    "total": len(sequences),
                    "time": time.strftime("%F %T"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    features = np.vstack(chunks)
    tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, features.astype(np.float32))
    os.replace(tmp_path, cache_path)
    del model, alphabet
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "event": "extract_embeddings_done",
                "model": spec["name"],
                "task": task_name,
                "shape": list(features.shape),
                "path": str(cache_path),
                "time": time.strftime("%F %T"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return features


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, task_type: str) -> DataLoader:
    y_tensor = torch.as_tensor(y, dtype=torch.long if task_type == "classification" else torch.float32)
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32), y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


def f1_max_torchdrug_style(scores: np.ndarray, target: np.ndarray) -> float:
    pred = torch.as_tensor(scores, dtype=torch.float32)
    y = torch.as_tensor(target, dtype=torch.float32)
    if pred.numel() == 0:
        return 0.0
    order = pred.argsort(descending=True, dim=1)
    y_ordered = y.gather(1, order)
    precision = y_ordered.cumsum(1) / torch.ones_like(y_ordered).cumsum(1)
    recall = y_ordered.cumsum(1) / (y_ordered.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(y_ordered).bool()
    is_start[:, 0] = True
    is_start = torch.scatter(is_start, 1, order, is_start)
    all_order = pred.flatten().argsort(descending=True)
    flat_order = order + torch.arange(order.shape[0]).unsqueeze(1) * order.shape[1]
    flat_order = flat_order.flatten()
    inv_order = torch.zeros_like(flat_order)
    inv_order[flat_order] = torch.arange(flat_order.shape[0])
    is_start_flat = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision_flat = precision.flatten()
    recall_flat = recall.flatten()
    all_precision = precision_flat[all_order] - torch.where(
        is_start_flat, torch.zeros_like(precision_flat), precision_flat[all_order - 1]
    )
    all_precision = all_precision.cumsum(0) / is_start_flat.cumsum(0)
    all_recall = recall_flat[all_order] - torch.where(
        is_start_flat, torch.zeros_like(recall_flat), recall_flat[all_order - 1]
    )
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return float(all_f1.max().item())


def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.as_tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            outputs.append(logits.float().cpu().numpy())
    return np.vstack(outputs)


def classification_metrics(logits: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }


def multilabel_metrics(logits: np.ndarray, y: np.ndarray) -> dict[str, float]:
    scores = 1.0 / (1.0 + np.exp(-logits))
    flat_y = y.reshape(-1)
    flat_scores = scores.reshape(-1)
    try:
        micro_auprc = float(average_precision_score(flat_y, flat_scores))
    except ValueError:
        micro_auprc = 0.0
    pred05 = (scores >= 0.5).astype(np.float32)
    return {
        "auprc_micro": micro_auprc,
        "f1_max": f1_max_torchdrug_style(scores, y),
        "micro_f1_at_0p5": float(f1_score(y, pred05, average="micro", zero_division=0)),
        "samples_f1_at_0p5": float(f1_score(y, pred05, average="samples", zero_division=0)),
    }


def train_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    task_type: str,
    out_dim: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    class_weights: np.ndarray | None = None,
) -> tuple[nn.Module, dict[str, float]]:
    set_seed(seed)
    model = ProbeMLP(x_train.shape[1], out_dim, dropout=args.probe_dropout, batch_norm=not args.no_batch_norm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.probe_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.6, patience=5)
    if task_type == "classification":
        criterion: nn.Module = nn.CrossEntropyLoss()
        primary_key = "accuracy"
    else:
        weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
        criterion = nn.BCEWithLogitsLoss(weight=weight_tensor)
        primary_key = "f1_max"
    loader = make_loader(x_train, y_train, args.probe_batch_size, shuffle=True, task_type=task_type)
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = float("-inf")
    best_valid: dict[str, float] = {}
    bad_epochs = 0
    for epoch in range(1, args.probe_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        valid_logits = predict(model, x_valid, args.eval_batch_size, device)
        valid_metrics = (
            classification_metrics(valid_logits, y_valid)
            if task_type == "classification"
            else multilabel_metrics(valid_logits, y_valid)
        )
        metric = valid_metrics[primary_key]
        scheduler.step(metric)
        if metric > best_metric + 1e-9:
            best_metric = metric
            best_valid = valid_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_valid


def evaluate_classification_task(
    task: ClassificationTask,
    features: np.ndarray,
    split_indices: dict[str, np.ndarray],
    model_name: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    probe, valid_metrics = train_probe(
        features[split_indices["train"]],
        task.train.labels,
        features[split_indices["valid"]],
        task.valid.labels,
        "classification",
        task.num_classes,
        seed,
        args,
        device,
    )
    rows: list[dict[str, object]] = []
    for split_name, split in task.tests.items():
        logits = predict(probe, features[split_indices[split_name]], args.eval_batch_size, device)
        metrics = classification_metrics(logits, split.labels)
        rows.append(
            {
                "task": task.name,
                "split": split_name,
                "model": model_name,
                "seed": seed,
                "num_train": len(task.train.ids),
                "num_valid": len(task.valid.ids),
                "num_test": len(split.ids),
                "valid_accuracy": valid_metrics.get("accuracy", 0.0),
                "valid_macro_f1": valid_metrics.get("macro_f1", 0.0),
                **metrics,
            }
        )
    return rows


def evaluate_multilabel_task(
    task: MultilabelTask,
    features: np.ndarray,
    split_indices: dict[str, np.ndarray],
    model_name: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    probe, valid_metrics = train_probe(
        features[split_indices["train"]],
        task.train.labels,
        features[split_indices["valid"]],
        task.valid.labels,
        "multilabel",
        task.train.labels.shape[1],
        seed,
        args,
        device,
        class_weights=task.class_weights,
    )
    rows: list[dict[str, object]] = []
    for split_name, split in task.tests.items():
        logits = predict(probe, features[split_indices[split_name]], args.eval_batch_size, device)
        metrics = multilabel_metrics(logits, split.labels)
        rows.append(
            {
                "task": task.name,
                "split": split_name,
                "model": model_name,
                "seed": seed,
                "num_train": len(task.train.ids),
                "num_valid": len(task.valid.ids),
                "num_test": len(split.ids),
                "num_classes": task.train.labels.shape[1],
                "valid_auprc_micro": valid_metrics.get("auprc_micro", 0.0),
                "valid_f1_max": valid_metrics.get("f1_max", 0.0),
                **metrics,
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task"]), str(row["split"]), str(row["model"])), []).append(row)
    out_rows: list[dict[str, object]] = []
    skip = {"task", "split", "model", "seed"}
    for (task, split, model), group in grouped.items():
        out: dict[str, object] = {"task": task, "split": split, "model": model, "seeds": len(group)}
        metric_keys = [
            key
            for key in group[0].keys()
            if key not in skip and isinstance(group[0].get(key), (int, float, np.integer, np.floating))
        ]
        for key in metric_keys:
            values = np.asarray([float(row[key]) for row in group], dtype=np.float64)
            out[f"{key}_mean"] = float(values.mean())
            out[f"{key}_std"] = float(values.std())
        out_rows.append(out)
    return sorted(out_rows, key=lambda row: (str(row["task"]), str(row["split"]), str(row["model"])))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_str_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=BRANCH_ROOT / "oxer11_esm_s_official_eval",
    )
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-spec", action="append", default=[])
    parser.add_argument("--tasks", default="fold,ec,go_mf,go_bp,go_cc")
    parser.add_argument("--percents", default="95,50,30")
    parser.add_argument("--seeds", default="7,17,23")
    parser.add_argument("--max-residues", type=int, default=550)
    parser.add_argument("--embedding-batch-size", type=int, default=12)
    parser.add_argument("--embedding-log-every", type=int, default=1024)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--probe-lr", type=float, default=1e-4)
    parser.add_argument("--probe-dropout", type=float, default=0.2)
    parser.add_argument("--no-batch-norm", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.feature_cache_dir is None:
        args.feature_cache_dir = args.out_dir / "feature_cache"
    device = torch.device(args.device)
    model_specs = parse_model_specs(args.model_spec or default_model_specs())
    seeds = parse_int_list(args.seeds)
    percents = parse_int_list(args.percents)
    task_names = set(parse_str_list(args.tasks))

    tasks: list[ClassificationTask | MultilabelTask] = []
    if "fold" in task_names:
        tasks.append(load_fold_task(args.data_root))
    if "ec" in task_names:
        tasks.append(load_ec_task(args.data_root, percents))
    for level in ["mf", "bp", "cc"]:
        if f"go_{level}" in task_names:
            tasks.append(load_go_task(args.data_root, level, percents))

    metadata = {
        "data_root": str(args.data_root),
        "out_dir": str(args.out_dir),
        "feature_cache_dir": str(args.feature_cache_dir),
        "device": args.device,
        "models": model_specs,
        "tasks": [task.name for task in tasks],
        "percents": percents,
        "seeds": seeds,
        "max_residues": args.max_residues,
        "probe": {
            "type": "frozen_embedding_mlp",
            "epochs": args.probe_epochs,
            "early_stop_patience": args.early_stop_patience,
            "lr": args.probe_lr,
            "dropout": args.probe_dropout,
            "batch_norm": not args.no_batch_norm,
            "batch_size": args.probe_batch_size,
        },
        "timestamp": time.strftime("%F %T"),
    }
    with (args.out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    all_rows: list[dict[str, object]] = []
    raw_path = args.out_dir / "oxer11_esm_s_eval_raw.csv"
    summary_path = args.out_dir / "oxer11_esm_s_eval_summary.csv"

    for task in tasks:
        ids, sequences, split_indices = combine_splits(task)
        id_path = args.out_dir / f"{task.name}_ids.json"
        with id_path.open("w", encoding="utf-8") as handle:
            json.dump({"ids": ids, "splits": {k: v.tolist() for k, v in split_indices.items()}}, handle, indent=2)
        print(
            json.dumps(
                {
                    "event": "task_loaded",
                    "task": task.name,
                    "num_unique_sequences": len(sequences),
                    "num_train": len(task.train.ids),
                    "num_valid": len(task.valid.ids),
                    "tests": {key: len(value.ids) for key, value in task.tests.items()},
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        for spec in model_specs:
            features = get_embeddings(spec, task.name, sequences, args, device)
            if features.shape[0] != len(sequences):
                raise ValueError(f"Feature row mismatch for {spec['name']} {task.name}: {features.shape} vs {len(sequences)}")
            for seed in seeds:
                print(
                    json.dumps(
                        {"event": "probe_start", "task": task.name, "model": spec["name"], "seed": seed},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if isinstance(task, ClassificationTask):
                    rows = evaluate_classification_task(task, features, split_indices, spec["name"], seed, args, device)
                else:
                    rows = evaluate_multilabel_task(task, features, split_indices, spec["name"], seed, args, device)
                all_rows.extend(rows)
                write_csv(raw_path, all_rows)
                write_csv(summary_path, summarize(all_rows))
                print(
                    json.dumps(
                        {"event": "probe_done", "task": task.name, "model": spec["name"], "seed": seed, "rows": rows},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary_rows = summarize(all_rows)
    write_csv(raw_path, all_rows)
    write_csv(summary_path, summary_rows)
    print(
        json.dumps(
            {
                "event": "done",
                "raw_path": str(raw_path),
                "summary_path": str(summary_path),
                "num_raw_rows": len(all_rows),
                "num_summary_rows": len(summary_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
