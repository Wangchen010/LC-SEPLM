#!/usr/bin/env python3
"""Train an attention-free TCN with residue-level structure topology supervision.

The input is still protein sequence only.  AlphaFold/ESMFold structures are used
offline to create residue labels such as contact degree, betweenness, and
community bridge score.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_betweenness_pair_encoder import (
    PAD_IDX,
    UNK_IDX,
    VOCAB_SIZE,
    build_model,
    encode_sequence,
)


DEFAULT_TARGETS = (
    "contact_degree",
    "long_range_contact_count",
    "betweenness_unweighted",
    "community_bridge_score",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mask_tokens_for_mlm(tokens: torch.Tensor, mask_probability: float) -> tuple[torch.Tensor, torch.Tensor]:
    valid = tokens.ge(2)
    mask = (torch.rand(tokens.shape, device=tokens.device) < mask_probability) & valid

    no_mask = valid.any(dim=1) & ~mask.any(dim=1)
    if no_mask.any():
        row_indices = no_mask.nonzero(as_tuple=False).squeeze(1)
        valid_float = valid[row_indices].float()
        sampled = torch.multinomial(valid_float, 1).squeeze(1)
        mask[row_indices, sampled] = True

    labels = torch.full_like(tokens, -100)
    labels[mask] = tokens[mask]
    masked_tokens = tokens.clone()
    masked_tokens[mask] = UNK_IDX
    return masked_tokens, labels


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    valid = labels.ne(-100)
    if not valid.any():
        return 0, 0
    pred = logits.argmax(dim=-1)
    return int((pred[valid] == labels[valid]).sum().item()), int(valid.sum().item())


def transform_values(values: np.ndarray, transform: str) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    if transform == "raw":
        return values
    if transform == "log1p":
        clipped = np.clip(values, a_min=0.0, a_max=None)
        return np.log1p(clipped)
    raise ValueError(f"Unknown target transform: {transform}")


@dataclass
class TargetScaler:
    target: str
    transform: str
    mean: float
    std: float
    bin_edges: list[float]

    @classmethod
    def fit(cls, target: str, values: np.ndarray, transform: str, num_bins: int) -> "TargetScaler":
        transformed = transform_values(values, transform)
        finite = transformed[np.isfinite(transformed)]
        if finite.size == 0:
            raise ValueError(f"No finite train values for target {target}")
        mean = float(finite.mean())
        std = float(finite.std())
        if std < 1e-8:
            std = 1.0
        quantiles = [idx / num_bins for idx in range(1, num_bins)]
        edges = [float(np.quantile(finite, q)) for q in quantiles]
        return cls(target=target, transform=transform, mean=mean, std=std, bin_edges=edges)

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        transformed = transform_values(values, self.transform)
        return ((transformed - self.mean) / self.std).astype(np.float32)

    def bin_array(self, values: np.ndarray, num_bins: int) -> np.ndarray:
        transformed = transform_values(values, self.transform)
        bins = np.digitize(transformed, self.bin_edges, right=True)
        return np.clip(bins, 0, num_bins - 1).astype(np.int64)

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "transform": self.transform,
            "mean": self.mean,
            "std": self.std,
            "bin_edges": self.bin_edges,
        }


def read_residue_labels(path: Path, targets: list[str]) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    required = {"protein_id", "split", "full_sequence", "aa", "source_position", *targets}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df.copy()
    df["source_position"] = pd.to_numeric(df["source_position"], errors="coerce").astype("Int64")
    for target in targets:
        df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=["source_position", *targets])
    return df


def fit_target_scalers(df: pd.DataFrame, targets: list[str], transform: str, num_bins: int) -> dict[str, TargetScaler]:
    train_df = df[df["split"] == "train"]
    if train_df.empty:
        raise ValueError("No train rows found in residue label dataset.")
    return {
        target: TargetScaler.fit(target, train_df[target].to_numpy(dtype=np.float64), transform, num_bins)
        for target in targets
    }


class StructureTopologyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        split: str,
        targets: list[str],
        scalers: dict[str, TargetScaler],
        max_len: int,
        objective: str,
        num_bins: int,
        max_proteins: int | None = None,
    ):
        split_df = df[df["split"] == split]
        self.examples: list[dict[str, object]] = []
        for protein_id, group in split_df.groupby("protein_id", sort=True):
            if max_proteins is not None and len(self.examples) >= max_proteins:
                break
            sequence = str(group["full_sequence"].iloc[0])
            if not sequence:
                continue
            tokens = torch.tensor(encode_sequence(sequence, max_len), dtype=torch.long)
            valid = np.zeros((max_len, len(targets)), dtype=bool)
            regression = np.zeros((max_len, len(targets)), dtype=np.float32)
            bins = np.full((max_len, len(targets)), -100, dtype=np.int64)

            for _, row in group.iterrows():
                position = int(row["source_position"]) - 1
                if position < 0 or position >= max_len:
                    continue
                for target_idx, target in enumerate(targets):
                    value = float(row[target])
                    if not np.isfinite(value):
                        continue
                    valid[position, target_idx] = True
                    regression[position, target_idx] = scalers[target].transform_array(np.asarray([value]))[0]
                    bins[position, target_idx] = scalers[target].bin_array(np.asarray([value]), num_bins)[0]

            if not valid.any():
                continue
            self.examples.append(
                {
                    "protein_id": str(protein_id),
                    "sequence": sequence,
                    "tokens": tokens,
                    "valid": torch.tensor(valid, dtype=torch.bool),
                    "regression": torch.tensor(regression, dtype=torch.float32),
                    "bins": torch.tensor(bins, dtype=torch.long),
                }
            )

        if not self.examples:
            raise ValueError(f"No usable {split} examples after max_len filtering.")
        self.objective = objective

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.examples[idx]


class TCNStructureTopologyModel(nn.Module):
    def __init__(self, model_config: dict[str, object], num_targets: int, num_bins: int, objective: str):
        super().__init__()
        if model_config.get("architecture") != "tcn":
            raise ValueError("This script is intentionally restricted to attention-free TCN.")
        self.encoder = build_model(model_config)
        channels = int(model_config["channels"])
        self.objective = objective
        self.num_targets = num_targets
        self.num_bins = num_bins
        self.topology_head = nn.Sequential(
            nn.Dropout(float(model_config.get("dropout", 0.25))),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(float(model_config.get("dropout", 0.25))),
            nn.Linear(channels, num_targets * (num_bins if objective == "bins" else 1)),
        )
        self.mlm_decoder = nn.Conv1d(channels, VOCAB_SIZE, kernel_size=1)

    def token_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(PAD_IDX).unsqueeze(1)
        hidden = self.encoder.input_projection(self.encoder.embedding(tokens).transpose(1, 2))
        for block in self.encoder.blocks:
            hidden = block(hidden)
            hidden = hidden.masked_fill(~mask, 0.0)
        return hidden

    def topology_outputs(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.token_hidden(tokens).transpose(1, 2)
        outputs = self.topology_head(hidden)
        if self.objective == "bins":
            return outputs.view(tokens.shape[0], tokens.shape[1], self.num_targets, self.num_bins)
        return outputs.view(tokens.shape[0], tokens.shape[1], self.num_targets)

    def mlm_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.mlm_decoder(self.token_hidden(tokens)).transpose(1, 2)


def model_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "architecture": "tcn",
        "embed_dim": args.embed_dim,
        "channels": args.channels,
        "num_layers": args.tcn_layers,
        "kernel_size": args.kernel_size,
        "dropout": args.dropout,
    }


def batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        "tokens": batch["tokens"].to(device),
        "valid": batch["valid"].to(device),
        "regression": batch["regression"].to(device),
        "bins": batch["bins"].to(device),
        "protein_id": batch["protein_id"],
        "sequence": batch["sequence"],
    }


def topology_loss_and_metrics(
    model: TCNStructureTopologyModel,
    batch: dict[str, object],
    target_names: list[str],
    objective: str,
    num_bins: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    outputs = model.topology_outputs(batch["tokens"])
    valid = batch["valid"]
    metrics: dict[str, float | int] = {}
    losses: list[torch.Tensor] = []

    if objective == "bins":
        labels = batch["bins"]
        for target_idx, target in enumerate(target_names):
            target_valid = valid[:, :, target_idx] & labels[:, :, target_idx].ge(0)
            count = int(target_valid.sum().item())
            metrics[f"{target}_count"] = count
            if count == 0:
                metrics[f"{target}_loss"] = 0.0
                metrics[f"{target}_accuracy"] = 0.0
                continue
            logits = outputs[:, :, target_idx, :][target_valid]
            target_labels = labels[:, :, target_idx][target_valid]
            loss = nn.functional.cross_entropy(logits, target_labels)
            losses.append(loss)
            pred = logits.argmax(dim=-1)
            correct = int((pred == target_labels).sum().item())
            metrics[f"{target}_loss"] = float(loss.detach().item())
            metrics[f"{target}_accuracy"] = correct / max(count, 1)
            metrics[f"{target}_correct"] = correct
        if not losses:
            return torch.tensor(0.0, device=batch["tokens"].device), metrics
        metrics["topology_accuracy_mean"] = float(
            np.mean([metrics[f"{target}_accuracy"] for target in target_names if metrics[f"{target}_count"]])
        )
        metrics["topology_loss_mean"] = float(np.mean([metrics[f"{target}_loss"] for target in target_names]))
        return torch.stack(losses).mean(), metrics

    labels = batch["regression"]
    for target_idx, target in enumerate(target_names):
        target_valid = valid[:, :, target_idx]
        count = int(target_valid.sum().item())
        metrics[f"{target}_count"] = count
        if count == 0:
            metrics[f"{target}_loss"] = 0.0
            continue
        pred = outputs[:, :, target_idx][target_valid]
        target_values = labels[:, :, target_idx][target_valid]
        loss = nn.functional.smooth_l1_loss(pred, target_values)
        losses.append(loss)
        metrics[f"{target}_loss"] = float(loss.detach().item())
    if not losses:
        return torch.tensor(0.0, device=batch["tokens"].device), metrics
    metrics["topology_loss_mean"] = float(np.mean([metrics[f"{target}_loss"] for target in target_names]))
    return torch.stack(losses).mean(), metrics


def aggregate_epoch_metrics(rows: list[dict[str, float | int]], targets: list[str], objective: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not rows:
        return out
    out["topology_loss_mean"] = float(np.mean([float(row.get("topology_loss_mean", 0.0)) for row in rows]))
    if objective == "bins":
        accuracies = []
        for target in targets:
            correct = sum(int(row.get(f"{target}_correct", 0)) for row in rows)
            count = sum(int(row.get(f"{target}_count", 0)) for row in rows)
            acc = correct / max(count, 1)
            out[f"{target}_accuracy"] = acc
            accuracies.append(acc)
        out["topology_accuracy_mean"] = float(np.mean(accuracies))
    mlm_correct = sum(int(row.get("mlm_correct", 0)) for row in rows)
    mlm_total = sum(int(row.get("mlm_total", 0)) for row in rows)
    if mlm_total:
        out["mlm_accuracy"] = mlm_correct / mlm_total
    if any("mlm_loss" in row for row in rows):
        out["mlm_loss"] = float(np.mean([float(row.get("mlm_loss", 0.0)) for row in rows]))
    return out


def evaluate(
    model: TCNStructureTopologyModel,
    loader: DataLoader,
    device: torch.device,
    targets: list[str],
    objective: str,
    num_bins: int,
    mlm_probability: float,
    topology_weight: float,
    mlm_weight: float,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float | int]] = []
    total_loss_sum = 0.0
    with torch.no_grad():
        for raw_batch in loader:
            batch = batch_to_device(raw_batch, device)
            topology_loss, metrics = topology_loss_and_metrics(model, batch, targets, objective, num_bins)
            loss = topology_weight * topology_loss
            if mlm_weight > 0:
                masked_tokens, mlm_labels = mask_tokens_for_mlm(batch["tokens"], mlm_probability)
                logits = model.mlm_logits(masked_tokens)
                mlm_loss = nn.functional.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE),
                    mlm_labels.reshape(-1),
                    ignore_index=-100,
                )
                correct, total = masked_accuracy(logits, mlm_labels)
                metrics["mlm_loss"] = float(mlm_loss.detach().item())
                metrics["mlm_correct"] = correct
                metrics["mlm_total"] = total
                loss = loss + mlm_weight * mlm_loss
            metrics["total_loss"] = float(loss.detach().item())
            rows.append(metrics)
            total_loss_sum += float(loss.detach().item())
    aggregate = aggregate_epoch_metrics(rows, targets, objective)
    aggregate["total_loss"] = total_loss_sum / max(len(rows), 1)
    return aggregate


def selection_value(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name not in metrics:
        raise KeyError(f"Unknown selection metric {metric_name!r}; available: {sorted(metrics)}")
    value = float(metrics[metric_name])
    if metric_name.endswith("_loss") or metric_name == "total_loss":
        return -value
    return value


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    if not targets:
        raise ValueError("At least one target is required.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = read_residue_labels(args.residue_labels, targets)
    scalers = fit_target_scalers(df, targets, args.target_transform, args.num_bins)

    train_dataset = StructureTopologyDataset(
        df,
        "train",
        targets,
        scalers,
        args.max_len,
        args.objective,
        args.num_bins,
        args.max_train_proteins,
    )
    val_dataset = StructureTopologyDataset(
        df,
        "val",
        targets,
        scalers,
        args.max_len,
        args.objective,
        args.num_bins,
        args.max_val_proteins,
    )
    test_dataset = StructureTopologyDataset(
        df,
        "test",
        targets,
        scalers,
        args.max_len,
        args.objective,
        args.num_bins,
        args.max_test_proteins,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_config = model_config_from_args(args)
    model = TCNStructureTopologyModel(model_config, len(targets), args.num_bins, args.objective).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_selection = -float("inf")
    best_path = args.out_dir / "best_model.pt"
    epochs_without_improvement = 0
    stopped_early = False
    history: list[dict[str, float]] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        rows: list[dict[str, float | int]] = []
        for raw_batch in train_loader:
            batch = batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            topology_loss, metrics = topology_loss_and_metrics(model, batch, targets, args.objective, args.num_bins)
            loss = args.topology_weight * topology_loss
            if args.mlm_weight > 0:
                masked_tokens, mlm_labels = mask_tokens_for_mlm(batch["tokens"], args.mlm_probability)
                token_logits = model.mlm_logits(masked_tokens)
                mlm_loss = nn.functional.cross_entropy(
                    token_logits.reshape(-1, VOCAB_SIZE),
                    mlm_labels.reshape(-1),
                    ignore_index=-100,
                )
                correct, total = masked_accuracy(token_logits, mlm_labels)
                metrics["mlm_loss"] = float(mlm_loss.detach().item())
                metrics["mlm_correct"] = correct
                metrics["mlm_total"] = total
                loss = loss + args.mlm_weight * mlm_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            metrics["total_loss"] = float(loss.detach().item())
            rows.append(metrics)

        train_metrics = aggregate_epoch_metrics(rows, targets, args.objective)
        train_metrics["total_loss"] = float(np.mean([float(row["total_loss"]) for row in rows]))
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            targets,
            args.objective,
            args.num_bins,
            args.mlm_probability,
            args.topology_weight,
            args.mlm_weight,
        )
        current_selection = selection_value(val_metrics, args.selection_metric)
        row = {
            "epoch": float(epoch),
            "train_total_loss": train_metrics.get("total_loss", 0.0),
            "train_topology_loss_mean": train_metrics.get("topology_loss_mean", 0.0),
            "train_topology_accuracy_mean": train_metrics.get("topology_accuracy_mean", 0.0),
            "train_mlm_accuracy": train_metrics.get("mlm_accuracy", 0.0),
            "val_total_loss": val_metrics.get("total_loss", 0.0),
            "val_topology_loss_mean": val_metrics.get("topology_loss_mean", 0.0),
            "val_topology_accuracy_mean": val_metrics.get("topology_accuracy_mean", 0.0),
            "val_mlm_accuracy": val_metrics.get("mlm_accuracy", 0.0),
            "val_selected_metric": current_selection,
            "elapsed_sec": time.time() - started,
        }
        for target in targets:
            if args.objective == "bins":
                row[f"train_{target}_accuracy"] = train_metrics.get(f"{target}_accuracy", 0.0)
                row[f"val_{target}_accuracy"] = val_metrics.get(f"{target}_accuracy", 0.0)
        history.append(row)
        print(json.dumps(row), flush=True)

        if current_selection > best_selection + args.min_delta:
            best_selection = current_selection
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.encoder.state_dict(),
                    "structure_head_state_dict": model.topology_head.state_dict(),
                    "mlm_decoder_state_dict": model.mlm_decoder.state_dict(),
                    "architecture": "tcn",
                    "model_config": model_config,
                    "task_type": "tcn_residue_structure_topology",
                    "objective": args.objective,
                    "targets": targets,
                    "num_bins": args.num_bins,
                    "target_transform": args.target_transform,
                    "target_scalers": {target: scaler.to_dict() for target, scaler in scalers.items()},
                    "args": vars(args),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "selection_metric": args.selection_metric,
                    "val_selection_score": best_selection,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            print(
                json.dumps(
                    {
                        "early_stop": True,
                        "epoch": epoch,
                        "best_val_selection_score": best_selection,
                        "epochs_without_improvement": epochs_without_improvement,
                    }
                ),
                flush=True,
            )
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.encoder.load_state_dict(checkpoint["model_state_dict"])
    model.topology_head.load_state_dict(checkpoint["structure_head_state_dict"])
    model.mlm_decoder.load_state_dict(checkpoint["mlm_decoder_state_dict"])

    final_val_metrics = evaluate(
        model,
        val_loader,
        device,
        targets,
        args.objective,
        args.num_bins,
        args.mlm_probability,
        args.topology_weight,
        args.mlm_weight,
    )
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        targets,
        args.objective,
        args.num_bins,
        args.mlm_probability,
        args.topology_weight,
        args.mlm_weight,
    )

    history_path = args.out_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "residue_labels": str(args.residue_labels),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "architecture": "tcn",
        "model_config": model_config,
        "task_type": "tcn_residue_structure_topology",
        "objective": args.objective,
        "targets": targets,
        "target_transform": args.target_transform,
        "num_bins": args.num_bins,
        "target_scalers": {target: scaler.to_dict() for target, scaler in scalers.items()},
        "topology_weight": args.topology_weight,
        "mlm_weight": args.mlm_weight,
        "mlm_probability": args.mlm_probability,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "epochs": args.epochs,
        "epochs_completed": int(history[-1]["epoch"]) if history else 0,
        "stopped_early": stopped_early,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_selection_score": float(checkpoint["val_selection_score"]),
        "best_val_metrics": checkpoint["val_metrics"],
        "final_val_metrics": final_val_metrics,
        "test_metrics": test_metrics,
        "best_model_path": str(best_path),
        "history_path": str(history_path),
        "elapsed_sec": time.time() - started,
    }
    with (args.out_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--residue-labels", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--objective", choices=["bins", "regression"], default="bins")
    parser.add_argument("--target-transform", choices=["raw", "log1p"], default="log1p")
    parser.add_argument("--num-bins", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--tcn-layers", type=int, default=5)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--mlm-weight", type=float, default=0.3)
    parser.add_argument("--topology-weight", type=float, default=1.0)
    parser.add_argument(
        "--selection-metric",
        default="topology_accuracy_mean",
        choices=["topology_accuracy_mean", "topology_loss_mean", "total_loss", "mlm_accuracy", "mlm_loss"],
    )
    parser.add_argument("--max-train-proteins", type=int, default=None)
    parser.add_argument("--max-val-proteins", type=int, default=None)
    parser.add_argument("--max-test-proteins", type=int, default=None)
    args = parser.parse_args()
    if args.num_bins < 2:
        raise ValueError("--num-bins must be at least 2")
    train(args)


if __name__ == "__main__":
    main()
