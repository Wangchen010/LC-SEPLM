#!/usr/bin/env python3
"""Train a small sequence encoder on betweenness pairwise ranking."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
PAD_IDX = 0
UNK_IDX = 1
AA_TO_IDX = {aa: idx + 2 for idx, aa in enumerate(AA_ALPHABET)}
VOCAB_SIZE = len(AA_TO_IDX) + 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_sequence(sequence: str, max_len: int) -> list[int]:
    tokens = [AA_TO_IDX.get(aa, UNK_IDX) for aa in sequence[:max_len]]
    if len(tokens) < max_len:
        tokens.extend([PAD_IDX] * (max_len - len(tokens)))
    return tokens


class PairDataset(Dataset):
    def __init__(self, path: Path, max_len: int, max_pairs: int | None = None):
        self.rows: list[dict[str, object]] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            for idx, row in enumerate(csv.DictReader(handle)):
                if max_pairs is not None and idx >= max_pairs:
                    break
                self.rows.append(
                    {
                        "pair_id": row["pair_id"],
                        "record_id_a": row["record_id_a"],
                        "record_id_b": row["record_id_b"],
                        "sequence_a": row["sequence_a"],
                        "sequence_b": row["sequence_b"],
                        "label": float(row["label_a_higher"]),
                        "score_a": float(row["score_a"]),
                        "score_b": float(row["score_b"]),
                    }
                )
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.rows[idx]
        seq_a = torch.tensor(encode_sequence(str(row["sequence_a"]), self.max_len), dtype=torch.long)
        seq_b = torch.tensor(encode_sequence(str(row["sequence_b"]), self.max_len), dtype=torch.long)
        return {
            "seq_a": seq_a,
            "seq_b": seq_b,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "pair_id": row["pair_id"],
            "record_id_a": row["record_id_a"],
            "record_id_b": row["record_id_b"],
            "score_a": row["score_a"],
            "score_b": row["score_b"],
        }


class CNNSequenceRanker(nn.Module):
    def __init__(self, embed_dim: int = 64, channels: int = 128, dropout: float = 0.15):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_IDX)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, channels, kernel_size=3, padding=1),
                nn.Conv1d(embed_dim, channels, kernel_size=5, padding=2),
                nn.Conv1d(embed_dim, channels, kernel_size=9, padding=4),
            ]
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(channels * len(self.convs), channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(PAD_IDX).unsqueeze(1)
        emb = self.embedding(tokens).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            hidden = self.activation(conv(emb))
            hidden = hidden.masked_fill(~mask, 0.0)
            denom = mask.sum(dim=2).clamp(min=1)
            pooled.append(hidden.sum(dim=2) / denom)
        return torch.cat(pooled, dim=1)

    def score(self, tokens: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(tokens)
        return self.head(self.dropout(encoded)).squeeze(-1)

    def forward(self, seq_a: torch.Tensor, seq_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score_a = self.score(seq_a)
        score_b = self.score(seq_b)
        logits = score_a - score_b
        return logits, score_a, score_b


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class TCNSequenceRanker(nn.Module):
    """Attention-free sequence ranker built from dilated 1D convolutions."""

    def __init__(
        self,
        embed_dim: int = 64,
        channels: int = 128,
        num_layers: int = 5,
        kernel_size: int = 5,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_IDX)
        self.input_projection = nn.Conv1d(embed_dim, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                TCNBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=2**layer_idx,
                    dropout=dropout,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(PAD_IDX).unsqueeze(1)
        hidden = self.input_projection(self.embedding(tokens).transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
            hidden = hidden.masked_fill(~mask, 0.0)

        denom = mask.sum(dim=2).clamp(min=1)
        mean_pool = hidden.sum(dim=2) / denom
        max_hidden = hidden.masked_fill(~mask, torch.finfo(hidden.dtype).min)
        max_pool = max_hidden.max(dim=2).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return torch.cat([mean_pool, max_pool], dim=1)

    def score(self, tokens: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(tokens)
        return self.head(self.dropout(encoded)).squeeze(-1)

    def forward(self, seq_a: torch.Tensor, seq_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score_a = self.score(seq_a)
        score_b = self.score(seq_b)
        logits = score_a - score_b
        return logits, score_a, score_b


def model_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {
        "architecture": args.architecture,
        "embed_dim": args.embed_dim,
        "channels": args.channels,
        "dropout": args.dropout,
    }
    if args.architecture == "tcn":
        config.update({"num_layers": args.tcn_layers, "kernel_size": args.kernel_size})
    return config


def build_model(config: dict[str, object]) -> nn.Module:
    architecture = str(config.get("architecture", "cnn"))
    if architecture == "cnn":
        return CNNSequenceRanker(
            embed_dim=int(config.get("embed_dim", 64)),
            channels=int(config.get("channels", 128)),
            dropout=float(config.get("dropout", 0.15)),
        )
    if architecture == "tcn":
        return TCNSequenceRanker(
            embed_dim=int(config.get("embed_dim", 64)),
            channels=int(config.get("channels", 128)),
            num_layers=int(config.get("num_layers", 5)),
            kernel_size=int(config.get("kernel_size", 5)),
            dropout=float(config.get("dropout", 0.15)),
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def batch_to_device(batch: dict[str, object], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["seq_a"].to(device),
        batch["seq_b"].to(device),
        batch["label"].to(device),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_prediction_rows: int = 0,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total = 0
    correct = 0
    predictions: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            seq_a, seq_b, label = batch_to_device(batch, device)
            logits, score_a, score_b = model(seq_a, seq_b)
            loss = loss_fn(logits, label)
            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).float()
            total_loss += float(loss.item())
            total += int(label.numel())
            correct += int((pred == label).sum().item())

            if max_prediction_rows and len(predictions) < max_prediction_rows:
                room = max_prediction_rows - len(predictions)
                for i in range(min(room, len(label))):
                    predictions.append(
                        {
                            "pair_id": batch["pair_id"][i],
                            "record_id_a": batch["record_id_a"][i],
                            "record_id_b": batch["record_id_b"][i],
                            "true_label_a_higher": int(label[i].cpu().item()),
                            "pred_prob_a_higher": float(probs[i].cpu().item()),
                            "pred_label_a_higher": int(pred[i].cpu().item()),
                            "model_score_a": float(score_a[i].cpu().item()),
                            "model_score_b": float(score_b[i].cpu().item()),
                            "true_betweenness_a": float(batch["score_a"][i]),
                            "true_betweenness_b": float(batch["score_b"][i]),
                        }
                    )
    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}, predictions


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = PairDataset(args.train_pairs, args.max_len, args.max_train_pairs)
    val_dataset = PairDataset(args.val_pairs, args.max_len, args.max_val_pairs)
    test_dataset = PairDataset(args.test_pairs, args.max_len, args.max_test_pairs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_config = model_config_from_args(args)
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    epochs_without_improvement = 0
    stopped_early = False
    best_path = args.out_dir / "best_model.pt"
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for batch in train_loader:
            seq_a, seq_b, label = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(seq_a, seq_b)
            loss = loss_fn(logits, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred = (probs >= 0.5).float()
                total_loss += float(loss.item()) * int(label.numel())
                total += int(label.numel())
                correct += int((pred == label).sum().item())

        train_metrics = {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}
        val_metrics, _ = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "elapsed_sec": time.time() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        improved = val_metrics["accuracy"] > best_val_acc + args.min_delta
        if improved:
            best_val_acc = val_metrics["accuracy"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": args.architecture,
                    "model_config": model_config,
                    "args": vars(args),
                    "epoch": epoch,
                    "val_accuracy": best_val_acc,
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
                        "best_val_accuracy": best_val_acc,
                        "epochs_without_improvement": epochs_without_improvement,
                    }
                ),
                flush=True,
            )
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, _ = evaluate(model, val_loader, device)
    test_metrics, predictions = evaluate(
        model,
        test_loader,
        device,
        max_prediction_rows=args.max_prediction_rows,
    )

    history_path = args.out_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    predictions_path = args.out_dir / "test_predictions_head.csv"
    if predictions:
        with predictions_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(predictions[0].keys()))
            writer.writeheader()
            writer.writerows(predictions)

    summary = {
        "train_pairs": str(args.train_pairs),
        "val_pairs": str(args.val_pairs),
        "test_pairs": str(args.test_pairs),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "epochs": args.epochs,
        "epochs_completed": int(history[-1]["epoch"]) if history else 0,
        "stopped_early": stopped_early,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "architecture": args.architecture,
        "model_config": model_config,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_accuracy_during_training": float(checkpoint["val_accuracy"]),
        "final_val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "elapsed_sec": time.time() - started,
        "best_model_path": str(best_path),
        "history_path": str(history_path),
        "predictions_head_path": str(predictions_path) if predictions else None,
    }
    with (args.out_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pairs", required=True, type=Path)
    parser.add_argument("--val-pairs", required=True, type=Path)
    parser.add_argument("--test-pairs", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--architecture", choices=["cnn", "tcn"], default="cnn")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--tcn-layers", type=int, default=5)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=None)
    parser.add_argument("--max-test-pairs", type=int, default=None)
    parser.add_argument("--max-prediction-rows", type=int, default=500)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
