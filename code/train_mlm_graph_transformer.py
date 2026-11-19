#!/usr/bin/env python3
"""Train a small protein Transformer with MLM and optional graph-signal ranking."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
PAD_IDX = 0
MASK_IDX = 1
UNK_IDX = 2
AA_TO_IDX = {aa: idx + 3 for idx, aa in enumerate(AA_ALPHABET)}
IDX_TO_AA = {idx: aa for aa, idx in AA_TO_IDX.items()}
VOCAB_SIZE = len(AA_TO_IDX) + 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(chunks)))
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        records.append((current_id, "".join(chunks)))
    return records


def encode_sequence(sequence: str, max_len: int) -> list[int]:
    tokens = [AA_TO_IDX.get(aa, UNK_IDX) for aa in sequence.strip().upper()[:max_len]]
    if len(tokens) < max_len:
        tokens.extend([PAD_IDX] * (max_len - len(tokens)))
    return tokens


class SequenceDataset(Dataset):
    def __init__(self, fasta_path: Path, max_len: int, max_sequences: int | None = None):
        records = read_fasta(fasta_path)
        if max_sequences is not None:
            records = records[:max_sequences]
        self.records = records
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(encode_sequence(self.records[idx][1], self.max_len), dtype=torch.long)


class PairDataset(Dataset):
    def __init__(self, path: Path, max_len: int, max_pairs: int | None = None):
        self.rows: list[dict[str, object]] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            for idx, row in enumerate(csv.DictReader(handle)):
                if max_pairs is not None and idx >= max_pairs:
                    break
                self.rows.append(
                    {
                        "sequence_a": row["sequence_a"],
                        "sequence_b": row["sequence_b"],
                        "label": float(row["label_a_higher"]),
                    }
                )
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        return {
            "seq_a": torch.tensor(encode_sequence(str(row["sequence_a"]), self.max_len), dtype=torch.long),
            "seq_b": torch.tensor(encode_sequence(str(row["sequence_b"]), self.max_len), dtype=torch.long),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
        }


class ProteinTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_len: int = 256,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.position = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.rank_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def contextualize(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.embedding(tokens) + self.position(positions)
        key_padding_mask = tokens.eq(PAD_IDX)
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        return self.norm(hidden)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.contextualize(tokens)
        mask = tokens.ne(PAD_IDX).unsqueeze(-1)
        hidden = hidden.masked_fill(~mask, 0.0)
        denom = mask.sum(dim=1).clamp(min=1)
        return hidden.sum(dim=1) / denom

    def mlm_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.mlm_head(self.contextualize(tokens))

    def rank_score(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.rank_head(self.encode(tokens)).squeeze(-1)


def mask_tokens(tokens: torch.Tensor, mlm_probability: float) -> tuple[torch.Tensor, torch.Tensor]:
    labels = tokens.clone()
    can_mask = tokens.ne(PAD_IDX)
    probability = torch.full(tokens.shape, mlm_probability, device=tokens.device)
    masked = torch.bernoulli(probability).bool() & can_mask

    for row_idx in range(tokens.shape[0]):
        if not masked[row_idx].any() and can_mask[row_idx].any():
            choices = torch.nonzero(can_mask[row_idx], as_tuple=False).flatten()
            masked[row_idx, choices[torch.randint(len(choices), (1,), device=tokens.device)]] = True

    labels[~masked] = -100
    inputs = tokens.clone()

    replace_with_mask = torch.bernoulli(torch.full(tokens.shape, 0.8, device=tokens.device)).bool() & masked
    inputs[replace_with_mask] = MASK_IDX

    replace_with_random = (
        torch.bernoulli(torch.full(tokens.shape, 0.5, device=tokens.device)).bool()
        & masked
        & ~replace_with_mask
    )
    random_tokens = torch.randint(3, VOCAB_SIZE, tokens.shape, dtype=torch.long, device=tokens.device)
    inputs[replace_with_random] = random_tokens[replace_with_random]

    return inputs, labels


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def evaluate_pair_accuracy(model: ProteinTransformer, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            seq_a = batch["seq_a"].to(device)
            seq_b = batch["seq_b"].to(device)
            label = batch["label"].to(device)
            logits = model.rank_score(seq_a) - model.rank_score(seq_b)
            loss = loss_fn(logits, label)
            pred = (torch.sigmoid(logits) >= 0.5).float()
            total_loss += float(loss.item())
            total += int(label.numel())
            correct += int((pred == label).sum().item())
    return {"rank_loss": total_loss / max(total, 1), "rank_accuracy": correct / max(total, 1)}


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seq_dataset = SequenceDataset(args.fasta, args.max_len, args.max_sequences)
    seq_loader = DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    pair_loader = None
    val_pair_loader = None
    if args.train_pairs is not None:
        pair_dataset = PairDataset(args.train_pairs, args.max_len, args.max_train_pairs)
        pair_loader = DataLoader(pair_dataset, batch_size=args.pair_batch_size, shuffle=True, num_workers=0, drop_last=True)
        if args.val_pairs is not None:
            val_pair_dataset = PairDataset(args.val_pairs, args.max_len, args.max_val_pairs)
            val_pair_loader = DataLoader(val_pair_dataset, batch_size=args.pair_batch_size, shuffle=False, num_workers=0)

    model = ProteinTransformer(
        max_len=args.max_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    rank_loss_fn = nn.BCEWithLogitsLoss()

    seq_iter = iter(seq_loader)
    pair_iter = iter(pair_loader) if pair_loader is not None else None
    history: list[dict[str, float]] = []
    started = time.time()
    best_metric = -1.0
    best_path = args.out_dir / "best_model.pt"

    for step in range(1, args.max_steps + 1):
        model.train()
        seq_batch, seq_iter = next_batch(seq_iter, seq_loader)
        seq_batch = seq_batch.to(device)
        mlm_inputs, mlm_labels = mask_tokens(seq_batch, args.mlm_probability)

        optimizer.zero_grad(set_to_none=True)
        mlm_logits = model.mlm_logits(mlm_inputs)
        mlm_loss = mlm_loss_fn(mlm_logits.view(-1, VOCAB_SIZE), mlm_labels.view(-1))
        total_loss = mlm_loss
        rank_loss_value = 0.0
        rank_accuracy_value = 0.0

        if pair_loader is not None and pair_iter is not None and args.graph_loss_weight > 0.0:
            pair_batch, pair_iter = next_batch(pair_iter, pair_loader)
            seq_a = pair_batch["seq_a"].to(device)
            seq_b = pair_batch["seq_b"].to(device)
            label = pair_batch["label"].to(device)
            logits = model.rank_score(seq_a) - model.rank_score(seq_b)
            rank_loss = rank_loss_fn(logits, label)
            total_loss = total_loss + args.graph_loss_weight * rank_loss
            with torch.no_grad():
                pred = (torch.sigmoid(logits) >= 0.5).float()
                rank_accuracy_value = float((pred == label).float().mean().item())
            rank_loss_value = float(rank_loss.item())

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            row = {
                "step": float(step),
                "loss": float(total_loss.item()),
                "mlm_loss": float(mlm_loss.item()),
                "rank_loss": rank_loss_value,
                "rank_accuracy": rank_accuracy_value,
                "elapsed_sec": time.time() - started,
            }
            if val_pair_loader is not None:
                val_metrics = evaluate_pair_accuracy(model, val_pair_loader, device)
                row["val_rank_loss"] = val_metrics["rank_loss"]
                row["val_rank_accuracy"] = val_metrics["rank_accuracy"]
                metric = val_metrics["rank_accuracy"]
            else:
                metric = -float(mlm_loss.item())
            history.append(row)
            print(json.dumps(row), flush=True)

            if metric > best_metric:
                best_metric = metric
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": {
                            "max_len": args.max_len,
                            "d_model": args.d_model,
                            "n_heads": args.n_heads,
                            "n_layers": args.n_layers,
                            "dim_feedforward": args.dim_feedforward,
                            "dropout": args.dropout,
                        },
                        "training_args": vars(args),
                        "step": step,
                        "best_metric": best_metric,
                    },
                    best_path,
                )

    if not best_path.exists():
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "max_len": args.max_len,
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "dim_feedforward": args.dim_feedforward,
                    "dropout": args.dropout,
                },
                "training_args": vars(args),
                "step": args.max_steps,
                "best_metric": best_metric,
            },
            best_path,
        )

    final_path = args.out_dir / "final_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "max_len": args.max_len,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
            },
            "training_args": vars(args),
            "step": args.max_steps,
        },
        final_path,
    )

    history_path = args.out_dir / "training_history.csv"
    if history:
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    summary = {
        "out_dir": str(args.out_dir),
        "checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "signal_name": args.signal_name,
        "num_sequences": len(seq_dataset),
        "has_graph_loss": args.train_pairs is not None and args.graph_loss_weight > 0.0,
        "history_path": str(history_path),
        "last_history": history[-1] if history else None,
    }
    with (args.out_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--signal-name", default="mlm_only")
    parser.add_argument("--train-pairs", type=Path, default=None)
    parser.add_argument("--val-pairs", type=Path, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-train-pairs", type=int, default=100000)
    parser.add_argument("--max-val-pairs", type=int, default=20000)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--graph-loss-weight", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
