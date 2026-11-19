#!/usr/bin/env python3
"""LoRA fine-tuning for ESM2 with residue-level structure topology labels.

The model input is protein sequence only.  ESMAtlas/AlphaFold-derived structure
labels are used as supervision during fine-tuning, then pooled ESM2
representations can be evaluated on protein-level downstream tasks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import esm
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_tcn_structure_topology import DEFAULT_TARGETS, TargetScaler, transform_values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_residue_labels(path: Path, targets: list[str]) -> pd.DataFrame:
    required = {"protein_id", "split", "full_sequence", "source_position", *targets}
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=sorted(required))
    else:
        df = pd.read_csv(path, usecols=lambda col: col in required)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df.copy()
    df["source_position"] = pd.to_numeric(df["source_position"], errors="coerce").astype("Int64")
    for target in targets:
        df[target] = pd.to_numeric(df[target], errors="coerce")
    return df.dropna(subset=["source_position", *targets])


def fit_target_scalers(df: pd.DataFrame, targets: list[str], transform: str, num_bins: int) -> dict[str, TargetScaler]:
    train_df = df[df["split"] == "train"]
    if train_df.empty:
        raise ValueError("No train rows found in residue label dataset.")
    return {
        target: TargetScaler.fit(target, train_df[target].to_numpy(dtype=np.float64), transform, num_bins)
        for target in targets
    }


def scaler_from_dict(data: dict[str, object]) -> TargetScaler:
    return TargetScaler(
        target=str(data["target"]),
        transform=str(data["transform"]),
        mean=float(data["mean"]),
        std=float(data["std"]),
        bin_edges=[float(value) for value in data["bin_edges"]],
    )


def iter_label_chunks(path: Path, columns: list[str], chunksize: int) -> Iterable[pd.DataFrame]:
    if path.suffix == ".parquet":
        yield pd.read_parquet(path, columns=columns)
        return
    yield from pd.read_csv(path, usecols=columns, chunksize=chunksize)


def fit_target_scalers_streaming(
    path: Path,
    targets: list[str],
    transform: str,
    num_bins: int,
    chunksize: int,
    progress_every: int,
) -> dict[str, TargetScaler]:
    values_by_target: dict[str, list[np.ndarray]] = {target: [] for target in targets}
    total_train_rows = 0
    columns = ["split", *targets]
    for chunk_idx, chunk in enumerate(iter_label_chunks(path, columns, chunksize), start=1):
        train_chunk = chunk[chunk["split"] == "train"]
        total_train_rows += len(train_chunk)
        for target in targets:
            values = pd.to_numeric(train_chunk[target], errors="coerce").to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                values_by_target[target].append(values)
        if progress_every > 0 and chunk_idx % progress_every == 0:
            print(
                json.dumps(
                    {
                        "cache_stage": "fit_target_scalers",
                        "chunk": chunk_idx,
                        "rows_seen": int(chunk_idx * chunksize),
                        "train_rows_seen": int(total_train_rows),
                    }
                ),
                flush=True,
            )
    if total_train_rows == 0:
        raise ValueError("No train rows found while streaming residue label dataset.")
    scalers = {}
    for target in targets:
        if not values_by_target[target]:
            raise ValueError(f"No finite train values for target {target}")
        scalers[target] = TargetScaler.fit(target, np.concatenate(values_by_target[target]), transform, num_bins)
    return scalers


def encode_esm_sequence(sequence: str, alphabet: esm.data.Alphabet, max_residues: int) -> torch.Tensor:
    sequence = sequence[:max_residues]
    tokens: list[int] = []
    if alphabet.prepend_bos:
        tokens.append(alphabet.cls_idx)
    tokens.extend(alphabet.get_idx(aa) for aa in sequence)
    if alphabet.append_eos:
        tokens.append(alphabet.eos_idx)
    max_tokens = max_residues + int(alphabet.prepend_bos) + int(alphabet.append_eos)
    if len(tokens) < max_tokens:
        tokens.extend([alphabet.padding_idx] * (max_tokens - len(tokens)))
    return torch.tensor(tokens[:max_tokens], dtype=torch.long)


def empty_example_record(
    protein_id: str,
    sequence: str,
    targets: list[str],
    alphabet: esm.data.Alphabet,
    max_residues: int,
) -> dict[str, object]:
    max_tokens = max_residues + int(alphabet.prepend_bos) + int(alphabet.append_eos)
    return {
        "protein_id": str(protein_id),
        "sequence": sequence[:max_residues],
        "tokens": encode_esm_sequence(sequence, alphabet, max_residues),
        "valid": np.zeros((max_tokens, len(targets)), dtype=bool),
        "regression": np.zeros((max_tokens, len(targets)), dtype=np.float32),
        "bins": np.full((max_tokens, len(targets)), -100, dtype=np.int64),
    }


def build_split_examples(
    df: pd.DataFrame,
    split: str,
    targets: list[str],
    scalers: dict[str, TargetScaler],
    alphabet: esm.data.Alphabet,
    max_residues: int,
    num_bins: int,
    max_proteins: int | None,
) -> list[dict[str, object]]:
    split_df = df[df["split"] == split]
    examples: list[dict[str, object]] = []
    max_tokens = max_residues + int(alphabet.prepend_bos) + int(alphabet.append_eos)
    residue_offset = int(alphabet.prepend_bos)

    for protein_id, group in split_df.groupby("protein_id", sort=True):
        if max_proteins is not None and len(examples) >= max_proteins:
            break
        sequence = str(group["full_sequence"].iloc[0]).strip().upper()
        if not sequence:
            continue
        tokens = encode_esm_sequence(sequence, alphabet, max_residues)
        valid = np.zeros((max_tokens, len(targets)), dtype=bool)
        regression = np.zeros((max_tokens, len(targets)), dtype=np.float32)
        bins = np.full((max_tokens, len(targets)), -100, dtype=np.int64)

        for _, row in group.iterrows():
            residue_pos = int(row["source_position"]) - 1
            if residue_pos < 0 or residue_pos >= max_residues:
                continue
            token_pos = residue_pos + residue_offset
            if token_pos < 0 or token_pos >= max_tokens:
                continue
            for target_idx, target in enumerate(targets):
                value = float(row[target])
                if not np.isfinite(value):
                    continue
                valid[token_pos, target_idx] = True
                transformed = transform_values(np.asarray([value], dtype=np.float64), scalers[target].transform)
                regression[token_pos, target_idx] = np.float32((transformed[0] - scalers[target].mean) / scalers[target].std)
                target_bins = np.digitize(transformed, scalers[target].bin_edges, right=True)
                bins[token_pos, target_idx] = int(np.clip(target_bins[0], 0, num_bins - 1))

        if not valid.any():
            continue
        examples.append(
            {
                "protein_id": str(protein_id),
                "sequence": sequence[:max_residues],
                "tokens": tokens,
                "valid": torch.tensor(valid, dtype=torch.bool),
                "regression": torch.tensor(regression, dtype=torch.float32),
                "bins": torch.tensor(bins, dtype=torch.long),
            }
        )
    if not examples:
        raise ValueError(f"No usable {split} examples after filtering.")
    return examples


def update_record_from_group(
    record: dict[str, object],
    group: pd.DataFrame,
    targets: list[str],
    scalers: dict[str, TargetScaler],
    max_residues: int,
    num_bins: int,
    residue_offset: int,
) -> None:
    positions = group["_source_position_int"].to_numpy(dtype=np.int64) - 1
    position_ok = (positions >= 0) & (positions < max_residues)
    if not position_ok.any():
        return
    token_positions = positions[position_ok] + residue_offset
    valid = record["valid"]
    regression = record["regression"]
    bins = record["bins"]
    assert isinstance(valid, np.ndarray)
    assert isinstance(regression, np.ndarray)
    assert isinstance(bins, np.ndarray)

    for target_idx, target in enumerate(targets):
        values = pd.to_numeric(group[target], errors="coerce").to_numpy(dtype=np.float64)[position_ok]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        target_positions = token_positions[finite]
        target_values = values[finite]
        transformed = transform_values(target_values, scalers[target].transform)
        valid[target_positions, target_idx] = True
        regression[target_positions, target_idx] = ((transformed - scalers[target].mean) / scalers[target].std).astype(
            np.float32
        )
        target_bins = np.digitize(transformed, scalers[target].bin_edges, right=True)
        bins[target_positions, target_idx] = np.clip(target_bins, 0, num_bins - 1).astype(np.int64)


def build_split_examples_streaming(
    path: Path,
    targets: list[str],
    scalers: dict[str, TargetScaler],
    alphabet: esm.data.Alphabet,
    max_residues: int,
    num_bins: int,
    max_proteins_by_split: dict[str, int | None],
    chunksize: int,
    progress_every: int,
) -> dict[str, list[dict[str, object]]]:
    residue_offset = int(alphabet.prepend_bos)
    records_by_split: dict[str, dict[str, dict[str, object]]] = {"train": {}, "val": {}, "test": {}}
    columns = ["protein_id", "split", "full_sequence", "source_position", *targets]
    rows_seen = 0
    for chunk_idx, chunk in enumerate(iter_label_chunks(path, columns, chunksize), start=1):
        rows_seen += len(chunk)
        chunk = chunk.copy()
        chunk["_source_position_int"] = pd.to_numeric(chunk["source_position"], errors="coerce")
        for target in targets:
            chunk[target] = pd.to_numeric(chunk[target], errors="coerce")
        chunk = chunk.dropna(subset=["protein_id", "split", "full_sequence", "_source_position_int", *targets])
        chunk["_source_position_int"] = chunk["_source_position_int"].astype(np.int64)

        for protein_id, group in chunk.groupby("protein_id", sort=False):
            split = str(group["split"].iloc[0]).strip()
            if split not in records_by_split:
                continue
            split_records = records_by_split[split]
            record = split_records.get(str(protein_id))
            if record is None:
                limit = max_proteins_by_split.get(split)
                if limit is not None and len(split_records) >= limit:
                    continue
                sequence = str(group["full_sequence"].iloc[0]).strip().upper()
                if not sequence:
                    continue
                record = empty_example_record(str(protein_id), sequence, targets, alphabet, max_residues)
                split_records[str(protein_id)] = record
            update_record_from_group(record, group, targets, scalers, max_residues, num_bins, residue_offset)

        if progress_every > 0 and chunk_idx % progress_every == 0:
            print(
                json.dumps(
                    {
                        "cache_stage": "build_examples",
                        "chunk": chunk_idx,
                        "rows_seen": int(rows_seen),
                        "proteins_seen": {
                            split: len(records_by_split[split])
                            for split in ("train", "val", "test")
                        },
                    }
                ),
                flush=True,
            )

    splits: dict[str, list[dict[str, object]]] = {}
    for split, records in records_by_split.items():
        examples = []
        for protein_id in sorted(records):
            record = records[protein_id]
            valid = record["valid"]
            assert isinstance(valid, np.ndarray)
            if not valid.any():
                continue
            examples.append(
                {
                    "protein_id": record["protein_id"],
                    "sequence": record["sequence"],
                    "tokens": record["tokens"],
                    "valid": torch.tensor(record["valid"], dtype=torch.bool),
                    "regression": torch.tensor(record["regression"], dtype=torch.float32),
                    "bins": torch.tensor(record["bins"], dtype=torch.long),
                }
            )
        if not examples:
            raise ValueError(f"No usable {split} examples after streaming cache build.")
        splits[split] = examples
    return splits


def dataset_cache_metadata(args: argparse.Namespace, targets: list[str], alphabet: esm.data.Alphabet) -> dict[str, object]:
    return {
        "residue_labels": str(args.residue_labels),
        "model_name": args.model_name,
        "alphabet_prepend_bos": bool(alphabet.prepend_bos),
        "alphabet_append_eos": bool(alphabet.append_eos),
        "max_residues": args.max_residues,
        "targets": targets,
        "target_transform": args.target_transform,
        "num_bins": args.num_bins,
        "max_train_proteins": args.max_train_proteins,
        "max_val_proteins": args.max_val_proteins,
        "max_test_proteins": args.max_test_proteins,
        "cache_builder": args.cache_builder,
    }


def metadata_is_compatible(
    cached: dict[str, object],
    requested: dict[str, object],
    allow_subset: bool,
) -> bool:
    strict_keys = [
        "residue_labels",
        "alphabet_prepend_bos",
        "alphabet_append_eos",
        "max_residues",
        "targets",
        "target_transform",
        "num_bins",
    ]
    for key in strict_keys:
        if cached.get(key) != requested.get(key):
            return False
    if not allow_subset:
        for key in ["max_train_proteins", "max_val_proteins", "max_test_proteins"]:
            if cached.get(key) != requested.get(key):
                return False
    else:
        for key in ["max_train_proteins", "max_val_proteins", "max_test_proteins"]:
            cached_limit = cached.get(key)
            requested_limit = requested.get(key)
            if requested_limit is None and cached_limit is not None:
                return False
            if cached_limit is not None and requested_limit is not None and int(cached_limit) < int(requested_limit):
                return False
    return True


def build_or_load_cached_examples(
    args: argparse.Namespace,
    targets: list[str],
    alphabet: esm.data.Alphabet,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, TargetScaler], dict[str, object]]:
    requested_metadata = dataset_cache_metadata(args, targets, alphabet)
    if args.dataset_cache is not None and args.dataset_cache.exists() and not args.rebuild_dataset_cache:
        cached = torch.load(args.dataset_cache, map_location="cpu")
        metadata = cached.get("metadata", {})
        if not metadata_is_compatible(metadata, requested_metadata, allow_subset=True):
            raise ValueError(
                f"Dataset cache metadata is incompatible with current args: {args.dataset_cache}\n"
                f"cached={metadata}\nrequested={requested_metadata}"
            )
        scalers = {target: scaler_from_dict(data) for target, data in cached["target_scalers"].items()}
        return cached["splits"], scalers, metadata

    if args.cache_builder == "streaming":
        scalers = fit_target_scalers_streaming(
            args.residue_labels,
            targets,
            args.target_transform,
            args.num_bins,
            args.cache_chunksize,
            args.cache_progress_every,
        )
        splits = build_split_examples_streaming(
            args.residue_labels,
            targets,
            scalers,
            alphabet,
            args.max_residues,
            args.num_bins,
            {
                "train": args.max_train_proteins,
                "val": args.max_val_proteins,
                "test": args.max_test_proteins,
            },
            args.cache_chunksize,
            args.cache_progress_every,
        )
    else:
        df = read_residue_labels(args.residue_labels, targets)
        scalers = fit_target_scalers(df, targets, args.target_transform, args.num_bins)
        splits = {
            split: build_split_examples(
                df,
                split,
                targets,
                scalers,
                alphabet,
                args.max_residues,
                args.num_bins,
                getattr(args, f"max_{split}_proteins"),
            )
            for split in ("train", "val", "test")
        }
    if args.dataset_cache is not None:
        args.dataset_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": requested_metadata,
                "target_scalers": {target: scaler.to_dict() for target, scaler in scalers.items()},
                "splits": splits,
            },
            args.dataset_cache,
        )
    return splits, scalers, requested_metadata


class CachedStructureTopologyDataset(Dataset):
    def __init__(self, examples: list[dict[str, object]], max_proteins: int | None = None):
        self.examples = examples if max_proteins is None else examples[:max_proteins]
        if not self.examples:
            raise ValueError("Empty cached dataset split.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.examples[idx]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def module_parent(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora_modules(
    model: nn.Module,
    target_suffixes: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
    layer_indices: Iterable[int] | None = None,
) -> list[str]:
    suffixes = tuple(s.strip() for s in target_suffixes if s.strip())
    selected_layers = None if layer_indices is None else {int(index) for index in layer_indices}
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(suffix) for suffix in suffixes):
            continue
        if selected_layers is not None:
            parts = name.split(".")
            layer_index = None
            for part_index, part in enumerate(parts[:-1]):
                if part == "layers" and parts[part_index + 1].isdigit():
                    layer_index = int(parts[part_index + 1])
                    break
            if layer_index not in selected_layers:
                continue
        parent, child_name = module_parent(model, name)
        setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout))
        replaced.append(name)
    if not replaced:
        raise ValueError(f"No Linear modules matched LoRA targets={suffixes}, layers={selected_layers}")
    return replaced


@dataclass
class LoRAConfig:
    rank: int
    alpha: float
    dropout: float
    targets: list[str]
    layers: list[int] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "targets": self.targets,
            "layers": self.layers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "LoRAConfig":
        return cls(
            rank=int(data["rank"]),
            alpha=float(data["alpha"]),
            dropout=float(data["dropout"]),
            targets=[str(value) for value in data["targets"]],
            layers=(None if data.get("layers") is None else [int(value) for value in data["layers"]]),
        )


def parse_lora_layers(value: str) -> list[int] | None:
    """Parse 'all' or comma-separated layer indices/ranges such as 22-29."""
    text = value.strip().lower()
    if text in {"", "all", "*"}:
        return None
    layers: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"Invalid LoRA layer range: {item!r}")
            layers.update(range(start, end + 1))
        else:
            index = int(item)
            if index < 0:
                raise ValueError(f"Invalid LoRA layer index: {item!r}")
            layers.add(index)
    if not layers:
        raise ValueError(f"No LoRA layers parsed from {value!r}")
    return sorted(layers)


class ESM2LoRAStructureModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_targets: int,
        num_bins: int,
        objective: str,
        lora_config: LoRAConfig,
        dropout: float,
        device: torch.device,
    ):
        super().__init__()
        base, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.alphabet = alphabet
        self.esm = base
        self.model_name = model_name
        self.num_targets = num_targets
        self.num_bins = num_bins
        self.objective = objective
        for param in self.esm.parameters():
            param.requires_grad = False
        self.lora_modules = inject_lora_modules(
            self.esm,
            lora_config.targets,
            lora_config.rank,
            lora_config.alpha,
            lora_config.dropout,
            lora_config.layers,
        )
        hidden_dim = int(self.esm.embed_dim)
        output_dim = num_targets * (num_bins if objective == "bins" else 1)
        self.topology_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.to(device)

    @property
    def embed_dim(self) -> int:
        return int(self.esm.embed_dim)

    @property
    def padding_idx(self) -> int:
        return int(self.alphabet.padding_idx)

    @property
    def cls_idx(self) -> int:
        return int(self.alphabet.cls_idx)

    @property
    def eos_idx(self) -> int:
        return int(self.alphabet.eos_idx)

    @property
    def mask_idx(self) -> int:
        return int(self.alphabet.mask_idx)

    def esm_forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.esm(tokens, repr_layers=[self.esm.num_layers], return_contacts=False)
        reps = outputs["representations"][self.esm.num_layers]
        return outputs["logits"], reps

    def topology_from_representations(self, representations: torch.Tensor) -> torch.Tensor:
        outputs = self.topology_head(representations)
        if self.objective == "bins":
            return outputs.view(
                representations.shape[0],
                representations.shape[1],
                self.num_targets,
                self.num_bins,
            )
        return outputs.view(representations.shape[0], representations.shape[1], self.num_targets)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: param.detach().cpu() for name, param in self.named_parameters() if param.requires_grad}


def batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        "tokens": batch["tokens"].to(device),
        "valid": batch["valid"].to(device),
        "regression": batch["regression"].to(device),
        "bins": batch["bins"].to(device),
        "protein_id": batch["protein_id"],
        "sequence": batch["sequence"],
    }


def mlm_candidate_mask(tokens: torch.Tensor, alphabet: esm.data.Alphabet) -> torch.Tensor:
    candidate = tokens.ne(alphabet.padding_idx)
    candidate &= tokens.ne(alphabet.cls_idx)
    candidate &= tokens.ne(alphabet.eos_idx)
    return candidate


def mask_tokens_for_mlm(
    tokens: torch.Tensor,
    alphabet: esm.data.Alphabet,
    mask_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate = mlm_candidate_mask(tokens, alphabet)
    mask = (torch.rand(tokens.shape, device=tokens.device) < mask_probability) & candidate
    no_mask = candidate.any(dim=1) & ~mask.any(dim=1)
    if no_mask.any():
        row_indices = no_mask.nonzero(as_tuple=False).squeeze(1)
        sampled = torch.multinomial(candidate[row_indices].float(), 1).squeeze(1)
        mask[row_indices, sampled] = True

    labels = torch.full_like(tokens, -100)
    labels[mask] = tokens[mask]
    masked_tokens = tokens.clone()
    replace_prob = torch.rand(tokens.shape, device=tokens.device)
    mask_replace = mask & replace_prob.lt(0.8)
    random_replace = mask & replace_prob.ge(0.8) & replace_prob.lt(0.9)
    masked_tokens[mask_replace] = alphabet.mask_idx
    if random_replace.any():
        standard = torch.tensor([alphabet.get_idx(aa) for aa in alphabet.standard_toks], device=tokens.device)
        sampled = standard[torch.randint(0, len(standard), (int(random_replace.sum().item()),), device=tokens.device)]
        masked_tokens[random_replace] = sampled
    return masked_tokens, labels


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    valid = labels.ne(-100)
    if not valid.any():
        return 0, 0
    pred = logits.argmax(dim=-1)
    return int((pred[valid] == labels[valid]).sum().item()), int(valid.sum().item())


def topology_loss_and_metrics(
    model: ESM2LoRAStructureModel,
    representations: torch.Tensor,
    batch: dict[str, object],
    target_names: list[str],
    objective: str,
    num_bins: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    outputs = model.topology_from_representations(representations)
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
                metrics[f"{target}_correct"] = 0
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
            return torch.tensor(0.0, device=representations.device), metrics
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
        return torch.tensor(0.0, device=representations.device), metrics
    metrics["topology_loss_mean"] = float(np.mean([metrics[f"{target}_loss"] for target in target_names]))
    return torch.stack(losses).mean(), metrics


def aggregate_epoch_metrics(rows: list[dict[str, float | int]], targets: list[str], objective: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not rows:
        return out
    if any("topology_loss_mean" in row for row in rows):
        out["topology_loss_mean"] = float(np.mean([float(row.get("topology_loss_mean", 0.0)) for row in rows]))
    if objective == "bins":
        accuracies = []
        for target in targets:
            correct = sum(int(row.get(f"{target}_correct", 0)) for row in rows)
            count = sum(int(row.get(f"{target}_count", 0)) for row in rows)
            if count:
                acc = correct / count
                out[f"{target}_accuracy"] = acc
                accuracies.append(acc)
        if accuracies:
            out["topology_accuracy_mean"] = float(np.mean(accuracies))
    mlm_correct = sum(int(row.get("mlm_correct", 0)) for row in rows)
    mlm_total = sum(int(row.get("mlm_total", 0)) for row in rows)
    if mlm_total:
        out["mlm_accuracy"] = mlm_correct / mlm_total
    if any("mlm_loss" in row for row in rows):
        out["mlm_loss"] = float(np.mean([float(row.get("mlm_loss", 0.0)) for row in rows]))
    return out


def forward_loss_and_metrics(
    model: ESM2LoRAStructureModel,
    batch: dict[str, object],
    targets: list[str],
    objective: str,
    num_bins: int,
    mlm_probability: float,
    topology_weight: float,
    mlm_weight: float,
    use_amp: bool,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    device_type = "cuda" if batch["tokens"].device.type == "cuda" else "cpu"
    metrics: dict[str, float | int] = {}
    labels = None
    input_tokens = batch["tokens"]
    if mlm_weight > 0:
        input_tokens, labels = mask_tokens_for_mlm(batch["tokens"], model.alphabet, mlm_probability)

    with torch.amp.autocast(device_type=device_type, enabled=use_amp):
        logits, representations = model.esm_forward(input_tokens)
        loss = torch.tensor(0.0, device=batch["tokens"].device)
        if topology_weight > 0:
            topology_loss, topology_metrics = topology_loss_and_metrics(
                model,
                representations,
                batch,
                targets,
                objective,
                num_bins,
            )
            metrics.update(topology_metrics)
            loss = loss + topology_weight * topology_loss
        if mlm_weight > 0 and labels is not None:
            mlm_loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss = loss + mlm_weight * mlm_loss
            metrics["mlm_loss"] = float(mlm_loss.detach().item())

    if mlm_weight > 0 and labels is not None:
        correct, total = masked_accuracy(logits.detach(), labels)
        metrics["mlm_correct"] = correct
        metrics["mlm_total"] = total
    metrics["total_loss"] = float(loss.detach().item())
    return loss, metrics


def evaluate(
    model: ESM2LoRAStructureModel,
    loader: DataLoader,
    device: torch.device,
    targets: list[str],
    objective: str,
    num_bins: int,
    mlm_probability: float,
    topology_weight: float,
    mlm_weight: float,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float | int]] = []
    total_loss = 0.0
    with torch.no_grad():
        for raw_batch in loader:
            batch = batch_to_device(raw_batch, device)
            loss, metrics = forward_loss_and_metrics(
                model,
                batch,
                targets,
                objective,
                num_bins,
                mlm_probability,
                topology_weight,
                mlm_weight,
                use_amp,
            )
            rows.append(metrics)
            total_loss += float(loss.detach().item())
    aggregate = aggregate_epoch_metrics(rows, targets, objective)
    aggregate["total_loss"] = total_loss / max(len(rows), 1)
    return aggregate


def selection_value(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name not in metrics:
        raise KeyError(f"Unknown selection metric {metric_name!r}; available: {sorted(metrics)}")
    value = float(metrics[metric_name])
    if metric_name.endswith("_loss") or metric_name == "total_loss":
        return -value
    return value


def save_checkpoint(
    path: Path,
    model: ESM2LoRAStructureModel,
    args: argparse.Namespace,
    lora_config: LoRAConfig,
    targets: list[str],
    scalers: dict[str, TargetScaler],
    epoch: int,
    val_metrics: dict[str, float],
    best_selection: float,
) -> None:
    torch.save(
        {
            "task_type": "esm2_lora_residue_structure_topology",
            "model_name": args.model_name,
            "architecture": "esm2_lora",
            "esm_num_layers": int(model.esm.num_layers),
            "esm_embed_dim": int(model.esm.embed_dim),
            "lora_config": lora_config.to_dict(),
            "lora_modules": model.lora_modules,
            "trainable_state_dict": model.trainable_state_dict(),
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
        path,
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    if not targets:
        raise ValueError("At least one target is required.")
    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        targets=[target.strip() for target in args.lora_targets.split(",") if target.strip()],
        layers=parse_lora_layers(args.lora_layers),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_for_alphabet, alphabet = esm.pretrained.load_model_and_alphabet(args.model_name)
    del base_for_alphabet
    splits, scalers, cache_metadata = build_or_load_cached_examples(args, targets, alphabet)
    train_dataset = CachedStructureTopologyDataset(splits["train"], args.max_train_proteins)
    val_dataset = CachedStructureTopologyDataset(splits["val"], args.max_val_proteins)
    test_dataset = CachedStructureTopologyDataset(splits["test"], args.max_test_proteins)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ESM2LoRAStructureModel(
        args.model_name,
        len(targets),
        args.num_bins,
        args.objective,
        lora_config,
        args.dropout,
        device,
    )
    total_params, trainable_params = count_parameters(model)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_selection = -float("inf")
    best_path = args.out_dir / "best_model.pt"
    epochs_without_improvement = 0
    stopped_early = False
    history: list[dict[str, float]] = []
    started = time.time()

    run_metadata = {
        "model_name": args.model_name,
        "lora_config": lora_config.to_dict(),
        "lora_modules": model.lora_modules,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_fraction": trainable_params / max(total_params, 1),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "cache_metadata": cache_metadata,
    }
    with (args.out_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"run_metadata": run_metadata}, indent=2), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        rows: list[dict[str, float | int]] = []
        for raw_batch in train_loader:
            batch = batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = forward_loss_and_metrics(
                model,
                batch,
                targets,
                args.objective,
                args.num_bins,
                args.mlm_probability,
                args.topology_weight,
                args.mlm_weight,
                args.amp and device.type == "cuda",
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
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
            args.amp and device.type == "cuda",
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
            save_checkpoint(best_path, model, args, lora_config, targets, scalers, epoch, val_metrics, best_selection)
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
    load_trainable_state(model, checkpoint["trainable_state_dict"])
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
        args.amp and device.type == "cuda",
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
        args.amp and device.type == "cuda",
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
        "architecture": "esm2_lora",
        "model_name": args.model_name,
        "lora_config": lora_config.to_dict(),
        "lora_modules": model.lora_modules,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_fraction": trainable_params / max(total_params, 1),
        "task_type": "esm2_lora_residue_structure_topology",
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


def load_trainable_state(model: nn.Module, trainable_state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    missing = []
    for name, tensor in trainable_state.items():
        if name not in current:
            missing.append(name)
            continue
        current[name].copy_(tensor.to(current[name].device, dtype=current[name].dtype))
    if missing:
        raise KeyError(f"Checkpoint contains unknown trainable parameters: {missing[:10]}")
    model.load_state_dict(current, strict=False)


def load_esm2_lora_for_inference(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ESM2LoRAStructureModel, esm.data.Alphabet]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    lora_config = LoRAConfig.from_dict(checkpoint["lora_config"])
    model = ESM2LoRAStructureModel(
        str(checkpoint["model_name"]),
        len(checkpoint["targets"]),
        int(checkpoint["num_bins"]),
        str(checkpoint["objective"]),
        lora_config,
        float(checkpoint.get("args", {}).get("dropout", 0.1)),
        device,
    )
    load_trainable_state(model, checkpoint["trainable_state_dict"])
    model.eval()
    return model, model.alphabet


def load_esm2_pretrained_for_inference(
    model_name: str,
    device: torch.device,
) -> tuple[nn.Module, esm.data.Alphabet]:
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.to(device)
    model.eval()
    return model, alphabet


def load_esm2_full_state_for_inference(
    checkpoint_path: Path,
    model_name: str,
    device: torch.device,
) -> tuple[nn.Module, esm.data.Alphabet]:
    """Load a full TorchDrug-style ESM state dict into the matching fair-esm model."""
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    raw_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(raw_state, dict):
        raise TypeError(f"Expected a state-dict mapping in {checkpoint_path}, got {type(raw_state)!r}")

    model_state: dict[str, torch.Tensor] = {}
    ignored = {"mapping"}
    unknown: list[str] = []
    for name, tensor in raw_state.items():
        if name in ignored:
            continue
        if name.startswith("model."):
            model_state[name.removeprefix("model.")] = tensor
        elif name in model.state_dict():
            model_state[name] = tensor
        else:
            unknown.append(name)
    if unknown:
        raise KeyError(f"Unrecognized full-state keys in {checkpoint_path}: {unknown[:10]}")

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Full-state checkpoint does not match {model_name}: "
            f"missing={list(missing)[:10]}, unexpected={list(unexpected)[:10]}"
        )
    model.to(device)
    model.eval()
    return model, alphabet


def esm2_representations(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    if isinstance(model, ESM2LoRAStructureModel):
        _, reps = model.esm_forward(tokens)
        return reps
    outputs = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
    return outputs["representations"][model.num_layers]


def esm2_sequence_embeddings(
    model: nn.Module,
    alphabet: esm.data.Alphabet,
    sequences: list[str],
    max_residues: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool = True,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    residue_offset = int(alphabet.prepend_bos)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            tokens = torch.stack([encode_esm_sequence(seq, alphabet, max_residues) for seq in batch_sequences]).to(device)
            with torch.amp.autocast(device_type=device_type, enabled=use_amp and device.type == "cuda"):
                reps = esm2_representations(model, tokens)
            valid = tokens.ne(alphabet.padding_idx)
            valid[:, :residue_offset] = False
            if alphabet.append_eos:
                valid &= tokens.ne(alphabet.eos_idx)
            pooled = []
            for row_idx in range(tokens.shape[0]):
                row_valid = valid[row_idx]
                if not row_valid.any():
                    pooled.append(torch.zeros(reps.shape[-1], device=device, dtype=reps.dtype))
                else:
                    pooled.append(reps[row_idx, row_valid].mean(dim=0))
            outputs.append(torch.stack(pooled).float().cpu().numpy())
    return np.vstack(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--residue-labels", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dataset-cache", type=Path, default=None)
    parser.add_argument("--rebuild-dataset-cache", action="store_true")
    parser.add_argument("--cache-builder", choices=["streaming", "pandas"], default="streaming")
    parser.add_argument("--cache-chunksize", type=int, default=500000)
    parser.add_argument("--cache-progress-every", type=int, default=5)
    parser.add_argument("--model-name", default="esm2_t6_8M_UR50D")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--objective", choices=["bins", "regression"], default="bins")
    parser.add_argument("--target-transform", choices=["raw", "log1p"], default="log1p")
    parser.add_argument("--num-bins", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-residues", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--mlm-weight", type=float, default=0.3)
    parser.add_argument("--topology-weight", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,v_proj")
    parser.add_argument(
        "--lora-layers",
        default="all",
        help="Transformer layers receiving LoRA: all, a range (22-29), or comma-separated indices/ranges.",
    )
    parser.add_argument("--amp", action="store_true")
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
    if args.lora_rank < 1:
        raise ValueError("--lora-rank must be at least 1")
    train(args)


if __name__ == "__main__":
    main()
