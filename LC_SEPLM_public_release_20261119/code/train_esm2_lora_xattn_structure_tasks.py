#!/usr/bin/env python3
"""ESM2 fine-tuning pilots with cross-attention structure supervision.

Two sequence-only continued-training tasks are supported:

1. topology_xattn: residue-level topology bins decoded through cross-attention.
2. lr_contact_xattn: long-range contact labels decoded by pair queries that
   cross-attend to the full sequence representation.

The structure heads are used only during continued training.  ``--finetune-mode
lora`` preserves the original adapter-only workflow; ``--finetune-mode full``
updates and saves the complete ESM2 state.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import esm
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_esm2_lora_structure_topology import (
    CachedStructureTopologyDataset,
    LoRAConfig,
    build_or_load_cached_examples,
    encode_esm_sequence,
    inject_lora_modules,
    mask_tokens_for_mlm,
    masked_accuracy,
    parse_lora_layers,
)
from train_tcn_structure_topology import DEFAULT_TARGETS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_ca_coordinates(pdb_path: Path, chain: str | None = None) -> dict[tuple[str, int, str], np.ndarray]:
    coords: dict[tuple[str, int, str], np.ndarray] = {}
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            line_chain = line[21].strip() or "_"
            if chain and chain != line_chain:
                continue
            try:
                resseq = int(line[22:26])
                icode = line[26].strip()
                xyz = np.asarray(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float32,
                )
            except ValueError:
                continue
            coords[(line_chain, resseq, icode)] = xyz
    return coords


def parse_ca_coordinates_from_text(text: str, chain: str | None = None) -> dict[tuple[str, int, str], np.ndarray]:
    coords: dict[tuple[str, int, str], np.ndarray] = {}
    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        line_chain = line[21].strip() or "_"
        if chain and chain != line_chain:
            continue
        try:
            resseq = int(line[22:26])
            icode = line[26].strip()
            xyz = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float32,
            )
        except ValueError:
            continue
        coords[(line_chain, resseq, icode)] = xyz
    return coords


def open_manifest(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return path.open("r", encoding="utf-8", errors="replace", newline="")


def lora_only_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if ".lora_a." in name or ".lora_b." in name
    }


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}


class ESM2LoRAXAttnModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        lora_config: LoRAConfig,
        dropout: float,
        num_targets: int,
        num_bins: int,
        num_heads: int,
        device: torch.device,
        pair_query_mode: str = "full",
        finetune_mode: str = "lora",
    ):
        super().__init__()
        if pair_query_mode not in {"full", "concat"}:
            raise ValueError(f"Unsupported pair_query_mode: {pair_query_mode}")
        if finetune_mode not in {"lora", "full"}:
            raise ValueError(f"Unsupported finetune_mode: {finetune_mode}")
        base, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.esm = base
        self.alphabet = alphabet
        self.model_name = model_name
        self.pair_query_mode = pair_query_mode
        self.finetune_mode = finetune_mode
        if finetune_mode == "full":
            for param in self.esm.parameters():
                param.requires_grad = True
            lora_targets = []
        else:
            for param in self.esm.parameters():
                param.requires_grad = False
            lora_targets = [target for target in lora_config.targets if target.strip()]
        if lora_targets:
            self.lora_modules = inject_lora_modules(
                self.esm,
                lora_targets,
                lora_config.rank,
                lora_config.alpha,
                lora_config.dropout,
                lora_config.layers,
            )
        else:
            self.lora_modules = []
        hidden = int(self.esm.embed_dim)
        self.num_targets = num_targets
        self.num_bins = num_bins
        self.topology_attn = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.topology_norm = nn.LayerNorm(hidden)
        self.topology_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_targets * num_bins),
        )
        pair_query_input_dim = hidden * 4 + 1 if pair_query_mode == "full" else hidden * 2
        self.pair_query = nn.Sequential(
            nn.Linear(pair_query_input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.pair_attn = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.pair_norm = nn.LayerNorm(hidden)
        self.contact_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.to(device)

    @property
    def padding_idx(self) -> int:
        return int(self.alphabet.padding_idx)

    @property
    def eos_idx(self) -> int:
        return int(self.alphabet.eos_idx)

    def esm_forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.esm(tokens, repr_layers=[self.esm.num_layers], return_contacts=False)
        return outputs["logits"], outputs["representations"][self.esm.num_layers]

    def residue_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(self.alphabet.padding_idx)
        if self.alphabet.prepend_bos:
            mask[:, 0] = False
        if self.alphabet.append_eos:
            mask &= tokens.ne(self.alphabet.eos_idx)
        return mask

    def topology_logits(self, reps: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        decoded, _ = self.topology_attn(
            query=reps,
            key=reps,
            value=reps,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = self.topology_norm(reps + decoded)
        logits = self.topology_head(hidden)
        return logits.view(reps.shape[0], reps.shape[1], self.num_targets, self.num_bins)

    def contact_logits(
        self,
        reps: torch.Tensor,
        tokens: torch.Tensor,
        pair_batch: torch.Tensor,
        pair_i: torch.Tensor,
        pair_j: torch.Tensor,
    ) -> torch.Tensor:
        key_padding_mask = ~self.residue_mask(tokens)
        logits = torch.empty(pair_batch.shape[0], device=reps.device, dtype=torch.float32)
        for batch_idx in torch.unique(pair_batch).tolist():
            pair_mask = pair_batch == int(batch_idx)
            pair_rows = torch.nonzero(pair_mask, as_tuple=False).squeeze(-1)
            hi = reps[pair_batch[pair_rows], pair_i[pair_rows]]
            hj = reps[pair_batch[pair_rows], pair_j[pair_rows]]
            if self.pair_query_mode == "concat":
                pair_features = torch.cat([hi, hj], dim=-1)
            else:
                sep = (pair_j[pair_rows] - pair_i[pair_rows]).float().unsqueeze(-1) / max(float(tokens.shape[1]), 1.0)
                pair_features = torch.cat([hi, hj, torch.abs(hi - hj), hi * hj, sep], dim=-1)
            query = self.pair_query(pair_features).unsqueeze(0)
            decoded, _ = self.pair_attn(
                query=query,
                key=reps[int(batch_idx) : int(batch_idx) + 1],
                value=reps[int(batch_idx) : int(batch_idx) + 1],
                key_padding_mask=key_padding_mask[int(batch_idx) : int(batch_idx) + 1],
                need_weights=False,
            )
            hidden = self.pair_norm(query + decoded).squeeze(0)
            logits[pair_rows] = self.contact_head(hidden).squeeze(-1).float()
        return logits


def set_head_trainability(model: ESM2LoRAXAttnModel, task: str) -> None:
    topology_modules = [model.topology_attn, model.topology_norm, model.topology_head]
    contact_modules = [model.pair_query, model.pair_attn, model.pair_norm, model.contact_head]

    if task in {"lr_contact_xattn", "mlm_only"}:
        for module in topology_modules:
            for param in module.parameters():
                param.requires_grad = False
    if task in {"topology_xattn", "mlm_only"}:
        for module in contact_modules:
            for param in module.parameters():
                param.requires_grad = False


class ContactStructureDataset(Dataset):
    def __init__(
        self,
        residue_labels: Path,
        alphabet: esm.data.Alphabet,
        split: str,
        max_residues: int,
        max_proteins: int | None,
    ):
        columns = [
            "protein_id",
            "split",
            "full_sequence",
            "source_position",
            "source_path",
            "chain_id",
            "pdb_resseq",
            "insertion_code",
        ]
        df = pd.read_csv(residue_labels, usecols=lambda col: col in columns)
        df = df[df["split"] == split].copy()
        if df.empty:
            raise ValueError(f"No rows for split {split}")
        self.examples: list[dict[str, object]] = []
        offset = int(alphabet.prepend_bos)
        max_tokens = max_residues + int(alphabet.prepend_bos) + int(alphabet.append_eos)
        for protein_id, group in df.groupby("protein_id", sort=True):
            if max_proteins is not None and len(self.examples) >= max_proteins:
                break
            sequence = str(group["full_sequence"].iloc[0]).strip().upper()
            source_path = Path(str(group["source_path"].iloc[0]))
            chain = str(group["chain_id"].iloc[0]).strip() or None
            if not sequence or not source_path.exists():
                continue
            ca = parse_ca_coordinates(source_path, chain)
            if not ca:
                continue
            coords = np.full((max_tokens, 3), np.nan, dtype=np.float32)
            coord_valid = np.zeros((max_tokens,), dtype=bool)
            group = group.copy()
            group["_pos"] = pd.to_numeric(group["source_position"], errors="coerce")
            group = group.dropna(subset=["_pos"]).sort_values("_pos")
            for _, row in group.iterrows():
                residue_pos = int(row["_pos"]) - 1
                if residue_pos < 0 or residue_pos >= max_residues:
                    continue
                token_pos = residue_pos + offset
                row_chain = str(row["chain_id"]).strip() or "_"
                try:
                    resseq = int(row["pdb_resseq"])
                except (TypeError, ValueError):
                    continue
                icode = "" if pd.isna(row["insertion_code"]) else str(row["insertion_code"]).strip()
                xyz = ca.get((row_chain, resseq, icode))
                if xyz is None:
                    continue
                coords[token_pos] = xyz
                coord_valid[token_pos] = True
            if int(coord_valid.sum()) < 8:
                continue
            self.examples.append(
                {
                    "protein_id": str(protein_id),
                    "sequence": sequence[:max_residues],
                    "tokens": encode_esm_sequence(sequence, alphabet, max_residues),
                    "coords": torch.tensor(coords, dtype=torch.float32),
                    "coord_valid": torch.tensor(coord_valid, dtype=torch.bool),
                }
            )
        if not self.examples:
            raise ValueError(f"No usable contact examples for split {split}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.examples[idx]


class TarContactStructureDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        structure_tar: Path,
        alphabet: esm.data.Alphabet,
        split: str,
        max_residues: int,
        max_proteins: int | None,
    ):
        self.structure_tar = structure_tar
        self.alphabet = alphabet
        self.max_residues = max_residues
        self.offset = int(alphabet.prepend_bos)
        self.max_tokens = max_residues + int(alphabet.prepend_bos) + int(alphabet.append_eos)
        self.rows: list[dict[str, object]] = []
        self._fh = None
        with open_manifest(manifest) as handle:
            reader = csv.DictReader(handle)
            required = {"accession", "split", "sequence", "offset_data", "size"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Contact manifest missing columns: {sorted(missing)}")
            for row in reader:
                if row["split"] != split:
                    continue
                sequence = row["sequence"].strip().upper()
                if not sequence:
                    continue
                self.rows.append(
                    {
                        "protein_id": row["accession"],
                        "sequence": sequence,
                        "offset_data": int(row["offset_data"]),
                        "size": int(row["size"]),
                    }
                )
                if max_proteins is not None and len(self.rows) >= max_proteins:
                    break
        if not self.rows:
            raise ValueError(f"No usable contact manifest rows for split {split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_fh"] = None
        return state

    def _file_handle(self):
        if self._fh is None:
            self._fh = self.structure_tar.open("rb")
        return self._fh

    def _read_pdb_text(self, offset_data: int, size: int) -> str:
        handle = self._file_handle()
        handle.seek(offset_data)
        payload = handle.read(size)
        return gzip.decompress(payload).decode("utf-8", errors="replace")

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.rows[idx]
        sequence = str(row["sequence"])
        pdb_text = self._read_pdb_text(int(row["offset_data"]), int(row["size"]))
        ca = parse_ca_coordinates_from_text(pdb_text, chain="A")
        coords = np.full((self.max_tokens, 3), np.nan, dtype=np.float32)
        coord_valid = np.zeros((self.max_tokens,), dtype=bool)
        for (_, resseq, icode), xyz in ca.items():
            if icode:
                continue
            residue_pos = resseq - 1
            if residue_pos < 0 or residue_pos >= self.max_residues:
                continue
            token_pos = residue_pos + self.offset
            if token_pos < 0 or token_pos >= self.max_tokens:
                continue
            coords[token_pos] = xyz
            coord_valid[token_pos] = True
        return {
            "protein_id": str(row["protein_id"]),
            "sequence": sequence[: self.max_residues],
            "tokens": encode_esm_sequence(sequence, self.alphabet, self.max_residues),
            "coords": torch.tensor(coords, dtype=torch.float32),
            "coord_valid": torch.tensor(coord_valid, dtype=torch.bool),
        }


class CombinedTopologyContactDataset(Dataset):
    def __init__(
        self,
        topology_examples: list[dict[str, object]],
        manifest: Path,
        structure_tar: Path,
        alphabet: esm.data.Alphabet,
        split: str,
        max_residues: int,
    ):
        self.topology_examples = topology_examples
        self.structure_tar = structure_tar
        self.alphabet = alphabet
        self.max_residues = max_residues
        self.offset = int(alphabet.prepend_bos)
        self.max_tokens = max_residues + int(alphabet.prepend_bos) + int(alphabet.append_eos)
        self._fh = None

        wanted = {str(example["protein_id"]) for example in topology_examples}
        self.manifest_by_id: dict[str, dict[str, object]] = {}
        with open_manifest(manifest) as handle:
            reader = csv.DictReader(handle)
            required = {"accession", "split", "sequence", "offset_data", "size"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Contact manifest missing columns: {sorted(missing)}")
            for row in reader:
                if row["split"] != split:
                    continue
                accession = row["accession"].strip()
                if accession not in wanted:
                    continue
                self.manifest_by_id[accession] = {
                    "sequence": row["sequence"].strip().upper(),
                    "offset_data": int(row["offset_data"]),
                    "size": int(row["size"]),
                }
        missing_ids = sorted(wanted - set(self.manifest_by_id))
        if missing_ids:
            raise ValueError(f"Combined dataset missing manifest rows for {len(missing_ids)} proteins; first={missing_ids[:5]}")

    def __len__(self) -> int:
        return len(self.topology_examples)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_fh"] = None
        return state

    def _file_handle(self):
        if self._fh is None:
            self._fh = self.structure_tar.open("rb")
        return self._fh

    def _read_pdb_text(self, offset_data: int, size: int) -> str:
        handle = self._file_handle()
        handle.seek(offset_data)
        payload = handle.read(size)
        return gzip.decompress(payload).decode("utf-8", errors="replace")

    def __getitem__(self, idx: int) -> dict[str, object]:
        example = dict(self.topology_examples[idx])
        protein_id = str(example["protein_id"])
        row = self.manifest_by_id[protein_id]
        pdb_text = self._read_pdb_text(int(row["offset_data"]), int(row["size"]))
        ca = parse_ca_coordinates_from_text(pdb_text, chain="A")
        coords = np.full((self.max_tokens, 3), np.nan, dtype=np.float32)
        coord_valid = np.zeros((self.max_tokens,), dtype=bool)
        for (_, resseq, icode), xyz in ca.items():
            if icode:
                continue
            residue_pos = resseq - 1
            if residue_pos < 0 or residue_pos >= self.max_residues:
                continue
            token_pos = residue_pos + self.offset
            if token_pos < 0 or token_pos >= self.max_tokens:
                continue
            coords[token_pos] = xyz
            coord_valid[token_pos] = True
        example["coords"] = torch.tensor(coords, dtype=torch.float32)
        example["coord_valid"] = torch.tensor(coord_valid, dtype=torch.bool)
        return example


class SequenceManifestDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        alphabet: esm.data.Alphabet,
        split: str,
        max_residues: int,
        max_proteins: int | None,
    ):
        self.alphabet = alphabet
        self.max_residues = max_residues
        self.rows: list[dict[str, str]] = []
        with open_manifest(manifest) as handle:
            reader = csv.DictReader(handle)
            required = {"accession", "split", "sequence"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Sequence manifest missing columns: {sorted(missing)}")
            for row in reader:
                if row["split"] != split:
                    continue
                sequence = row["sequence"].strip().upper()
                if not sequence:
                    continue
                self.rows.append({"protein_id": row["accession"], "sequence": sequence})
                if max_proteins is not None and len(self.rows) >= max_proteins:
                    break
        if not self.rows:
            raise ValueError(f"No usable sequence manifest rows for split {split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.rows[idx]
        sequence = row["sequence"]
        return {
            "protein_id": row["protein_id"],
            "sequence": sequence[: self.max_residues],
            "tokens": encode_esm_sequence(sequence, self.alphabet, self.max_residues),
        }


def batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def topology_loss_and_metrics(
    model: ESM2LoRAXAttnModel,
    reps: torch.Tensor,
    tokens: torch.Tensor,
    batch: dict[str, object],
    target_names: list[str],
) -> tuple[torch.Tensor, dict[str, float | int]]:
    logits = model.topology_logits(reps, key_padding_mask=~model.residue_mask(tokens))
    valid = batch["valid"]
    labels = batch["bins"]
    losses = []
    metrics: dict[str, float | int] = {}
    accuracies = []
    for target_idx, target in enumerate(target_names):
        target_valid = valid[:, :, target_idx] & labels[:, :, target_idx].ge(0)
        count = int(target_valid.sum().item())
        metrics[f"{target}_count"] = count
        if count == 0:
            metrics[f"{target}_loss"] = 0.0
            metrics[f"{target}_accuracy"] = 0.0
            metrics[f"{target}_correct"] = 0
            continue
        target_logits = logits[:, :, target_idx, :][target_valid]
        target_labels = labels[:, :, target_idx][target_valid]
        loss = nn.functional.cross_entropy(target_logits, target_labels)
        pred = target_logits.argmax(dim=-1)
        correct = int((pred == target_labels).sum().item())
        losses.append(loss)
        acc = correct / max(count, 1)
        accuracies.append(acc)
        metrics[f"{target}_loss"] = float(loss.detach().item())
        metrics[f"{target}_accuracy"] = acc
        metrics[f"{target}_correct"] = correct
    if not losses:
        return torch.tensor(0.0, device=reps.device), metrics
    metrics["topology_loss_mean"] = float(np.mean([float(metrics[f"{target}_loss"]) for target in target_names]))
    metrics["topology_accuracy_mean"] = float(np.mean(accuracies)) if accuracies else 0.0
    return torch.stack(losses).mean(), metrics


def sample_contact_pairs(
    coords: torch.Tensor,
    coord_valid: torch.Tensor,
    long_range_cutoff: int,
    max_sequence_separation: int | None,
    contact_cutoff: float,
    max_pairs_per_protein: int,
    negative_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pair_b: list[torch.Tensor] = []
    pair_i: list[torch.Tensor] = []
    pair_j: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch_idx in range(coords.shape[0]):
        valid_pos = coord_valid[batch_idx].nonzero(as_tuple=False).squeeze(1)
        if valid_pos.numel() < 2:
            continue
        c = coords[batch_idx, valid_pos]
        dist = torch.cdist(c.float(), c.float())
        sep = torch.abs(valid_pos[:, None] - valid_pos[None, :])
        mask = sep >= long_range_cutoff
        if max_sequence_separation is not None:
            mask = mask & (sep <= max_sequence_separation)
        mask = torch.triu(mask, diagonal=1)
        if not mask.any():
            continue
        contact = (dist <= contact_cutoff) & mask
        non_contact = (dist > contact_cutoff) & mask
        pos = contact.nonzero(as_tuple=False)
        neg = non_contact.nonzero(as_tuple=False)
        if pos.numel() == 0 and neg.numel() == 0:
            continue
        if pos.shape[0] > max_pairs_per_protein:
            pos = pos[torch.randperm(pos.shape[0], device=coords.device)[:max_pairs_per_protein]]
        neg_limit = max_pairs_per_protein if pos.shape[0] == 0 else int(max(pos.shape[0] * negative_ratio, 1))
        neg_limit = min(neg_limit, max_pairs_per_protein)
        if neg.shape[0] > neg_limit:
            neg = neg[torch.randperm(neg.shape[0], device=coords.device)[:neg_limit]]
        pairs = torch.cat([pos, neg], dim=0)
        if pairs.numel() == 0:
            continue
        y = torch.cat(
            [
                torch.ones(pos.shape[0], device=coords.device),
                torch.zeros(neg.shape[0], device=coords.device),
            ],
            dim=0,
        )
        order = torch.randperm(pairs.shape[0], device=coords.device)
        pairs = pairs[order]
        y = y[order]
        pair_b.append(torch.full((pairs.shape[0],), batch_idx, dtype=torch.long, device=coords.device))
        pair_i.append(valid_pos[pairs[:, 0]])
        pair_j.append(valid_pos[pairs[:, 1]])
        labels.append(y)
    if not pair_b:
        empty_long = torch.empty(0, dtype=torch.long, device=coords.device)
        empty_float = torch.empty(0, dtype=torch.float32, device=coords.device)
        return empty_long, empty_long, empty_long, empty_float
    return torch.cat(pair_b), torch.cat(pair_i), torch.cat(pair_j), torch.cat(labels)


def contact_loss_and_metrics(
    model: ESM2LoRAXAttnModel,
    reps: torch.Tensor,
    tokens: torch.Tensor,
    batch: dict[str, object],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    pair_b, pair_i, pair_j, labels = sample_contact_pairs(
        batch["coords"],
        batch["coord_valid"],
        args.long_range_cutoff,
        args.max_sequence_separation,
        args.contact_cutoff,
        args.max_pairs_per_protein,
        args.negative_ratio,
    )
    metrics: dict[str, float | int] = {"contact_pairs": int(labels.numel())}
    if labels.numel() == 0:
        return torch.tensor(0.0, device=tokens.device), metrics
    logits = model.contact_logits(reps, tokens, pair_b, pair_i, pair_j)
    pos = float(labels.sum().item())
    neg = float(labels.numel() - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=tokens.device)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    pred = (torch.sigmoid(logits) >= 0.5).float()
    correct = int((pred == labels).sum().item())
    tp = int(((pred == 1) & (labels == 1)).sum().item())
    tn = int(((pred == 0) & (labels == 0)).sum().item())
    fp = int(((pred == 1) & (labels == 0)).sum().item())
    fn = int(((pred == 0) & (labels == 1)).sum().item())
    metrics.update(
        {
            "contact_loss": float(loss.detach().item()),
            "contact_accuracy": correct / max(int(labels.numel()), 1),
            "contact_correct": correct,
            "contact_positive": int(pos),
            "contact_negative": int(neg),
            "contact_tp": tp,
            "contact_tn": tn,
            "contact_fp": fp,
            "contact_fn": fn,
        }
    )
    return loss, metrics


def forward_loss_and_metrics(
    model: ESM2LoRAXAttnModel,
    batch: dict[str, object],
    args: argparse.Namespace,
    targets: list[str],
    use_amp: bool,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    device_type = "cuda" if batch["tokens"].device.type == "cuda" else "cpu"
    input_tokens = batch["tokens"]
    mlm_labels = None
    if args.mlm_weight > 0:
        input_tokens, mlm_labels = mask_tokens_for_mlm(batch["tokens"], model.alphabet, args.mlm_probability)
    metrics: dict[str, float | int] = {}
    with torch.amp.autocast(device_type=device_type, enabled=use_amp):
        mlm_logits, reps = model.esm_forward(input_tokens)
        loss = torch.tensor(0.0, device=batch["tokens"].device)
        if args.task == "topology_xattn":
            task_loss, task_metrics = topology_loss_and_metrics(model, reps, input_tokens, batch, targets)
            loss = loss + args.structure_weight * task_loss
            metrics.update(task_metrics)
        elif args.task == "lr_contact_xattn":
            task_loss, task_metrics = contact_loss_and_metrics(model, reps, input_tokens, batch, args)
            loss = loss + args.structure_weight * task_loss
            metrics.update(task_metrics)
        elif args.task == "combined_xattn":
            topology_loss, topology_metrics = topology_loss_and_metrics(model, reps, input_tokens, batch, targets)
            contact_loss, contact_metrics = contact_loss_and_metrics(model, reps, input_tokens, batch, args)
            loss = loss + args.topology_weight * topology_loss + args.contact_weight * contact_loss
            metrics.update(topology_metrics)
            metrics.update(contact_metrics)
        elif args.task == "mlm_only":
            metrics["sequence_count"] = int(input_tokens.shape[0])
        else:
            raise ValueError(f"Unknown task: {args.task}")
        if args.mlm_weight > 0 and mlm_labels is not None:
            mlm_loss = nn.functional.cross_entropy(
                mlm_logits.reshape(-1, mlm_logits.shape[-1]),
                mlm_labels.reshape(-1),
                ignore_index=-100,
            )
            loss = loss + args.mlm_weight * mlm_loss
            metrics["mlm_loss"] = float(mlm_loss.detach().item())
    if args.mlm_weight > 0 and mlm_labels is not None:
        correct, total = masked_accuracy(mlm_logits.detach(), mlm_labels)
        metrics["mlm_correct"] = correct
        metrics["mlm_total"] = total
    metrics["total_loss"] = float(loss.detach().item())
    return loss, metrics


def aggregate_metrics(rows: list[dict[str, float | int]], targets: list[str], task: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not rows:
        return out
    out["total_loss"] = float(np.mean([float(row.get("total_loss", 0.0)) for row in rows]))
    if task in {"topology_xattn", "combined_xattn"}:
        accuracies = []
        for target in targets:
            correct = sum(int(row.get(f"{target}_correct", 0)) for row in rows)
            count = sum(int(row.get(f"{target}_count", 0)) for row in rows)
            if count:
                acc = correct / count
                out[f"{target}_accuracy"] = acc
                accuracies.append(acc)
        out["topology_accuracy_mean"] = float(np.mean(accuracies)) if accuracies else 0.0
        out["topology_loss_mean"] = float(np.mean([float(row.get("topology_loss_mean", 0.0)) for row in rows]))
    if task in {"lr_contact_xattn", "combined_xattn"}:
        correct = sum(int(row.get("contact_correct", 0)) for row in rows)
        pairs = sum(int(row.get("contact_pairs", 0)) for row in rows)
        positives = sum(int(row.get("contact_positive", 0)) for row in rows)
        negatives = sum(int(row.get("contact_negative", 0)) for row in rows)
        tp = sum(int(row.get("contact_tp", 0)) for row in rows)
        tn = sum(int(row.get("contact_tn", 0)) for row in rows)
        fp = sum(int(row.get("contact_fp", 0)) for row in rows)
        fn = sum(int(row.get("contact_fn", 0)) for row in rows)
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        out["contact_accuracy"] = correct / max(pairs, 1)
        out["contact_balanced_accuracy"] = 0.5 * (sensitivity + specificity)
        out["contact_sensitivity"] = sensitivity
        out["contact_specificity"] = specificity
        out["contact_loss"] = float(np.mean([float(row.get("contact_loss", 0.0)) for row in rows]))
        out["contact_pairs"] = float(pairs)
        out["contact_positive_fraction"] = positives / max(positives + negatives, 1)
    mlm_correct = sum(int(row.get("mlm_correct", 0)) for row in rows)
    mlm_total = sum(int(row.get("mlm_total", 0)) for row in rows)
    if mlm_total:
        out["mlm_accuracy"] = mlm_correct / mlm_total
        out["mlm_loss"] = float(np.mean([float(row.get("mlm_loss", 0.0)) for row in rows]))
    return out


def selection_value(metrics: dict[str, float], metric: str) -> float:
    value = float(metrics.get(metric, float("nan")))
    if not math.isfinite(value):
        raise KeyError(f"Selection metric {metric!r} missing from metrics: {sorted(metrics)}")
    if metric.endswith("_loss") or metric == "total_loss":
        return -value
    return value


def build_loaders(
    args: argparse.Namespace,
    alphabet: esm.data.Alphabet,
    targets: list[str],
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, object]]:
    if args.task == "mlm_only":
        if args.contact_manifest is None:
            raise ValueError("--contact-manifest is required for mlm_only")
        train_ds = SequenceManifestDataset(args.contact_manifest, alphabet, "train", args.max_residues, args.max_train_proteins)
        val_ds = SequenceManifestDataset(args.contact_manifest, alphabet, "val", args.max_residues, args.max_val_proteins)
        test_ds = SequenceManifestDataset(args.contact_manifest, alphabet, "test", args.max_residues, args.max_test_proteins)
        metadata = {
            "dataset_kind": "sequence_manifest",
            "contact_manifest": str(args.contact_manifest),
            "target_scalers": {},
        }
    elif args.task == "topology_xattn":
        if args.residue_labels is None:
            raise ValueError("--residue-labels is required for topology_xattn")
        splits, scalers, metadata = build_or_load_cached_examples(args, targets, alphabet)
        train_ds = CachedStructureTopologyDataset(splits["train"], args.max_train_proteins)
        val_ds = CachedStructureTopologyDataset(splits["val"], args.max_val_proteins)
        test_ds = CachedStructureTopologyDataset(splits["test"], args.max_test_proteins)
        metadata = {"dataset_kind": "topology_bins", "cache_metadata": metadata, "target_scalers": {k: v.to_dict() for k, v in scalers.items()}}
    elif args.task == "combined_xattn":
        if args.residue_labels is None:
            raise ValueError("--residue-labels is required for combined_xattn")
        if args.contact_manifest is None or args.structure_tar is None:
            raise ValueError("--contact-manifest and --structure-tar are required for combined_xattn")
        splits, scalers, metadata = build_or_load_cached_examples(args, targets, alphabet)
        train_ds = CombinedTopologyContactDataset(
            splits["train"],
            args.contact_manifest,
            args.structure_tar,
            alphabet,
            "train",
            args.max_residues,
        )
        val_ds = CombinedTopologyContactDataset(
            splits["val"],
            args.contact_manifest,
            args.structure_tar,
            alphabet,
            "val",
            args.max_residues,
        )
        test_ds = CombinedTopologyContactDataset(
            splits["test"],
            args.contact_manifest,
            args.structure_tar,
            alphabet,
            "test",
            args.max_residues,
        )
        metadata = {
            "dataset_kind": "combined_topology_bins_long_range_contact_tar_manifest",
            "contact_manifest": str(args.contact_manifest),
            "structure_tar": str(args.structure_tar),
            "cache_metadata": metadata,
            "target_scalers": {k: v.to_dict() for k, v in scalers.items()},
        }
    elif args.contact_manifest is not None:
        if args.structure_tar is None:
            raise ValueError("--structure-tar is required with --contact-manifest")
        train_ds = TarContactStructureDataset(
            args.contact_manifest,
            args.structure_tar,
            alphabet,
            "train",
            args.max_residues,
            args.max_train_proteins,
        )
        val_ds = TarContactStructureDataset(
            args.contact_manifest,
            args.structure_tar,
            alphabet,
            "val",
            args.max_residues,
            args.max_val_proteins,
        )
        test_ds = TarContactStructureDataset(
            args.contact_manifest,
            args.structure_tar,
            alphabet,
            "test",
            args.max_residues,
            args.max_test_proteins,
        )
        metadata = {
            "dataset_kind": "long_range_contact_tar_manifest",
            "contact_manifest": str(args.contact_manifest),
            "structure_tar": str(args.structure_tar),
            "target_scalers": {},
        }
    else:
        if args.residue_labels is None:
            raise ValueError("--residue-labels or --contact-manifest is required for lr_contact_xattn")
        train_ds = ContactStructureDataset(args.residue_labels, alphabet, "train", args.max_residues, args.max_train_proteins)
        val_ds = ContactStructureDataset(args.residue_labels, alphabet, "val", args.max_residues, args.max_val_proteins)
        test_ds = ContactStructureDataset(args.residue_labels, alphabet, "test", args.max_residues, args.max_test_proteins)
        metadata = {"dataset_kind": "long_range_contact", "target_scalers": {}}
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if args.task in {"lr_contact_xattn", "combined_xattn"}:
        metadata.update(
            {
                "minimum_sequence_separation": args.long_range_cutoff,
                "maximum_sequence_separation": args.max_sequence_separation,
                "contact_cutoff_angstrom": args.contact_cutoff,
            }
        )
    metadata.update({"train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds)})
    return train_loader, val_loader, test_loader, metadata


def evaluate(
    model: ESM2LoRAXAttnModel,
    loader: DataLoader,
    args: argparse.Namespace,
    targets: list[str],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float | int]] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = batch_to_device(raw_batch, device)
            _, metrics = forward_loss_and_metrics(model, batch, args, targets, args.amp and device.type == "cuda")
            rows.append(metrics)
    return aggregate_metrics(rows, targets, args.task)


def save_checkpoint(
    path: Path,
    model: ESM2LoRAXAttnModel,
    args: argparse.Namespace,
    lora_config: LoRAConfig,
    targets: list[str],
    epoch: int,
    val_metrics: dict[str, float],
    best_selection: float,
    dataset_metadata: dict[str, object],
) -> None:
    compatible_targets = targets if targets else (["mlm_only"] if args.task == "mlm_only" else ["lr_contact"])
    checkpoint = {
            "task_type": f"esm2_{args.finetune_mode}_{args.task}",
            "architecture": f"esm2_{args.finetune_mode}",
            "model_name": args.model_name,
            "esm_num_layers": int(model.esm.num_layers),
            "esm_embed_dim": int(model.esm.embed_dim),
            "pair_query_mode": model.pair_query_mode,
            "lora_config": lora_config.to_dict(),
            "lora_modules": model.lora_modules,
            "finetune_mode": model.finetune_mode,
            # Keep legacy adapter/head fields for LoRA checkpoints. Full
            # checkpoints store one complete state dict to avoid duplicating
            # roughly 150M parameters on disk.
            "trainable_state_dict": lora_only_state_dict(model) if args.finetune_mode == "lora" else {},
            "xattn_trainable_state_dict": trainable_state_dict(model) if args.finetune_mode == "lora" else {},
            "objective": "bins",
            "targets": compatible_targets,
            "num_bins": args.num_bins,
            "target_transform": args.target_transform,
            "target_scalers": dataset_metadata.get("target_scalers", {}),
            "args": vars(args),
            "epoch": epoch,
            "val_metrics": val_metrics,
            "selection_metric": args.selection_metric,
            "val_selection_score": best_selection,
        }
    if args.finetune_mode == "full":
        checkpoint["model_state_dict"] = model.state_dict()
    torch.save(checkpoint, path)


def load_xattn_trainable_state(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    full_state = checkpoint.get("model_state_dict")
    if full_state is not None:
        missing, unexpected = model.load_state_dict(full_state, strict=False)
        return {
            "path": str(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "val_metrics": checkpoint.get("val_metrics", {}),
            "val_selection_score": checkpoint.get("val_selection_score"),
            "loaded": "model_state_dict",
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        }
    state = checkpoint.get("xattn_trainable_state_dict", checkpoint.get("trainable_state_dict"))
    if state is None:
        raise KeyError(f"Checkpoint has no trainable state: {checkpoint_path}")
    current = model.state_dict()
    missing = []
    for name, tensor in state.items():
        if name not in current:
            missing.append(name)
            continue
        current[name].copy_(tensor.to(current[name].device, dtype=current[name].dtype))
    if missing:
        raise KeyError(f"Checkpoint contains unknown trainable parameters: {missing[:10]}")
    model.load_state_dict(current, strict=False)
    return {
        "path": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "val_metrics": checkpoint.get("val_metrics", {}),
        "val_selection_score": checkpoint.get("val_selection_score"),
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    if args.task in {"lr_contact_xattn", "mlm_only"}:
        targets = []
    lora_targets = [target.strip() for target in args.lora_targets.split(",") if target.strip()]
    if args.finetune_mode == "full":
        # Full fine-tuning must not carry adapter targets in the run metadata.
        lora_targets = []
    if args.freeze_esm:
        if args.task == "mlm_only":
            raise ValueError("--freeze-esm is not meaningful for mlm_only because no trainable path reaches the MLM loss")
        lora_targets = []
    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        targets=lora_targets,
        layers=parse_lora_layers(args.lora_layers),
    )
    if args.finetune_mode == "full" and args.freeze_esm:
        raise ValueError("--freeze-esm cannot be combined with --finetune-mode full")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_for_alphabet, alphabet = esm.pretrained.load_model_and_alphabet(args.model_name)
    del base_for_alphabet
    train_loader, val_loader, test_loader, dataset_metadata = build_loaders(args, alphabet, targets)
    model = ESM2LoRAXAttnModel(
        args.model_name,
        lora_config,
        args.dropout,
        max(len(targets), 1),
        args.num_bins,
        args.attention_heads,
        device,
        args.pair_query_mode,
        args.finetune_mode,
    )
    set_head_trainability(model, args.task)
    init_checkpoint_metadata = None
    if args.init_checkpoint is not None:
        init_checkpoint_metadata = load_xattn_trainable_state(model, args.init_checkpoint, device)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    encoder_params = [param for param in model.esm.parameters() if param.requires_grad]
    head_params = [
        param for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith("esm.")
    ]
    param_groups = []
    if encoder_params:
        param_groups.append({
            "params": encoder_params,
            "lr": args.encoder_lr if args.encoder_lr is not None else args.lr,
        })
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    run_metadata = {
        "task": args.task,
        "model_name": args.model_name,
        "lora_config": lora_config.to_dict(),
        "lora_modules": model.lora_modules,
        "finetune_mode": model.finetune_mode,
        "encoder_lr": args.encoder_lr if args.encoder_lr is not None else args.lr,
        "pair_query_mode": model.pair_query_mode,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_fraction": trainable_params / max(total_params, 1),
        "dataset_metadata": dataset_metadata,
        "init_checkpoint": init_checkpoint_metadata,
        "epoch_offset": args.epoch_offset,
    }
    with (args.out_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"run_metadata": run_metadata}, indent=2), flush=True)

    best_selection = -float("inf")
    best_path = args.out_dir / "best_model.pt"
    epoch_checkpoint_paths: list[str] = []
    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        reported_epoch = args.epoch_offset + epoch
        model.train()
        rows: list[dict[str, float | int]] = []
        for raw_batch in train_loader:
            batch = batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = forward_loss_and_metrics(model, batch, args, targets, args.amp and device.type == "cuda")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            rows.append(metrics)
        train_metrics = aggregate_metrics(rows, targets, args.task)
        val_metrics = evaluate(model, val_loader, args, targets, device)
        current_selection = selection_value(val_metrics, args.selection_metric)
        row: dict[str, float] = {
            "epoch": float(reported_epoch),
            "elapsed_sec": time.time() - started,
            "val_selected_metric": current_selection,
        }
        for prefix, metrics in [("train", train_metrics), ("val", val_metrics)]:
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = float(value)
        history.append(row)
        print(json.dumps(row), flush=True)
        if args.save_every_epoch:
            epoch_path = args.out_dir / f"epoch_{reported_epoch}.pt"
            save_checkpoint(
                epoch_path,
                model,
                args,
                lora_config,
                targets,
                reported_epoch,
                val_metrics,
                current_selection,
                dataset_metadata,
            )
            epoch_checkpoint_paths.append(str(epoch_path))
        if current_selection > best_selection + args.min_delta:
            best_selection = current_selection
            save_checkpoint(
                best_path,
                model,
                args,
                lora_config,
                targets,
                reported_epoch,
                val_metrics,
                best_selection,
                dataset_metadata,
            )

    checkpoint = torch.load(best_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        current = model.state_dict()
        for name, tensor in checkpoint["xattn_trainable_state_dict"].items():
            if name in current:
                current[name].copy_(tensor.to(current[name].device, dtype=current[name].dtype))
        model.load_state_dict(current, strict=False)
    final_val_metrics = evaluate(model, val_loader, args, targets, device)
    test_metrics = evaluate(model, test_loader, args, targets, device)
    history_path = args.out_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "task": args.task,
        "residue_labels": str(args.residue_labels),
        "out_dir": str(args.out_dir),
        "model_name": args.model_name,
        "lora_config": lora_config.to_dict(),
        "pair_query_mode": model.pair_query_mode,
        "targets": targets,
        "num_bins": args.num_bins,
        "dataset_metadata": dataset_metadata,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_fraction": trainable_params / max(total_params, 1),
        "init_checkpoint": init_checkpoint_metadata,
        "epoch_offset": args.epoch_offset,
        "best_model_path": str(best_path),
        "epoch_checkpoint_paths": epoch_checkpoint_paths,
        "history_path": str(history_path),
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_metrics": checkpoint["val_metrics"],
        "final_val_metrics": final_val_metrics,
        "test_metrics": test_metrics,
        "elapsed_sec": time.time() - started,
    }
    with (args.out_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--task", choices=["topology_xattn", "lr_contact_xattn", "combined_xattn", "mlm_only"], required=True)
    parser.add_argument("--residue-labels", type=Path, default=None)
    parser.add_argument("--contact-manifest", type=Path, default=None)
    parser.add_argument("--structure-tar", type=Path, default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--epoch-offset", type=int, default=0)
    parser.add_argument("--dataset-cache", type=Path, default=None)
    parser.add_argument("--rebuild-dataset-cache", action="store_true")
    parser.add_argument("--cache-builder", choices=["streaming", "pandas"], default="streaming")
    parser.add_argument("--cache-chunksize", type=int, default=500000)
    parser.add_argument("--cache-progress-every", type=int, default=5)
    parser.add_argument("--model-name", default="esm2_t30_150M_UR50D")
    parser.add_argument(
        "--finetune-mode",
        choices=["lora", "full"],
        default="lora",
        help="lora trains adapters only; full updates all ESM2 parameters.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--target-transform", choices=["raw", "log1p"], default="log1p")
    parser.add_argument("--num-bins", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-residues", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=None,
        help="Learning rate for ESM2 parameters; defaults to --lr. Use a smaller value for full fine-tuning.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--mlm-weight", type=float, default=0.3)
    parser.add_argument("--structure-weight", type=float, default=1.0)
    parser.add_argument("--topology-weight", type=float, default=1.0)
    parser.add_argument("--contact-weight", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,v_proj")
    parser.add_argument(
        "--lora-layers",
        default="all",
        help="Transformer layers receiving LoRA: all, a range (22-29), or comma-separated indices/ranges.",
    )
    parser.add_argument("--freeze-esm", action="store_true", help="Train only the task head, leaving the ESM2 encoder frozen.")
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument(
        "--pair-query-mode",
        choices=["full", "concat"],
        default="full",
        help="Pair-query features for lr_contact_xattn: full uses [hi,hj,|hi-hj|,hi*hj,sep]; concat uses [hi,hj].",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Save epoch_N.pt after every epoch so external downstream validation can select the checkpoint.",
    )
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--selection-metric", default=None)
    parser.add_argument("--max-train-proteins", type=int, default=None)
    parser.add_argument("--max-val-proteins", type=int, default=None)
    parser.add_argument("--max-test-proteins", type=int, default=None)
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--long-range-cutoff", type=int, default=12)
    parser.add_argument(
        "--max-sequence-separation",
        type=int,
        default=None,
        help="Optional inclusive upper bound on residue sequence separation; use 3-11 for a short-range control.",
    )
    parser.add_argument("--max-pairs-per-protein", type=int, default=512)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    args = parser.parse_args()
    if args.selection_metric is None:
        if args.task == "topology_xattn":
            args.selection_metric = "topology_accuracy_mean"
        elif args.task in {"lr_contact_xattn", "combined_xattn"}:
            args.selection_metric = "contact_balanced_accuracy"
        else:
            args.selection_metric = "mlm_accuracy"
    if args.attention_heads < 1:
        raise ValueError("--attention-heads must be >= 1")
    if args.num_bins < 2:
        raise ValueError("--num-bins must be >= 2")
    if args.long_range_cutoff < 1:
        raise ValueError("--long-range-cutoff must be >= 1")
    if args.max_sequence_separation is not None and args.max_sequence_separation < args.long_range_cutoff:
        raise ValueError("--max-sequence-separation must be >= --long-range-cutoff")
    if args.task == "mlm_only" and args.mlm_weight <= 0:
        raise ValueError("--mlm-weight must be positive for mlm_only")
    train(args)


if __name__ == "__main__":
    main()
