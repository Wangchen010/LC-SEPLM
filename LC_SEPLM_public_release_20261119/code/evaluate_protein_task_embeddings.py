#!/usr/bin/env python3
"""Evaluate sequence encoder embeddings on standardized protein-level tasks."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multiclass import OneVsRestClassifier

from train_betweenness_pair_encoder import (
    AA_ALPHABET,
    AA_TO_IDX,
    PAD_IDX,
    UNK_IDX,
    CNNSequenceRanker,
    build_model,
)
from train_mlm_graph_transformer import (
    AA_TO_IDX as TRANSFORMER_AA_TO_IDX,
    PAD_IDX as TRANSFORMER_PAD_IDX,
    UNK_IDX as TRANSFORMER_UNK_IDX,
    ProteinTransformer,
)
try:
    from train_esm2_lora_structure_topology import (
        esm2_sequence_embeddings,
        load_esm2_full_state_for_inference,
        load_esm2_lora_for_inference,
        load_esm2_pretrained_for_inference,
    )
    ESM2_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:
    esm2_sequence_embeddings = None
    load_esm2_full_state_for_inference = None
    load_esm2_lora_for_inference = None
    load_esm2_pretrained_for_inference = None
    ESM2_IMPORT_ERROR = exc


KYTE_DOOLITTLE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}
HYDROPHOBIC_AA = set("AILMFWVYC")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def encode_sequence(sequence: str, max_len: int) -> torch.Tensor:
    tokens = [AA_TO_IDX.get(aa, UNK_IDX) for aa in sequence[:max_len]]
    if len(tokens) < max_len:
        tokens.extend([PAD_IDX] * (max_len - len(tokens)))
    return torch.tensor(tokens, dtype=torch.long)


def aa_composition(sequence: str) -> np.ndarray:
    counts = np.zeros(len(AA_ALPHABET), dtype=np.float32)
    aa_to_pos = {aa: idx for idx, aa in enumerate(AA_ALPHABET)}
    for aa in sequence:
        if aa in aa_to_pos:
            counts[aa_to_pos[aa]] += 1.0
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def hydrophobicity_features(sequence: str, windows: tuple[int, ...] = (9, 19)) -> np.ndarray:
    values = np.asarray([KYTE_DOOLITTLE.get(aa, 0.0) for aa in sequence], dtype=np.float32)
    if len(values) == 0:
        return np.zeros(8, dtype=np.float32)

    hydrophobic = np.asarray([1.0 if aa in HYDROPHOBIC_AA else 0.0 for aa in sequence], dtype=np.float32)
    features = [
        float(values.mean()),
        float(values.max()),
        float(values.min()),
        float(values.std()),
        float(hydrophobic.mean()),
    ]
    for window in windows:
        if len(values) >= window:
            kernel = np.ones(window, dtype=np.float32) / float(window)
            features.append(float(np.convolve(values, kernel, mode="valid").max()))
        else:
            features.append(float(values.mean()))

    longest_run = 0
    current_run = 0
    for is_hydrophobic in hydrophobic:
        if is_hydrophobic:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    features.append(float(longest_run) / float(len(sequence)))
    return np.asarray(features, dtype=np.float32)


def load_model(
    checkpoint_path: Path | None,
    seed: int,
    device: torch.device,
    default_architecture: str = "cnn",
    model_config: dict[str, object] | None = None,
) -> nn.Module:
    set_seed(seed)
    config = model_config or {
        "architecture": default_architecture,
        "embed_dim": 64,
        "channels": 128,
        "dropout": 0.15,
    }
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get("model_config") or {
            "architecture": checkpoint.get("architecture", default_architecture),
            "embed_dim": 64,
            "channels": 128,
            "dropout": 0.15,
        }
        model = build_model(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = build_model(config).to(device)
    model.eval()
    return model


def encoder_embeddings(
    model: nn.Module,
    sequences: list[str],
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = torch.stack([encode_sequence(seq, max_len) for seq in sequences[start : start + batch_size]])
            batch = batch.to(device)
            outputs.append(model.encode(batch).cpu().numpy())
    return np.vstack(outputs)


def encode_transformer_sequence(sequence: str, max_len: int) -> torch.Tensor:
    tokens = [TRANSFORMER_AA_TO_IDX.get(aa, TRANSFORMER_UNK_IDX) for aa in sequence[:max_len]]
    if len(tokens) < max_len:
        tokens.extend([TRANSFORMER_PAD_IDX] * (max_len - len(tokens)))
    return torch.tensor(tokens, dtype=torch.long)


def load_transformer_model(checkpoint_path: Path, device: torch.device) -> ProteinTransformer:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("model_config", {})
    model = ProteinTransformer(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def random_transformer_model(args: argparse.Namespace, seed: int, device: torch.device) -> ProteinTransformer:
    set_seed(seed)
    model = ProteinTransformer(
        max_len=args.transformer_random_max_len,
        d_model=args.transformer_random_d_model,
        n_heads=args.transformer_random_n_heads,
        n_layers=args.transformer_random_n_layers,
        dim_feedforward=args.transformer_random_dim_feedforward,
        dropout=args.transformer_random_dropout,
    ).to(device)
    model.eval()
    return model


def transformer_embeddings(
    model: ProteinTransformer,
    sequences: list[str],
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = torch.stack(
                [encode_transformer_sequence(seq, max_len) for seq in sequences[start : start + batch_size]]
            )
            batch = batch.to(device)
            outputs.append(model.encode(batch).cpu().numpy())
    return np.vstack(outputs)


def read_model_list(path: Path | None) -> list[tuple[str, Path]]:
    if path is None:
        return []
    specs: list[tuple[str, Path]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            feature_set = row["feature_set"].strip()
            checkpoint = Path(row["checkpoint"].strip())
            if not feature_set:
                raise ValueError(f"Empty feature_set in {path}")
            specs.append((feature_set, checkpoint))
    return specs


def read_transformer_model_list(path: Path | None) -> list[tuple[str, Path]]:
    return read_model_list(path)


def parse_named_transformer_specs(values: list[str] | None) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected NAME=CHECKPOINT for --transformer-model, got {value!r}")
        name, checkpoint = value.split("=", 1)
        specs.append((name.strip(), Path(checkpoint.strip())))
    return specs


def parse_named_model_specs(values: list[str] | None, option_name: str) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected NAME=CHECKPOINT for {option_name}, got {value!r}")
        name, checkpoint = value.split("=", 1)
        specs.append((name.strip(), Path(checkpoint.strip())))
    return specs


def parse_named_string_specs(values: list[str] | None, option_name: str) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected NAME=VALUE for {option_name}, got {value!r}")
        name, spec = value.split("=", 1)
        specs.append((name.strip(), spec.strip()))
    return specs


def parse_named_esm2_full_state_specs(values: list[str] | None) -> list[tuple[str, str, Path]]:
    specs: list[tuple[str, str, Path]] = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected NAME=MODEL_NAME,CHECKPOINT for --esm2-full-state-model, got {value!r}")
        name, payload = value.split("=", 1)
        if "," not in payload:
            raise ValueError(f"Expected NAME=MODEL_NAME,CHECKPOINT for --esm2-full-state-model, got {value!r}")
        model_name, checkpoint = payload.split(",", 1)
        specs.append((name.strip(), model_name.strip(), Path(checkpoint.strip())))
    return specs


def cache_key(name: str, num_sequences: int, max_len: int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return f"{safe}_n{num_sequences}_maxlen{max_len}.npy"


def load_cached_features(args: argparse.Namespace, name: str, num_sequences: int) -> np.ndarray | None:
    return load_cached_features_for_len(args, name, num_sequences, args.max_len)


def load_cached_features_for_len(
    args: argparse.Namespace,
    name: str,
    num_sequences: int,
    max_len: int,
) -> np.ndarray | None:
    if args.feature_cache_dir is None:
        return None
    path = args.feature_cache_dir / cache_key(name, num_sequences, max_len)
    if path.exists():
        return np.load(path)
    return None


def save_cached_features(args: argparse.Namespace, name: str, features: np.ndarray) -> None:
    save_cached_features_for_len(args, name, features, args.max_len)


def save_cached_features_for_len(
    args: argparse.Namespace,
    name: str,
    features: np.ndarray,
    max_len: int,
) -> None:
    if args.feature_cache_dir is None:
        return
    args.feature_cache_dir.mkdir(parents=True, exist_ok=True)
    path = args.feature_cache_dir / cache_key(name, features.shape[0], max_len)
    if not path.exists():
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with tmp_path.open("wb") as handle:
                np.save(handle, features)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


def parse_random_transformer_specs(values: list[str] | None) -> list[tuple[str, int]]:
    specs: list[tuple[str, int]] = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected NAME=SEED for --transformer-random-init, got {value!r}")
        name, seed = value.split("=", 1)
        specs.append((name.strip(), int(seed)))
    return specs


SEQTOPO_NUMERIC_COLUMNS = [
    "length",
    "degree",
    "weighted_degree_similarity",
    "community_size",
    "intra_community_degree",
    "inter_community_degree",
    "intra_community_weight",
    "inter_community_weight",
    "neighbor_community_count",
    "participation_coefficient",
    "weighted_participation_coefficient",
    "community_bridge_score",
]


SEQTOPO_GROUP_COLUMNS = {
    "all": SEQTOPO_NUMERIC_COLUMNS,
    "degree": ["degree"],
    "community_bridge": ["community_bridge_score"],
}


def load_seqtopo_features(path: Path, record_ids: list[str]) -> tuple[np.ndarray, float]:
    rows_by_id: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows_by_id[row["record_id"]] = row

    features = np.zeros((len(record_ids), len(SEQTOPO_NUMERIC_COLUMNS)), dtype=np.float32)
    matched = 0
    for row_idx, record_id in enumerate(record_ids):
        row = rows_by_id.get(record_id)
        if row is None:
            continue
        matched += 1
        for col_idx, column in enumerate(SEQTOPO_NUMERIC_COLUMNS):
            value = row.get(column, "")
            features[row_idx, col_idx] = float(value) if value not in {"", None} else 0.0
    coverage = matched / max(len(record_ids), 1)
    return features, coverage


def degree_controlled_bridge_features(seqtopo_features: np.ndarray) -> np.ndarray:
    degree_idx = SEQTOPO_NUMERIC_COLUMNS.index("degree")
    bridge_idx = SEQTOPO_NUMERIC_COLUMNS.index("community_bridge_score")
    degree = np.log1p(seqtopo_features[:, degree_idx].astype(np.float64))
    bridge = seqtopo_features[:, bridge_idx].astype(np.float64)
    design = np.column_stack([np.ones_like(degree), degree])
    coef, *_ = np.linalg.lstsq(design, bridge, rcond=None)
    residual = bridge - design @ coef
    return residual.astype(np.float32)[:, None]


def select_seqtopo_group(
    seqtopo_features: np.ndarray,
    group: str,
    seed: int,
) -> tuple[str, np.ndarray]:
    if group in SEQTOPO_GROUP_COLUMNS:
        indices = [SEQTOPO_NUMERIC_COLUMNS.index(column) for column in SEQTOPO_GROUP_COLUMNS[group]]
        return group, seqtopo_features[:, indices]
    if group == "degree_controlled_bridge":
        return group, degree_controlled_bridge_features(seqtopo_features)
    if group == "shuffled_all":
        rng = np.random.default_rng(seed)
        shuffled = seqtopo_features.copy()
        shuffled = shuffled[rng.permutation(len(shuffled))]
        return group, shuffled
    raise ValueError(f"Unsupported SeqTopo feature group: {group}")


def make_feature_sets(args: argparse.Namespace, sequences: list[str]) -> dict[str, np.ndarray]:
    feature_sets = {}
    if not args.skip_default_baselines:
        aa_features = np.vstack([aa_composition(seq) for seq in sequences])
        hydrophobicity = np.vstack([hydrophobicity_features(seq) for seq in sequences])
        feature_sets.update(
            {
                "aa_composition": aa_features,
                "hydrophobicity": hydrophobicity,
                "aa_composition_plus_hydrophobicity": np.hstack([aa_features, hydrophobicity]),
            }
        )
    device = torch.device(args.device)
    model_specs = [] if args.skip_default_baselines else [("cnn_untrained", None)]
    optional_model_specs = [
        ("cnn_original_betweenness", args.original_model),
        ("cnn_residual_betweenness", args.residual_model),
        ("cnn_community_bridge", args.bridge_model),
        ("cnn_degree_controlled_bridge", args.degree_controlled_bridge_model),
        ("cnn_random_labels", args.random_label_model),
        ("cnn_degree", args.degree_model),
    ]
    model_specs.extend((name, checkpoint) for name, checkpoint in optional_model_specs if checkpoint is not None)
    for name, checkpoint in model_specs:
        if checkpoint is not None and not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = load_model(checkpoint, args.seed, device)
        feature_sets[name] = encoder_embeddings(model, sequences, args.max_len, args.batch_size, device)
    tcn_specs = []
    if args.include_tcn_untrained:
        tcn_specs.append(("tcn_untrained", None))
    tcn_specs.extend(read_model_list(args.tcn_model_list))
    tcn_specs.extend(parse_named_model_specs(args.tcn_model, "--tcn-model"))
    tcn_config = {
        "architecture": "tcn",
        "embed_dim": args.tcn_embed_dim,
        "channels": args.tcn_channels,
        "num_layers": args.tcn_layers,
        "kernel_size": args.tcn_kernel_size,
        "dropout": args.tcn_dropout,
    }
    for name, checkpoint in tcn_specs:
        cached = load_cached_features(args, name, len(sequences))
        if cached is not None:
            feature_sets[name] = cached
            continue
        if checkpoint is not None and not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = load_model(checkpoint, args.seed, device, default_architecture="tcn", model_config=tcn_config)
        features = encoder_embeddings(model, sequences, args.max_len, args.batch_size, device)
        feature_sets[name] = features
        save_cached_features(args, name, features)
    transformer_specs = [
        ("transformer_mlm_only", args.transformer_mlm_only_model),
        ("transformer_mlm_degree", args.transformer_mlm_degree_model),
        ("transformer_mlm_community_bridge", args.transformer_mlm_bridge_model),
        ("transformer_mlm_degree_controlled_bridge", args.transformer_mlm_degree_controlled_bridge_model),
        ("transformer_mlm_random_labels", args.transformer_mlm_random_label_model),
    ]
    transformer_specs.extend(read_transformer_model_list(args.transformer_model_list))
    transformer_specs.extend(parse_named_transformer_specs(args.transformer_model))
    for name, checkpoint in transformer_specs:
        if checkpoint is None:
            continue
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = load_transformer_model(checkpoint, device)
        model_max_len = int(getattr(model, "max_len", args.max_len))
        feature_sets[name] = transformer_embeddings(model, sequences, model_max_len, args.batch_size, device)
    for name, model_seed in parse_random_transformer_specs(args.transformer_random_init):
        model = random_transformer_model(args, model_seed, device)
        model_max_len = int(getattr(model, "max_len", args.max_len))
        feature_sets[name] = transformer_embeddings(model, sequences, model_max_len, args.batch_size, device)
    esm2_pretrained_specs = parse_named_string_specs(args.esm2_pretrained_model, "--esm2-pretrained-model")
    esm2_lora_specs = parse_named_model_specs(args.esm2_lora_model, "--esm2-lora-model")
    esm2_full_state_specs = parse_named_esm2_full_state_specs(args.esm2_full_state_model)
    if (esm2_pretrained_specs or esm2_lora_specs or esm2_full_state_specs) and ESM2_IMPORT_ERROR is not None:
        raise RuntimeError("ESM2 evaluation requested, but optional ESM dependencies are unavailable") from ESM2_IMPORT_ERROR
    for name, model_name in esm2_pretrained_specs:
        cached = load_cached_features_for_len(args, name, len(sequences), args.esm2_max_residues)
        if cached is not None:
            feature_sets[name] = cached
            continue
        model, alphabet = load_esm2_pretrained_for_inference(model_name, device)
        features = esm2_sequence_embeddings(
            model,
            alphabet,
            sequences,
            args.esm2_max_residues,
            args.esm2_batch_size,
            device,
            use_amp=args.esm2_amp,
        )
        feature_sets[name] = features
        save_cached_features_for_len(args, name, features, args.esm2_max_residues)
    for name, checkpoint in esm2_lora_specs:
        cached = load_cached_features_for_len(args, name, len(sequences), args.esm2_max_residues)
        if cached is not None:
            feature_sets[name] = cached
            continue
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model, alphabet = load_esm2_lora_for_inference(checkpoint, device)
        features = esm2_sequence_embeddings(
            model,
            alphabet,
            sequences,
            args.esm2_max_residues,
            args.esm2_batch_size,
            device,
            use_amp=args.esm2_amp,
        )
        feature_sets[name] = features
        save_cached_features_for_len(args, name, features, args.esm2_max_residues)
    for name, model_name, checkpoint in esm2_full_state_specs:
        cached = load_cached_features_for_len(args, name, len(sequences), args.esm2_max_residues)
        if cached is not None:
            feature_sets[name] = cached
            continue
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model, alphabet = load_esm2_full_state_for_inference(checkpoint, model_name, device)
        features = esm2_sequence_embeddings(
            model,
            alphabet,
            sequences,
            args.esm2_max_residues,
            args.esm2_batch_size,
            device,
            use_amp=args.esm2_amp,
        )
        feature_sets[name] = features
        save_cached_features_for_len(args, name, features, args.esm2_max_residues)
    return feature_sets


def add_seqtopo_feature_sets(
    feature_sets: dict[str, np.ndarray],
    args: argparse.Namespace,
    record_ids: list[str],
) -> dict[str, float]:
    if args.seqtopo_features_csv is None:
        return {}
    seqtopo_features, coverage = load_seqtopo_features(args.seqtopo_features_csv, record_ids)
    if coverage < args.seqtopo_min_coverage:
        raise ValueError(
            f"SeqTopo feature coverage {coverage:.4f} is below --seqtopo-min-coverage "
            f"{args.seqtopo_min_coverage:.4f}"
        )

    requested_groups = args.seqtopo_feature_group or ["all"]
    existing = list(feature_sets.items())
    group_dimensions = {}
    for group in requested_groups:
        group_key, group_features = select_seqtopo_group(seqtopo_features, group, args.seqtopo_shuffle_seed or args.seed)
        base_name = args.seqtopo_feature_set_name if len(requested_groups) == 1 else f"{args.seqtopo_feature_set_name}_{group_key}"
        feature_sets[base_name] = group_features
        group_dimensions[group_key] = int(group_features.shape[1])
        for name, features in existing:
            if args.seqtopo_concat_prefix and not name.startswith(args.seqtopo_concat_prefix):
                continue
            if args.seqtopo_concat_only and name not in set(args.seqtopo_concat_only):
                continue
            feature_sets[f"{name}_plus_{base_name}"] = np.hstack([features, group_features])
    return {
        "seqtopo_coverage": coverage,
        "seqtopo_num_features": float(seqtopo_features.shape[1]),
        "seqtopo_groups": requested_groups,
        "seqtopo_group_dimensions": group_dimensions,
    }


def add_concat_feature_sets(
    feature_sets: dict[str, np.ndarray],
    specs: list[str],
) -> dict[str, object]:
    concat_dimensions: dict[str, dict[str, int]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=FEATURE_A,FEATURE_B for --concat-feature-set, got {spec!r}")
        name, members_text = spec.split("=", 1)
        name = name.strip()
        members = [member.strip() for member in members_text.split(",") if member.strip()]
        if not name or len(members) < 2:
            raise ValueError(f"Concat spec must name at least two feature sets, got {spec!r}")
        missing = [member for member in members if member not in feature_sets]
        if missing:
            available = ", ".join(sorted(feature_sets))
            raise ValueError(
                f"Unknown feature set(s) for concat {name}: {missing}. "
                f"Available feature sets: {available}"
            )
        lengths = {feature_sets[member].shape[0] for member in members}
        if len(lengths) != 1:
            raise ValueError(f"Feature row-count mismatch for concat {name}: {members}")
        feature_sets[name] = np.hstack([feature_sets[member] for member in members])
        concat_dimensions[name] = {member: int(feature_sets[member].shape[1]) for member in members}
    return {"concat_feature_sets": list(concat_dimensions), "concat_dimensions": concat_dimensions}


def filter_feature_sets(
    feature_sets: dict[str, np.ndarray],
    include_names: list[str],
) -> dict[str, np.ndarray]:
    if not include_names:
        return feature_sets
    missing = [name for name in include_names if name not in feature_sets]
    if missing:
        available = ", ".join(sorted(feature_sets))
        raise ValueError(f"Unknown --include-feature-set value(s): {missing}. Available feature sets: {available}")
    return {name: feature_sets[name] for name in include_names}


def split_indices_from_rows(
    rows: list[dict[str, str]],
    eval_split_mode: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not rows or "split" not in rows[0]:
        return None
    splits = [row.get("split", "").strip().lower() for row in rows]
    if not any(splits):
        return None
    if eval_split_mode == "val":
        train_names, eval_names = {"train"}, {"val"}
    elif eval_split_mode == "test":
        train_names, eval_names = {"train", "val"}, {"test"}
    elif eval_split_mode == "val_test":
        train_names, eval_names = {"train"}, {"val", "test"}
    else:
        raise ValueError(f"Unknown eval split mode: {eval_split_mode}")
    train = np.asarray([idx for idx, split in enumerate(splits) if split in train_names], dtype=np.int64)
    test = np.asarray([idx for idx, split in enumerate(splits) if split in eval_names], dtype=np.int64)
    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            f"Split mode {eval_split_mode!r} requires non-empty train={sorted(train_names)} "
            f"and eval={sorted(eval_names)} rows"
        )
    return train, test


def classification_eval(
    name: str,
    x: np.ndarray,
    labels: list[str],
    rows: list[dict[str, str]],
    seed: int,
    n_splits: int,
    classifier: str,
    max_iter: int,
    eval_split_mode: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    # Some prepared datasets encode the same binary labels as 0/1 in train
    # and 0.0/1.0 in validation/test. Canonicalize numeric labels before
    # constructing the class map so split-specific formatting cannot create
    # phantom classes.
    normalized_labels: list[str] = []
    for label in labels:
        try:
            value = float(label)
            normalized_labels.append(str(int(value)) if value.is_integer() else str(value))
        except ValueError:
            normalized_labels.append(label)
    labels = normalized_labels
    label_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    y = np.asarray([label_to_id[label] for label in labels], dtype=np.int64)
    fixed_split = split_indices_from_rows(rows, eval_split_mode)
    fold_rows = []
    if fixed_split is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(x, y)
    else:
        train_idx, test_idx = fixed_split
        split_iter = [(train_idx, test_idx)]

    for fold, (train_idx, test_idx) in enumerate(split_iter):
        if classifier == "sgd_log_loss":
            estimator = SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                random_state=seed,
                max_iter=max_iter,
                tol=1e-3,
                n_jobs=-1,
            )
        else:
            estimator = LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=seed)
        clf = make_pipeline(
            StandardScaler(),
            estimator,
        )
        clf.fit(x[train_idx], y[train_idx])
        pred = clf.predict(x[test_idx])
        fold_rows.append(
            {
                "feature_set": name,
                "fold": fold,
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "macro_f1": float(f1_score(y[test_idx], pred, average="macro")),
                "weighted_f1": float(f1_score(y[test_idx], pred, average="weighted")),
            }
        )
    return (
        {
            "feature_set": name,
            "accuracy_mean": float(np.mean([row["accuracy"] for row in fold_rows])),
            "accuracy_std": float(np.std([row["accuracy"] for row in fold_rows])),
            "macro_f1_mean": float(np.mean([row["macro_f1"] for row in fold_rows])),
            "macro_f1_std": float(np.std([row["macro_f1"] for row in fold_rows])),
            "weighted_f1_mean": float(np.mean([row["weighted_f1"] for row in fold_rows])),
            "weighted_f1_std": float(np.std([row["weighted_f1"] for row in fold_rows])),
        },
        fold_rows,
    )


def multilabel_eval(
    name: str,
    x: np.ndarray,
    labels: list[str],
    rows: list[dict[str, str]],
    delimiter: str,
    seed: int,
    n_splits: int,
    classifier: str,
    max_iter: int,
    eval_split_mode: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    label_lists = [[part.strip() for part in label.split(delimiter) if part.strip()] for label in labels]
    y = MultiLabelBinarizer().fit_transform(label_lists)
    fixed_split = split_indices_from_rows(rows, eval_split_mode)
    fold_rows = []
    if fixed_split is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(x)
    else:
        train_idx, test_idx = fixed_split
        split_iter = [(train_idx, test_idx)]

    for fold, (train_idx, test_idx) in enumerate(split_iter):
        if classifier == "sgd_log_loss":
            estimator = SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                random_state=seed,
                max_iter=max_iter,
                tol=1e-3,
                n_jobs=-1,
            )
        else:
            estimator = LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=seed)
        clf = make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(estimator, n_jobs=-1),
        )
        clf.fit(x[train_idx], y[train_idx])
        pred = clf.predict(x[test_idx])
        fold_rows.append(
            {
                "feature_set": name,
                "fold": fold,
                "micro_f1": float(f1_score(y[test_idx], pred, average="micro", zero_division=0)),
                "macro_f1": float(f1_score(y[test_idx], pred, average="macro", zero_division=0)),
                "samples_f1": float(f1_score(y[test_idx], pred, average="samples", zero_division=0)),
            }
        )
    return (
        {
            "feature_set": name,
            "micro_f1_mean": float(np.mean([row["micro_f1"] for row in fold_rows])),
            "micro_f1_std": float(np.std([row["micro_f1"] for row in fold_rows])),
            "macro_f1_mean": float(np.mean([row["macro_f1"] for row in fold_rows])),
            "macro_f1_std": float(np.std([row["macro_f1"] for row in fold_rows])),
            "samples_f1_mean": float(np.mean([row["samples_f1"] for row in fold_rows])),
            "samples_f1_std": float(np.std([row["samples_f1"] for row in fold_rows])),
        },
        fold_rows,
    )


def regression_eval(
    name: str,
    x: np.ndarray,
    values: np.ndarray,
    rows: list[dict[str, str]],
    seed: int,
    n_splits: int,
    eval_split_mode: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    fixed_split = split_indices_from_rows(rows, eval_split_mode)
    fold_rows = []
    if fixed_split is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(x)
    else:
        train_idx, test_idx = fixed_split
        split_iter = [(train_idx, test_idx)]

    for fold, (train_idx, test_idx) in enumerate(split_iter):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x[train_idx], values[train_idx])
        pred = model.predict(x[test_idx])
        rho = spearmanr(values[test_idx], pred).statistic
        fold_rows.append(
            {
                "feature_set": name,
                "fold": fold,
                "spearman": float(0.0 if np.isnan(rho) else rho),
                "r2": float(r2_score(values[test_idx], pred)),
                "rmse": float(np.sqrt(mean_squared_error(values[test_idx], pred))),
            }
        )
    return (
        {
            "feature_set": name,
            "spearman_mean": float(np.mean([row["spearman"] for row in fold_rows])),
            "spearman_std": float(np.std([row["spearman"] for row in fold_rows])),
            "r2_mean": float(np.mean([row["r2"] for row in fold_rows])),
            "r2_std": float(np.std([row["r2"] for row in fold_rows])),
            "rmse_mean": float(np.mean([row["rmse"] for row in fold_rows])),
            "rmse_std": float(np.std([row["rmse"] for row in fold_rows])),
        },
        fold_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--task-type", choices=["classification", "multilabel", "regression"], required=True)
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--label-delimiter", default=";")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--eval-split-mode",
        choices=["val", "test", "val_test"],
        default="val_test",
        help="Use val for tuning, test for final evaluation (training on train+val), or legacy val_test.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--classifier", choices=["logistic_regression", "sgd_log_loss"], default="logistic_regression")
    parser.add_argument("--classifier-max-iter", type=int, default=5000)
    parser.add_argument("--original-model", type=Path, default=None)
    parser.add_argument("--residual-model", type=Path, default=None)
    parser.add_argument("--bridge-model", type=Path, default=None)
    parser.add_argument("--degree-controlled-bridge-model", type=Path, default=None)
    parser.add_argument("--random-label-model", type=Path, default=None)
    parser.add_argument("--degree-model", type=Path, default=None)
    parser.add_argument("--skip-default-baselines", action="store_true")
    parser.add_argument("--include-tcn-untrained", action="store_true")
    parser.add_argument("--tcn-model-list", type=Path, default=None)
    parser.add_argument("--tcn-model", action="append", default=[])
    parser.add_argument("--tcn-embed-dim", type=int, default=64)
    parser.add_argument("--tcn-channels", type=int, default=128)
    parser.add_argument("--tcn-layers", type=int, default=5)
    parser.add_argument("--tcn-kernel-size", type=int, default=5)
    parser.add_argument("--tcn-dropout", type=float, default=0.15)
    parser.add_argument("--transformer-mlm-only-model", type=Path, default=None)
    parser.add_argument("--transformer-mlm-degree-model", type=Path, default=None)
    parser.add_argument("--transformer-mlm-bridge-model", type=Path, default=None)
    parser.add_argument("--transformer-mlm-degree-controlled-bridge-model", type=Path, default=None)
    parser.add_argument("--transformer-mlm-random-label-model", type=Path, default=None)
    parser.add_argument("--transformer-model-list", type=Path, default=None)
    parser.add_argument("--transformer-model", action="append", default=[])
    parser.add_argument("--transformer-random-init", action="append", default=[])
    parser.add_argument("--transformer-random-max-len", type=int, default=256)
    parser.add_argument("--transformer-random-d-model", type=int, default=128)
    parser.add_argument("--transformer-random-n-heads", type=int, default=4)
    parser.add_argument("--transformer-random-n-layers", type=int, default=2)
    parser.add_argument("--transformer-random-dim-feedforward", type=int, default=256)
    parser.add_argument("--transformer-random-dropout", type=float, default=0.1)
    parser.add_argument("--esm2-pretrained-model", action="append", default=[])
    parser.add_argument("--esm2-lora-model", action="append", default=[])
    parser.add_argument(
        "--esm2-full-state-model",
        action="append",
        default=[],
        help="Add a full ESM state dict as NAME=MODEL_NAME,CHECKPOINT (for example ESM_S_150M=esm2_t30_150M_UR50D,/path/esm_150m_s.pth).",
    )
    parser.add_argument("--esm2-max-residues", type=int, default=512)
    parser.add_argument("--esm2-batch-size", type=int, default=8)
    parser.add_argument("--esm2-amp", action="store_true")
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--seqtopo-features-csv", type=Path, default=None)
    parser.add_argument("--seqtopo-feature-set-name", default="seqtopo_graph_features")
    parser.add_argument(
        "--seqtopo-feature-group",
        action="append",
        choices=["all", "degree", "community_bridge", "degree_controlled_bridge", "shuffled_all"],
        default=[],
    )
    parser.add_argument("--seqtopo-min-coverage", type=float, default=0.99)
    parser.add_argument("--seqtopo-concat-prefix", default="")
    parser.add_argument("--seqtopo-concat-only", action="append", default=[])
    parser.add_argument("--seqtopo-shuffle-seed", type=int, default=None)
    parser.add_argument(
        "--concat-feature-set",
        action="append",
        default=[],
        help="Add a feature set by horizontal concatenation, formatted as NAME=FEATURE_A,FEATURE_B[,FEATURE_C].",
    )
    parser.add_argument(
        "--include-feature-set",
        action="append",
        default=[],
        help="If provided, evaluate only these named feature sets after all feature construction is complete.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in read_rows(args.task_csv)
        if row.get(args.sequence_col, "").strip() and row.get(args.label_col, "").strip()
    ]
    if not rows:
        raise ValueError("No usable rows after filtering empty sequence or label")

    sequences = [row[args.sequence_col].strip().upper() for row in rows]
    record_ids = [row.get("record_id", row.get("id", "")).strip() for row in rows]
    if any(not record_id for record_id in record_ids):
        raise ValueError("Every task row must provide either record_id or id")
    labels = [row[args.label_col].strip() for row in rows]
    feature_sets = make_feature_sets(args, sequences)
    feature_metadata = add_seqtopo_feature_sets(feature_sets, args, record_ids)
    feature_metadata.update(add_concat_feature_sets(feature_sets, args.concat_feature_set))
    feature_sets = filter_feature_sets(feature_sets, args.include_feature_set)

    summary_rows = []
    fold_rows = []
    for name, features in feature_sets.items():
        if args.task_type == "classification":
            summary, folds = classification_eval(
                name,
                features,
                labels,
                rows,
                args.seed,
                args.n_splits,
                args.classifier,
                args.classifier_max_iter,
                args.eval_split_mode,
            )
            sort_key = "macro_f1_mean"
        elif args.task_type == "multilabel":
            summary, folds = multilabel_eval(
                name,
                features,
                labels,
                rows,
                args.label_delimiter,
                args.seed,
                args.n_splits,
                args.classifier,
                args.classifier_max_iter,
                args.eval_split_mode,
            )
            sort_key = "micro_f1_mean"
        else:
            values = np.asarray([float(label) for label in labels], dtype=np.float32)
            summary, folds = regression_eval(
                name,
                features,
                values,
                rows,
                args.seed,
                args.n_splits,
                args.eval_split_mode,
            )
            sort_key = "spearman_mean"
        summary_rows.append(summary)
        fold_rows.extend(folds)

    summary_rows.sort(key=lambda row: row[sort_key], reverse=True)
    summary_path = args.out_dir / "protein_task_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    folds_path = args.out_dir / "protein_task_folds.csv"
    with folds_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)

    metadata = {
        "task_csv": str(args.task_csv),
        "task_type": args.task_type,
        "num_examples": len(rows),
        "out_dir": str(args.out_dir),
        "summary_path": str(summary_path),
        "folds_path": str(folds_path),
        "sort_key": sort_key,
        **feature_metadata,
        "best": summary_rows[0],
    }
    with (args.out_dir / "protein_task_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
