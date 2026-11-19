#!/usr/bin/env python3
"""Fair frozen-embedding contact probe on homology-filtered AF2 test sets."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train_esm2_lora_structure_topology import (  # noqa: E402
    esm2_representations,
    load_esm2_full_state_for_inference,
    load_esm2_lora_for_inference,
    load_esm2_pretrained_for_inference,
)
from train_esm2_lora_xattn_structure_tasks import (  # noqa: E402
    TarContactStructureDataset,
    encode_esm_sequence,
)


class ExternalNPZContactDataset:
    """Contact structures stored as aligned sequence/CA-coordinate NPZ files."""

    def __init__(self, manifest: Path, alphabet, max_residues: int):
        self.alphabet = alphabet
        self.max_residues = max_residues
        self.offset = int(alphabet.prepend_bos)
        self.max_tokens = max_residues + self.offset + int(alphabet.append_eos)
        self.rows: list[dict[str, str]] = []
        with manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"accession", "sequence", "npz_path"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"External contact manifest missing columns: {sorted(missing)}")
            for row in reader:
                sequence = row["sequence"].strip().upper()
                npz_path = Path(row["npz_path"])
                if sequence and npz_path.is_file():
                    self.rows.append(
                        {"accession": row["accession"], "sequence": sequence, "npz_path": str(npz_path)}
                    )
        if not self.rows:
            raise ValueError(f"No usable external structures in {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.rows[idx]
        sequence = row["sequence"][: self.max_residues]
        with np.load(row["npz_path"], allow_pickle=False) as payload:
            source_coords = np.asarray(payload["coords"], dtype=np.float32)[: self.max_residues]
        if source_coords.shape != (len(sequence), 3):
            raise ValueError(
                f"Coordinate/sequence mismatch for {row['accession']}: {source_coords.shape} vs {len(sequence)}"
            )
        coords = np.full((self.max_tokens, 3), np.nan, dtype=np.float32)
        coords[self.offset : self.offset + len(sequence)] = source_coords
        coord_valid = np.isfinite(coords).all(axis=1)
        return {
            "protein_id": row["accession"],
            "sequence": sequence,
            "tokens": encode_esm_sequence(sequence, self.alphabet, self.max_residues),
            "coords": torch.tensor(coords, dtype=torch.float32),
            "coord_valid": torch.tensor(coord_valid, dtype=torch.bool),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument(
        "--test-manifest",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Repeat for each homology-filtered test set.",
    )
    parser.add_argument(
        "--external-test-manifest",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Repeat for external CSV manifests containing accession, sequence, and npz_path.",
    )
    parser.add_argument(
        "--structure-tar",
        type=Path,
        required=True,
        help="Path to the externally obtained structure archive.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-seeds", default="7,17,23")
    parser.add_argument("--max-residues", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--min-seq-sep", type=int, default=12)
    parser.add_argument("--max-train-proteins", type=int, default=500)
    parser.add_argument("--max-train-pairs", type=int, default=50000)
    parser.add_argument("--max-pairs-per-protein", type=int, default=128)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--pair-batch-size", type=int, default=4096)
    parser.add_argument("--classifier-max-iter", type=int, default=2000)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--model-spec",
        action="append",
        required=True,
        metavar="LABEL=KIND:VALUE",
        help="KIND is pretrained, lora, or full_state.",
    )
    parser.add_argument("--baseline-label", default="ESM2_150M")
    return parser.parse_args()


def parse_labeled_paths(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid labeled path {value!r}; expected LABEL=PATH")
        label, path = value.split("=", 1)
        parsed.append((label.strip(), Path(path.strip())))
    return parsed


def parse_model_specs(values: list[str]) -> list[dict[str, str]]:
    specs = []
    for value in values:
        if "=" not in value or ":" not in value.split("=", 1)[1]:
            raise ValueError(f"Invalid model spec {value!r}; expected LABEL=KIND:VALUE")
        label, payload = value.split("=", 1)
        kind, target = payload.split(":", 1)
        if kind not in {"pretrained", "lora", "full_state"}:
            raise ValueError(f"Unsupported model kind in {value!r}")
        specs.append({"label": label.strip(), "kind": kind.strip(), "target": target.strip()})
    return specs


def load_model(spec: dict[str, str], device: torch.device):
    if spec["kind"] == "pretrained":
        return load_esm2_pretrained_for_inference(spec["target"], device)
    if spec["kind"] == "lora":
        return load_esm2_lora_for_inference(Path(spec["target"]), device)
    model_name, checkpoint = spec["target"].split(",", 1)
    return load_esm2_full_state_for_inference(Path(checkpoint), model_name, device)


def batched_indices(length: int, batch_size: int):
    for start in range(0, length, batch_size):
        yield range(start, min(start + batch_size, length))


def embed_examples(model, examples: list[dict[str, object]], device: torch.device, amp: bool) -> list[np.ndarray]:
    tokens = torch.stack([example["tokens"] for example in examples]).to(device)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.inference_mode(), torch.amp.autocast(device_type=device_type, enabled=amp and device.type == "cuda"):
        reps = esm2_representations(model, tokens)
    return [row.float().cpu().numpy() for row in reps]


def contact_pairs(example: dict[str, object], cutoff: float, min_seq_sep: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(example["coords"], dtype=np.float32)
    valid = np.asarray(example["coord_valid"], dtype=bool)
    positions = np.flatnonzero(valid)
    if len(positions) < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    left, right = np.triu_indices(len(positions), k=1)
    separated = positions[right] - positions[left] >= min_seq_sep
    pair_positions = np.column_stack([positions[left[separated]], positions[right[separated]]])
    if not len(pair_positions):
        return pair_positions, np.empty(0, dtype=np.int64)
    distances = np.linalg.norm(coords[pair_positions[:, 0]] - coords[pair_positions[:, 1]], axis=1)
    return pair_positions, (distances <= cutoff).astype(np.int64)


def pair_features(reps: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    left = reps[pairs[:, 0]]
    right = reps[pairs[:, 1]]
    return np.concatenate([np.abs(left - right), left * right], axis=1).astype(np.float32, copy=False)


def sample_pairs(
    reps: np.ndarray,
    example: dict[str, object],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    pairs, labels = contact_pairs(example, args.contact_cutoff, args.min_seq_sep)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        return np.empty((0, reps.shape[1] * 2), dtype=np.float32), np.empty(0, dtype=np.int64)
    max_positive = max(1, int(args.max_pairs_per_protein / (1.0 + args.negative_ratio)))
    if len(positive) > max_positive:
        positive = rng.choice(positive, size=max_positive, replace=False)
    max_negative = min(len(negative), int(math.ceil(len(positive) * args.negative_ratio)))
    negative = rng.choice(negative, size=max_negative, replace=False)
    selected = np.concatenate([positive, negative])
    rng.shuffle(selected)
    return pair_features(reps, pairs[selected]), labels[selected]


def collect_training_matrix(model, alphabet, args: argparse.Namespace, device: torch.device):
    dataset = TarContactStructureDataset(
        args.train_manifest,
        args.structure_tar,
        alphabet,
        "train",
        args.max_residues,
        args.max_train_proteins,
    )
    rng = np.random.default_rng(20260825)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    proteins = 0
    total_pairs = 0
    for index_group in batched_indices(len(dataset), args.batch_size):
        examples = [dataset[index] for index in index_group]
        reps_list = embed_examples(model, examples, device, args.amp)
        for example, reps in zip(examples, reps_list):
            x_part, y_part = sample_pairs(reps, example, args, rng)
            remaining = args.max_train_pairs - total_pairs
            if not len(y_part) or remaining <= 0:
                continue
            if len(y_part) > remaining:
                x_part = x_part[:remaining]
                y_part = y_part[:remaining]
            xs.append(x_part)
            ys.append(y_part)
            proteins += 1
            total_pairs += len(y_part)
        if total_pairs >= args.max_train_pairs:
            break
    if not xs:
        raise ValueError("No usable contact training pairs were collected")
    return np.vstack(xs), np.concatenate(ys), proteins


def safe_metric(metric, labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(metric(labels, scores))
    except ValueError:
        return float("nan")


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(max(1, int(k)), len(labels))
    selected = np.argpartition(-scores, kth=k - 1)[:k]
    return float(labels[selected].mean())


def score_pairs(clf, reps: np.ndarray, pairs: np.ndarray, batch_size: int) -> np.ndarray:
    scores = []
    for start in range(0, len(pairs), batch_size):
        features = pair_features(reps, pairs[start : start + batch_size])
        scores.append(clf.predict_proba(features)[:, 1].astype(np.float32))
    return np.concatenate(scores)


def evaluate_test_set(
    model,
    alphabet,
    clf,
    manifest: Path,
    args: argparse.Namespace,
    device: torch.device,
    external_npz: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if external_npz:
        dataset = ExternalNPZContactDataset(manifest, alphabet, args.max_residues)
    else:
        dataset = TarContactStructureDataset(
            manifest,
            args.structure_tar,
            alphabet,
            "test",
            args.max_residues,
            None,
        )
    protein_rows: list[dict[str, object]] = []
    for index_group in batched_indices(len(dataset), args.batch_size):
        examples = [dataset[index] for index in index_group]
        reps_list = embed_examples(model, examples, device, args.amp)
        for example, reps in zip(examples, reps_list):
            pairs, labels = contact_pairs(example, args.contact_cutoff, args.min_seq_sep)
            if not len(labels):
                continue
            scores = score_pairs(clf, reps, pairs, args.pair_batch_size)
            length = min(len(str(example["sequence"])), args.max_residues)
            protein_rows.append(
                {
                    "protein_id": str(example["protein_id"]),
                    "sequence_length": length,
                    "valid_pairs": int(len(labels)),
                    "contacts": int(labels.sum()),
                    "precision_at_l": precision_at_k(labels, scores, length),
                    "precision_at_l2": precision_at_k(labels, scores, max(1, length // 2)),
                    "average_precision": safe_metric(average_precision_score, labels, scores),
                    "auroc": safe_metric(roc_auc_score, labels, scores),
                }
            )
    if not protein_rows:
        raise ValueError(f"No usable test proteins in {manifest}")
    summary = {
        "test_proteins": len(protein_rows),
        "test_pairs": int(sum(int(row["valid_pairs"]) for row in protein_rows)),
        "test_contacts": int(sum(int(row["contacts"]) for row in protein_rows)),
    }
    for metric in ["precision_at_l", "precision_at_l2", "average_precision", "auroc"]:
        summary[f"{metric}_mean"] = float(np.nanmean([float(row[metric]) for row in protein_rows]))
    return summary, protein_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(seed_rows: list[dict[str, object]], baseline_label: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["test_set"]), str(row["feature_set"]))].append(row)
    baseline_by_test = {
        test_set: np.nanmean([float(row["precision_at_l_mean"]) for row in rows])
        for (test_set, model), rows in grouped.items()
        if model == baseline_label
    }
    output = []
    for (test_set, model), rows in grouped.items():
        result: dict[str, object] = {"test_set": test_set, "feature_set": model, "seeds": len(rows)}
        for metric in ["precision_at_l_mean", "precision_at_l2_mean", "average_precision_mean", "auroc_mean"]:
            values = np.asarray([float(row[metric]) for row in rows])
            result[f"{metric}_mean"] = float(np.nanmean(values))
            result[f"{metric}_std"] = float(np.nanstd(values))
        result["delta_precision_at_l_vs_esm2"] = float(
            result["precision_at_l_mean_mean"] - baseline_by_test.get(test_set, float("nan"))
        )
        output.append(result)
    output.sort(key=lambda row: (str(row["test_set"]), -float(row["precision_at_l_mean_mean"])))
    return output


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    seeds = [int(seed) for seed in args.eval_seeds.split(",") if seed.strip()]
    test_manifests = [(label, path, False) for label, path in parse_labeled_paths(args.test_manifest)]
    test_manifests.extend(
        (label, path, True) for label, path in parse_labeled_paths(args.external_test_manifest)
    )
    if not test_manifests:
        raise ValueError("Provide at least one --test-manifest or --external-test-manifest")
    seed_rows: list[dict[str, object]] = []
    all_protein_rows: list[dict[str, object]] = []

    for spec in parse_model_specs(args.model_spec):
        label = spec["label"]
        started = time.time()
        print(f"Loading {label} on {device}", flush=True)
        model, alphabet = load_model(spec, device)
        model.eval()
        x_train, y_train, train_proteins = collect_training_matrix(model, alphabet, args, device)
        print(
            f"{label}: train_pairs={len(y_train)} train_proteins={train_proteins} "
            f"positive_rate={float(y_train.mean()):.4f}",
            flush=True,
        )
        for seed in seeds:
            clf = make_pipeline(
                StandardScaler(),
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    random_state=seed,
                    max_iter=args.classifier_max_iter,
                    tol=1e-3,
                    n_jobs=-1,
                ),
            )
            clf.fit(x_train, y_train)
            for test_label, manifest, external_npz in test_manifests:
                metrics, protein_rows = evaluate_test_set(
                    model, alphabet, clf, manifest, args, device, external_npz=external_npz
                )
                metrics.update(
                    {
                        "test_set": test_label,
                        "feature_set": label,
                        "seed": seed,
                        "train_pairs": len(y_train),
                        "train_proteins": train_proteins,
                    }
                )
                seed_rows.append(metrics)
                for row in protein_rows:
                    row.update({"test_set": test_label, "feature_set": label, "seed": seed})
                    all_protein_rows.append(row)
                print(
                    f"{label} seed={seed} test={test_label} P@L={metrics['precision_at_l_mean']:.6f} "
                    f"AUPRC={metrics['average_precision_mean']:.6f}",
                    flush=True,
                )
        print(f"{label}: elapsed={time.time() - started:.1f}s", flush=True)
        del model, alphabet, x_train, y_train
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = summarize(seed_rows, args.baseline_label)
    write_csv(args.out_dir / "contact_probe_seed_metrics.csv", seed_rows)
    write_csv(args.out_dir / "contact_probe_per_protein.csv", all_protein_rows)
    write_csv(args.out_dir / "contact_probe_summary.csv", summary_rows)
    metadata = {
        "train_manifest": str(args.train_manifest),
        "test_manifests": {
            label: {"path": str(path), "format": "external_npz" if external_npz else "af2_tar"}
            for label, path, external_npz in test_manifests
        },
        "structure_tar": str(args.structure_tar),
        "pair_feature": "absolute difference plus elementwise product",
        "frozen_encoder": True,
        "probe": "StandardScaler + class-balanced SGD logistic regression",
        "contact_cutoff_angstrom": args.contact_cutoff,
        "minimum_sequence_separation": args.min_seq_sep,
        "max_train_proteins": args.max_train_proteins,
        "max_train_pairs": args.max_train_pairs,
        "model_specs": parse_model_specs(args.model_spec),
    }
    (args.out_dir / "contact_probe_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
