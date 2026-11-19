#!/usr/bin/env python3
"""Evaluate arbitrary ESM2/LoRA checkpoints on ProteinGym single substitutions.

The score is the official ESM ``wt-marginals`` score: on the unmasked wild-type
sequence, log p(mutant amino acid) - log p(wild-type amino acid). ProteinGym's
``DMS_score`` is already direction-normalized, so raw_DMS_directionality is not
applied unless the explicitly named legacy flag is supplied.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_esm2_lora_structure_topology import (  # noqa: E402
    load_esm2_full_state_for_inference,
    load_esm2_lora_for_inference,
    load_esm2_pretrained_for_inference,
)


DEFAULT_ASSAYS = REPO_ROOT / "data/proteingym_assay_candidates.csv"
DEFAULT_OUT = REPO_ROOT / "results/proteingym_wt_marginals"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--assay-table", type=Path, default=DEFAULT_ASSAYS)
    parser.add_argument(
        "--assay-id-file",
        type=Path,
        default=None,
        help=(
            "Optional newline-delimited DMS_id allow-list. Use a fixed development "
            "allow-list during hyperparameter selection and reserve the complementary "
            "assays for final testing."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-assays", type=int, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=1022)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--model-spec",
        action="append",
        required=True,
        metavar="LABEL=KIND:VALUE",
        help=(
            "Repeatable model specification. KIND is pretrained (fair-esm model name) "
            "or lora (checkpoint path) or full_state (MODEL_NAME,CHECKPOINT)."
        ),
    )
    parser.add_argument("--baseline-label", default="ESM2_pretrained")
    parser.add_argument(
        "--apply-raw-directionality",
        action="store_true",
        help=(
            "Legacy diagnostic only. Do not use for standard ProteinGym DMS_score files, "
            "which are already direction-normalized."
        ),
    )
    return parser.parse_args()


def parse_model_specs(values: list[str]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value or ":" not in value.split("=", 1)[1]:
            raise ValueError(f"Invalid --model-spec {value!r}; expected LABEL=KIND:VALUE")
        label, payload = value.split("=", 1)
        kind, target = payload.split(":", 1)
        label, kind, target = label.strip(), kind.strip(), target.strip()
        if not label or not target or kind not in {"pretrained", "lora", "full_state"}:
            raise ValueError(f"Invalid --model-spec {value!r}")
        if kind == "full_state" and "," not in target:
            raise ValueError(f"full_state spec needs MODEL_NAME,CHECKPOINT: {value!r}")
        if label in seen:
            raise ValueError(f"Duplicate model label: {label}")
        seen.add(label)
        specs.append({"label": label, "kind": kind, "target": target})
    return specs


def load_model(spec: dict[str, str], device: torch.device):
    if spec["kind"] == "pretrained":
        return load_esm2_pretrained_for_inference(spec["target"], device)
    if spec["kind"] == "full_state":
        model_name, checkpoint = spec["target"].split(",", 1)
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        return load_esm2_full_state_for_inference(checkpoint_path, model_name, device)
    checkpoint = Path(spec["target"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return load_esm2_lora_for_inference(checkpoint, device)


def parse_single_substitution(mutation: str, sequence: str) -> tuple[int, str, str]:
    if ":" in mutation or len(mutation) < 3:
        raise ValueError("not a single substitution")
    wild_type, mutant = mutation[0], mutation[-1]
    position_text = mutation[1:-1]
    if not position_text.isdigit():
        raise ValueError("non-numeric position")
    position = int(position_text) - 1
    if position < 0 or position >= len(sequence) or sequence[position] != wild_type:
        raise ValueError("position or wild-type mismatch")
    return position, wild_type, mutant


def load_valid_mutants(row: pd.Series, alphabet) -> pd.DataFrame:
    path = Path(str(row["local_dms_path"]))
    frame = pd.read_csv(path, usecols=lambda col: col in {"mutant", "DMS_score"})
    if not {"mutant", "DMS_score"}.issubset(frame.columns):
        raise ValueError(f"Missing mutant/DMS_score columns in {path}")
    sequence = str(row["target_seq"]).strip().upper()
    records: list[dict[str, object]] = []
    for mutation, target in frame[["mutant", "DMS_score"]].itertuples(index=False, name=None):
        if pd.isna(mutation) or pd.isna(target):
            continue
        try:
            position, wild_type, mutant = parse_single_substitution(str(mutation), sequence)
        except ValueError:
            continue
        wild_type_idx = alphabet.get_idx(wild_type)
        mutant_idx = alphabet.get_idx(mutant)
        if wild_type_idx == alphabet.unk_idx or mutant_idx == alphabet.unk_idx:
            continue
        records.append(
            {
                "mutant": str(mutation),
                "position": position,
                "wild_type_idx": wild_type_idx,
                "mutant_idx": mutant_idx,
                "DMS_score": float(target),
            }
        )
    return pd.DataFrame(records)


def sequence_logits(model, alphabet, sequence: str, device: torch.device, use_amp: bool) -> torch.Tensor:
    converter = alphabet.get_batch_converter()
    _, _, tokens = converter([("protein", sequence)])
    tokens = tokens.to(device)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad(), torch.amp.autocast(
        device_type=device_type,
        enabled=use_amp and device.type == "cuda",
    ):
        if hasattr(model, "esm_forward"):
            logits, _ = model.esm_forward(tokens)
        else:
            logits = model(tokens, repr_layers=[], return_contacts=False)["logits"]
    return logits[0].float().cpu()


def score_assay(
    model,
    alphabet,
    row: pd.Series,
    device: torch.device,
    use_amp: bool,
    apply_raw_directionality: bool,
) -> dict[str, object]:
    sequence = str(row["target_seq"]).strip().upper()
    mutants = load_valid_mutants(row, alphabet)
    if len(mutants) < 3:
        raise ValueError("fewer than three valid single substitutions")
    logits = sequence_logits(model, alphabet, sequence, device, use_amp)
    log_probs = F.log_softmax(logits, dim=-1)
    offset = int(bool(alphabet.prepend_bos))
    token_positions = torch.as_tensor(mutants["position"].to_numpy(np.int64) + offset)
    wild_type_idx = torch.as_tensor(mutants["wild_type_idx"].to_numpy(np.int64))
    mutant_idx = torch.as_tensor(mutants["mutant_idx"].to_numpy(np.int64))
    predictions = (
        log_probs[token_positions, mutant_idx] - log_probs[token_positions, wild_type_idx]
    ).numpy()
    if apply_raw_directionality:
        predictions = predictions * float(row.get("raw_DMS_directionality", 1.0))
    correlation = spearmanr(predictions, mutants["DMS_score"].to_numpy(), nan_policy="omit").correlation
    return {
        "assay_id": str(row["DMS_id"]),
        "uniprot_id": str(row.get("UniProt_ID", "")),
        "coarse_selection_type": str(row.get("coarse_selection_type", "")),
        "seq_len": len(sequence),
        "n_valid_single": len(mutants),
        "raw_DMS_directionality": float(row.get("raw_DMS_directionality", 1.0)),
        "spearman": float(correlation),
        "metadata_ESM2_150M_spearman": float(row.get("ESM2_150M_spearman", np.nan)),
    }


def add_deltas_and_summarize(per_assay: pd.DataFrame, baseline_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if baseline_label not in set(per_assay["model"]):
        raise ValueError(f"Baseline label {baseline_label!r} not found")
    base = per_assay[per_assay["model"] == baseline_label][["assay_id", "spearman"]].rename(
        columns={"spearman": "baseline_spearman"}
    )
    merged = per_assay.merge(base, on="assay_id", how="left", validate="many_to_one")
    merged["delta_vs_baseline"] = merged["spearman"] - merged["baseline_spearman"]
    rows: list[dict[str, object]] = []
    for model, frame in merged.groupby("model", sort=False):
        delta = frame["delta_vs_baseline"].dropna()
        n = len(delta)
        se = float(delta.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        rows.append(
            {
                "model": model,
                "assays": n,
                "mean_spearman": float(frame["spearman"].mean()),
                "median_spearman": float(frame["spearman"].median()),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "delta_standard_error": se,
                "delta_ci95_low": float(delta.mean() - 1.96 * se) if n > 1 else float("nan"),
                "delta_ci95_high": float(delta.mean() + 1.96 * se) if n > 1 else float("nan"),
                "positive_assays": int((delta > 0).sum()),
                "positive_rate": float((delta > 0).mean()),
            }
        )
    return merged, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    specs = parse_model_specs(args.model_spec)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.assay_table)
    required = {"DMS_id", "target_seq", "local_dms_path"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Assay table is missing columns: {sorted(missing)}")
    table = table[table["target_seq"].astype(str).str.len() <= args.max_sequence_length].copy()
    if "usable_now" in table.columns:
        table = table[table["usable_now"].fillna(False).astype(bool)].copy()
    requested_assay_ids: list[str] | None = None
    if args.assay_id_file is not None:
        if not args.assay_id_file.is_file():
            raise FileNotFoundError(args.assay_id_file)
        requested_assay_ids = [
            line.strip()
            for line in args.assay_id_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not requested_assay_ids:
            raise ValueError(f"No assay IDs found in {args.assay_id_file}")
        requested_set = set(requested_assay_ids)
        available_set = set(table["DMS_id"].astype(str))
        missing_ids = sorted(requested_set - available_set)
        if missing_ids:
            preview = ", ".join(missing_ids[:10])
            raise ValueError(
                f"{len(missing_ids)} requested assay IDs are absent after table filters: {preview}"
            )
        table = table[table["DMS_id"].astype(str).isin(requested_set)].copy()
        order = {assay_id: index for index, assay_id in enumerate(requested_assay_ids)}
        table["_requested_order"] = table["DMS_id"].astype(str).map(order)
        table = table.sort_values("_requested_order").drop(columns="_requested_order")
    if args.max_assays is not None:
        table = table.head(args.max_assays)
    if table.empty:
        raise ValueError("No assays selected")

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    for spec in specs:
        label = spec["label"]
        print(f"Loading {label} on {device}", flush=True)
        model, alphabet = load_model(spec, device)
        model.eval()
        started = time.time()
        for index, (_, assay) in enumerate(table.iterrows(), start=1):
            result = score_assay(
                model,
                alphabet,
                assay,
                device,
                use_amp=args.amp,
                apply_raw_directionality=args.apply_raw_directionality,
            )
            result["model"] = label
            rows.append(result)
            if index % 20 == 0 or index == len(table):
                print(f"{label}: {index}/{len(table)} assays, elapsed={time.time() - started:.1f}s", flush=True)
        del model, alphabet
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw = pd.DataFrame(rows)
    merged, summary = add_deltas_and_summarize(raw, args.baseline_label)
    raw_path = args.out_dir / "proteingym_wt_marginals_per_assay_raw.csv"
    merged_path = args.out_dir / "proteingym_wt_marginals_per_assay.csv"
    summary_path = args.out_dir / "proteingym_wt_marginals_summary.csv"
    raw.to_csv(raw_path, index=False)
    merged.to_csv(merged_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "assay_table": str(args.assay_table),
        "assay_id_file": str(args.assay_id_file) if args.assay_id_file is not None else None,
        "requested_assay_ids": len(requested_assay_ids) if requested_assay_ids is not None else None,
        "assays": int(len(table)),
        "scoring_strategy": "wt-marginals",
        "target": "DMS_score",
        "apply_raw_directionality": bool(args.apply_raw_directionality),
        "baseline_label": args.baseline_label,
        "models": specs,
    }
    (args.out_dir / "proteingym_wt_marginals_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    md_path = args.out_dir / "proteingym_wt_marginals_summary.md"
    md_path.write_text(
        "# ProteinGym wt-marginals evaluation\n\n"
        + "ProteinGym `DMS_score` is already direction-normalized; raw directionality applied: "
        + f"`{args.apply_raw_directionality}`.\n\n"
        + summary.to_markdown(index=False, floatfmt=".6f")
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
