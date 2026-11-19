#!/usr/bin/env python3
"""Build MMseqs2 sequence-identity exclusion tables for LRContact holdouts."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import textwrap
from pathlib import Path

import pandas as pd

from evaluate_lrcontact_strict_holdouts import TASKS


def write_fasta_record(handle, record_id: str, sequence: str) -> None:
    handle.write(f">{record_id}\n")
    handle.write("\n".join(textwrap.wrap(sequence.strip().upper(), 80)))
    handle.write("\n")


def write_pretrain_fasta(manifest: Path, out_fasta: Path, max_train: int) -> int:
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with manifest.open(newline="", encoding="utf-8") as inp, out_fasta.open("w", encoding="utf-8") as out:
        for row in csv.DictReader(inp):
            if row["split"] != "train":
                continue
            sequence = row["sequence"].strip().upper()
            if not sequence:
                continue
            write_fasta_record(out, row["accession"].strip(), sequence)
            count += 1
            if count >= max_train:
                break
    return count


def write_query_fasta(task_root: Path, out_fasta: Path, map_csv: Path, tasks: list[str]) -> int:
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_fasta.open("w", encoding="utf-8") as fasta, map_csv.open("w", newline="", encoding="utf-8") as mapping:
        writer = csv.DictWriter(mapping, fieldnames=["query_id", "task", "record_id", "split"])
        writer.writeheader()
        for task in tasks:
            _, _, rel_path, _, _ = TASKS[task]
            df = pd.read_csv(task_root / rel_path)
            eval_df = df[df["split"].astype(str).str.lower().isin(["val", "test"])]
            for _, row in eval_df.iterrows():
                query_id = f"q{count}"
                write_fasta_record(fasta, query_id, str(row["sequence"]))
                writer.writerow(
                    {
                        "query_id": query_id,
                        "task": task,
                        "record_id": str(row["record_id"]).strip(),
                        "split": str(row["split"]).strip(),
                    }
                )
                count += 1
    return count


def run_mmseqs(
    mmseqs: Path,
    query_fasta: Path,
    target_fasta: Path,
    out_dir: Path,
    threshold: float,
    threads: int,
    sensitivity: float,
    coverage: float,
    cov_mode: int,
) -> Path:
    label = str(threshold).replace(".", "p")
    tmp_dir = out_dir / f"tmp_{label}"
    result_tsv = out_dir / f"identity_hits_minseqid_{label}.tsv"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(mmseqs),
        "easy-search",
        str(query_fasta),
        str(target_fasta),
        str(result_tsv),
        str(tmp_dir),
        "--min-seq-id",
        str(threshold),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "-s",
        str(sensitivity),
        "--threads",
        str(threads),
        "--format-output",
        "query,target,pident,alnlen,qcov,tcov,evalue,bits",
    ]
    print("Run " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return result_tsv


def write_exclusion_table(hits_tsv: Path, map_csv: Path, out_csv: Path) -> int:
    mapping = pd.read_csv(map_csv)
    hit_queries: set[str] = set()
    if hits_tsv.exists() and hits_tsv.stat().st_size > 0:
        with hits_tsv.open(newline="", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    hit_queries.add(line.split("\t", 1)[0])
    excluded = mapping[mapping["query_id"].isin(hit_queries)].copy()
    excluded = excluded[["task", "record_id", "split", "query_id"]].drop_duplicates()
    excluded.to_csv(out_csv, index=False)
    return len(excluded)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", required=True, help="Project data root containing prepared task splits.")
    parser.add_argument(
        "--pretrain-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path, default=Path("mmseqs"), help="MMseqs2 executable.")
    parser.add_argument("--thresholds", default="0.3,0.4")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--max-pretrain-train", type=int, default=500000)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--sensitivity", type=float, default=7.5)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--cov-mode", type=int, default=2, help="MMseqs coverage mode; 2 means query coverage.")
    args = parser.parse_args()

    root = Path(args.root)
    task_root = root / "sequence_graph_pretraining" / "formal_downstream_tasks"
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pretrain_fasta = args.out_dir / "af2_swissprot_train500k.fasta"
    query_fasta = args.out_dir / "downstream_val_test_queries.fasta"
    query_map = args.out_dir / "downstream_val_test_query_map.csv"
    pretrain_n = write_pretrain_fasta(args.pretrain_manifest, pretrain_fasta, args.max_pretrain_train)
    query_n = write_query_fasta(task_root, query_fasta, query_map, tasks)

    metadata = {
        "pretrain_fasta": str(pretrain_fasta),
        "pretrain_n": pretrain_n,
        "query_fasta": str(query_fasta),
        "query_map": str(query_map),
        "query_n": query_n,
        "thresholds": [],
        "coverage": args.coverage,
        "cov_mode": args.cov_mode,
        "sensitivity": args.sensitivity,
    }
    for threshold in [float(value) for value in args.thresholds.split(",") if value.strip()]:
        label = str(threshold).replace(".", "p")
        hits = run_mmseqs(
            args.mmseqs,
            query_fasta,
            pretrain_fasta,
            args.out_dir,
            threshold,
            args.threads,
            args.sensitivity,
            args.coverage,
            args.cov_mode,
        )
        excluded_csv = args.out_dir / f"identity_excluded_minseqid_{label}.csv"
        excluded_n = write_exclusion_table(hits, query_map, excluded_csv)
        metadata["thresholds"].append(
            {
                "threshold": threshold,
                "hits_tsv": str(hits),
                "excluded_csv": str(excluded_csv),
                "excluded_n": excluded_n,
            }
        )
        print(f"threshold={threshold} excluded={excluded_n}", flush=True)

    metadata_path = args.out_dir / "identity_exclusion_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(metadata_path, flush=True)


if __name__ == "__main__":
    main()
