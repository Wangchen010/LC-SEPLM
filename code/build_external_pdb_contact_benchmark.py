#!/usr/bin/env python3
"""Build a recent-PDB contact benchmark and remove homologs of the training set."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import subprocess
import textwrap
import time
from pathlib import Path

import numpy as np
import requests
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"


def as_list(value):
    return value if isinstance(value, list) else [value]


def normalized_sequence(value: str) -> str:
    sequence = re.sub(r"\([^)]*\)", "X", value)
    sequence = re.sub(r"\s+", "", sequence).upper()
    return "".join(residue if residue in "ACDEFGHIKLMNPQRSTVWYBXZUOJ" else "X" for residue in sequence)


def search_recent_entries(release_from: str, release_to: str, rows: int, timeout: int) -> list[str]:
    payload = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_accession_info.initial_release_date",
                "operator": "range",
                "value": {
                    "from": f"{release_from}T00:00:00Z",
                    "to": f"{release_to}T23:59:59Z",
                    "include_lower": True,
                    "include_upper": True,
                },
            },
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}],
        },
    }
    response = requests.post(SEARCH_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    return [row["identifier"].upper() for row in response.json().get("result_set", [])]


def download_cif(session: requests.Session, pdb_id: str, path: Path, timeout: int) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    for attempt in range(3):
        try:
            response = session.get(DOWNLOAD_URL.format(pdb_id=pdb_id), timeout=timeout)
            response.raise_for_status()
            path.write_bytes(response.content)
            return
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def entity_sequences(cif: dict[str, object]) -> dict[str, str]:
    entity_ids = as_list(cif.get("_entity_poly.entity_id", []))
    entity_types = as_list(cif.get("_entity_poly.type", []))
    sequences = as_list(cif.get("_entity_poly.pdbx_seq_one_letter_code_can", []))
    return {
        str(entity_id): normalized_sequence(str(sequence))
        for entity_id, entity_type, sequence in zip(entity_ids, entity_types, sequences)
        if str(entity_type).lower().startswith("polypeptide")
    }


def extract_best_chain(cif_path: Path, min_length: int, max_length: int, min_coverage: float):
    with gzip.open(cif_path, "rt", encoding="utf-8", errors="replace") as handle:
        cif = MMCIF2Dict(handle)
    entities = entity_sequences(cif)
    asym_ids = as_list(cif.get("_struct_asym.id", []))
    asym_entities = as_list(cif.get("_struct_asym.entity_id", []))
    chain_sequences = {
        str(asym_id): entities[str(entity_id)]
        for asym_id, entity_id in zip(asym_ids, asym_entities)
        if str(entity_id) in entities
    }
    if not chain_sequences:
        return None

    groups = as_list(cif.get("_atom_site.group_PDB", []))
    atom_names = as_list(cif.get("_atom_site.label_atom_id", []))
    atom_chains = as_list(cif.get("_atom_site.label_asym_id", []))
    atom_positions = as_list(cif.get("_atom_site.label_seq_id", []))
    xs = as_list(cif.get("_atom_site.Cartn_x", []))
    ys = as_list(cif.get("_atom_site.Cartn_y", []))
    zs = as_list(cif.get("_atom_site.Cartn_z", []))
    model_numbers = as_list(cif.get("_atom_site.pdbx_PDB_model_num", ["1"] * len(groups)))
    alt_ids = as_list(cif.get("_atom_site.label_alt_id", ["."] * len(groups)))

    coordinates = {
        chain: np.full((len(sequence), 3), np.nan, dtype=np.float32)
        for chain, sequence in chain_sequences.items()
        if min_length <= len(sequence) <= max_length
    }
    for group, atom, chain, position, x, y, z, model, alt in zip(
        groups, atom_names, atom_chains, atom_positions, xs, ys, zs, model_numbers, alt_ids
    ):
        if group != "ATOM" or atom != "CA" or str(model) != "1" or str(alt) not in {".", "?", "A"}:
            continue
        if chain not in coordinates or str(position) in {".", "?"}:
            continue
        index = int(position) - 1
        if 0 <= index < len(coordinates[chain]) and not np.isfinite(coordinates[chain][index]).all():
            coordinates[chain][index] = (float(x), float(y), float(z))

    candidates = []
    for chain, coords in coordinates.items():
        coverage = float(np.isfinite(coords).all(axis=1).mean())
        if coverage >= min_coverage:
            candidates.append((coverage, len(chain_sequences[chain]), chain, chain_sequences[chain], coords))
    if not candidates:
        return None
    _, _, chain, sequence, coords = max(candidates)
    return chain, sequence, coords, float(np.isfinite(coords).all(axis=1).mean())


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_fasta(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f">{row['accession']}\n")
            handle.write("\n".join(textwrap.wrap(str(row["sequence"]), 80)) + "\n")


def filter_training_homologs(args, rows: list[dict[str, object]], raw_fasta: Path) -> list[dict[str, object]]:
    hits_path = args.out_dir / "pdb_vs_train_identity30_qcov80.tsv"
    tmp_dir = args.out_dir / "mmseqs_tmp_train_filter"
    cmd = [
        str(args.mmseqs),
        "easy-search",
        str(raw_fasta),
        str(args.train_fasta),
        str(hits_path),
        str(tmp_dir),
        "--min-seq-id",
        str(args.max_train_identity),
        "-c",
        str(args.min_query_coverage),
        "--cov-mode",
        "2",
        "-s",
        str(args.sensitivity),
        "--threads",
        str(args.threads),
        "--format-output",
        "query,target,pident,alnlen,qcov,tcov,evalue,bits",
    ]
    print("Run " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    hit_ids = set()
    if hits_path.is_file():
        with hits_path.open(encoding="utf-8") as handle:
            hit_ids = {line.split("\t", 1)[0] for line in handle if line.strip()}
    return [row for row in rows if row["accession"] not in hit_ids]


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--release-from", default="2025-01-01")
    parser.add_argument("--release-to", default="2026-07-31")
    parser.add_argument("--search-rows", type=int, default=3000)
    parser.add_argument("--target-raw-chains", type=int, default=1000)
    parser.add_argument("--target-filtered-chains", type=int, default=300)
    parser.add_argument("--min-length", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-coordinate-coverage", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--train-fasta",
        type=Path,
        required=True,
        help="FASTA of the training sequences used for identity filtering.",
    )
    parser.add_argument("--mmseqs", type=Path, default=Path("mmseqs"), help="MMseqs2 executable.")
    parser.add_argument("--max-train-identity", type=float, default=0.3)
    parser.add_argument("--min-query-coverage", type=float, default=0.8)
    parser.add_argument("--sensitivity", type=float, default=7.5)
    parser.add_argument("--threads", type=int, default=24)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = args.out_dir / "mmcif"
    npz_dir = args.out_dir / "chains_npz"
    cif_dir.mkdir(exist_ok=True)
    npz_dir.mkdir(exist_ok=True)

    pdb_ids = search_recent_entries(args.release_from, args.release_to, args.search_rows, args.timeout)
    random.Random(args.seed).shuffle(pdb_ids)
    session = requests.Session()
    rows: list[dict[str, object]] = []
    failures = 0
    for index, pdb_id in enumerate(pdb_ids, start=1):
        if len(rows) >= args.target_raw_chains:
            break
        cif_path = cif_dir / f"{pdb_id}.cif.gz"
        try:
            download_cif(session, pdb_id, cif_path, args.timeout)
            chain_record = extract_best_chain(
                cif_path, args.min_length, args.max_length, args.min_coordinate_coverage
            )
            if chain_record is None:
                continue
            chain, sequence, coords, coverage = chain_record
            accession = f"{pdb_id}_{chain}"
            npz_path = npz_dir / f"{accession}.npz"
            np.savez_compressed(npz_path, coords=coords)
            rows.append(
                {
                    "accession": accession,
                    "pdb_id": pdb_id,
                    "chain_id": chain,
                    "split": "test",
                    "sequence": sequence,
                    "sequence_length": len(sequence),
                    "coordinate_coverage": coverage,
                    "npz_path": str(npz_path.resolve()),
                }
            )
        except Exception as exc:
            failures += 1
            print(f"Skip {pdb_id}: {exc}", flush=True)
        if index % 100 == 0:
            print(f"Scanned={index} accepted={len(rows)} failures={failures}", flush=True)
    if not rows:
        raise RuntimeError("No usable PDB protein chains were collected")

    raw_manifest = args.out_dir / "recent_pdb_raw.csv"
    raw_fasta = args.out_dir / "recent_pdb_raw.fasta"
    write_manifest(raw_manifest, rows)
    write_fasta(raw_fasta, rows)
    filtered = filter_training_homologs(args, rows, raw_fasta)
    random.Random(args.seed).shuffle(filtered)
    filtered = sorted(filtered[: args.target_filtered_chains], key=lambda row: str(row["accession"]))
    if not filtered:
        raise RuntimeError("Homology filtering removed every external PDB chain")
    filtered_manifest = args.out_dir / "recent_pdb_identity_lt30_qcov80.csv"
    filtered_fasta = args.out_dir / "recent_pdb_identity_lt30_qcov80.fasta"
    write_manifest(filtered_manifest, filtered)
    write_fasta(filtered_fasta, filtered)
    metadata = {
        "source": "RCSB Protein Data Bank experimental structures",
        "release_window": [args.release_from, args.release_to],
        "search_entries_returned": len(pdb_ids),
        "raw_chains": len(rows),
        "filtered_chains": len(filtered),
        "training_sequences": str(args.train_fasta),
        "filter": {
            "maximum_sequence_identity": args.max_train_identity,
            "minimum_query_coverage": args.min_query_coverage,
            "mmseqs_sensitivity": args.sensitivity,
        },
        "length_range": [args.min_length, args.max_length],
        "minimum_coordinate_coverage": args.min_coordinate_coverage,
        "raw_manifest": str(raw_manifest),
        "filtered_manifest": str(filtered_manifest),
        "failed_download_or_parse": failures,
    }
    metadata_path = args.out_dir / "benchmark_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
