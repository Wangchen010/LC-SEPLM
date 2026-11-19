#!/usr/bin/env python3
"""Build residue-level PTM site labels from UniProtKB reviewed JSON entries.

The intended first-pass benchmark is deliberately conservative and reproducible:

* Human reviewed UniProtKB proteins.
* Length-limited proteins, matching the ESM2 downstream setting used elsewhere.
* Exact single-residue PTM features only.
* Candidate negatives are residue-type matched:
  - phosphorylation_sty: S/T/Y sites.
  - acetylation_k: K sites.
  - glycosylation_n: N sites for N-linked glycosylation.

The script downloads via the UniProt REST API unless a raw JSONL already exists.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/ptm_uniprot"),
    )
    parser.add_argument("--organism-id", default="9606", help="NCBI taxonomy id; 9606 = human.")
    parser.add_argument("--min-len", type=int, default=80)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def next_link(link_header: str) -> str | None:
    match = re.search(r'<([^>]+)>;\s*rel="next"', link_header or "")
    return match.group(1) if match else None


def download_uniprot_entries(args: argparse.Namespace, raw_jsonl: Path) -> list[dict]:
    if raw_jsonl.exists() and not args.force_download:
        entries = []
        with raw_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    query = (
        f"reviewed:true AND organism_id:{args.organism_id} "
        f"AND length:[{args.min_len} TO {args.max_len}] "
        "AND (ft_mod_res:* OR ft_carbohyd:*)"
    )
    params = {
        "query": query,
        "format": "json",
        "fields": (
            "accession,id,protein_name,gene_primary,organism_name,length,"
            "sequence,ft_mod_res,ft_carbohyd"
        ),
        "size": str(args.page_size),
    }

    entries: list[dict] = []
    url: str | None = UNIPROT_SEARCH_URL
    first = True
    while url:
        response = requests.get(url, params=params if first else None, timeout=90)
        response.raise_for_status()
        payload = response.json()
        entries.extend(payload.get("results", []))
        print(f"downloaded_entries={len(entries)}", flush=True)
        url = next_link(response.headers.get("Link", ""))
        first = False
        if url and args.sleep > 0:
            time.sleep(args.sleep)

    with raw_jsonl.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entries


def first_gene(entry: dict) -> str:
    genes = entry.get("genes") or []
    if not genes:
        return ""
    return str((genes[0].get("geneName") or {}).get("value", ""))


def sequence_value(entry: dict) -> str:
    sequence = entry.get("sequence") or {}
    return str(sequence.get("value", ""))


def exact_single_position(feature: dict) -> int | None:
    location = feature.get("location") or {}
    start = location.get("start") or {}
    end = location.get("end") or {}
    if start.get("modifier") not in {None, "EXACT"}:
        return None
    if end.get("modifier") not in {None, "EXACT"}:
        return None
    try:
        start_value = int(start.get("value"))
        end_value = int(end.get("value"))
    except (TypeError, ValueError):
        return None
    if start_value != end_value:
        return None
    return start_value


def task_positive_sites(entry: dict, sequence: str) -> dict[str, dict[int, list[str]]]:
    positives: dict[str, dict[int, list[str]]] = {
        "phosphorylation_sty": defaultdict(list),
        "acetylation_k": defaultdict(list),
        "glycosylation_n": defaultdict(list),
    }
    for feature in entry.get("features") or []:
        pos = exact_single_position(feature)
        if pos is None or pos < 1 or pos > len(sequence):
            continue
        residue = sequence[pos - 1]
        ftype = str(feature.get("type", ""))
        desc = str(feature.get("description", "") or "")
        desc_lower = desc.lower()
        if ftype == "Modified residue" and residue in {"S", "T", "Y"} and "phospho" in desc_lower:
            positives["phosphorylation_sty"][pos].append(desc)
        if ftype == "Modified residue" and residue == "K" and "acetyl" in desc_lower:
            positives["acetylation_k"][pos].append(desc)
        if ftype == "Glycosylation" and residue == "N" and "n-linked" in desc_lower:
            positives["glycosylation_n"][pos].append(desc)
    return positives


def candidate_positions(sequence: str) -> dict[str, Iterable[int]]:
    return {
        "phosphorylation_sty": (i + 1 for i, aa in enumerate(sequence) if aa in {"S", "T", "Y"}),
        "acetylation_k": (i + 1 for i, aa in enumerate(sequence) if aa == "K"),
        "glycosylation_n": (i + 1 for i, aa in enumerate(sequence) if aa == "N"),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = args.out_dir / "uniprot_human_reviewed_len80_512_ptm_entries.jsonl"
    proteins_csv = args.out_dir / "uniprot_human_reviewed_len80_512_ptm_proteins.csv"
    sites_csv = args.out_dir / "uniprot_human_reviewed_len80_512_ptm_sites.csv"
    summary_json = args.out_dir / "uniprot_human_reviewed_len80_512_ptm_summary.json"
    summary_md = args.out_dir / "uniprot_human_reviewed_len80_512_ptm_summary.md"

    entries = download_uniprot_entries(args, raw_jsonl)

    proteins = []
    site_rows = []
    feature_counter = Counter()
    description_counter = Counter()
    skipped_nonstandard = 0

    for entry in entries:
        accession = str(entry.get("primaryAccession", ""))
        uniprot_id = str(entry.get("uniProtkbId", ""))
        sequence = sequence_value(entry)
        if not accession or not sequence:
            continue
        if not set(sequence).issubset(STANDARD_AA):
            skipped_nonstandard += 1
            continue
        if not (args.min_len <= len(sequence) <= args.max_len):
            continue

        positives = task_positive_sites(entry, sequence)
        for feature in entry.get("features") or []:
            feature_counter[str(feature.get("type", ""))] += 1
            description_counter[(str(feature.get("type", "")), str(feature.get("description", "") or ""))] += 1

        proteins.append(
            {
                "accession": accession,
                "uniprot_id": uniprot_id,
                "gene": first_gene(entry),
                "seq_len": len(sequence),
                "sequence": sequence,
                "n_phosphorylation_sty": len(positives["phosphorylation_sty"]),
                "n_acetylation_k": len(positives["acetylation_k"]),
                "n_glycosylation_n": len(positives["glycosylation_n"]),
            }
        )

        candidates = candidate_positions(sequence)
        for task, positions in candidates.items():
            task_pos = positives[task]
            for pos in positions:
                site_rows.append(
                    {
                        "task": task,
                        "accession": accession,
                        "uniprot_id": uniprot_id,
                        "gene": first_gene(entry),
                        "seq_len": len(sequence),
                        "position": pos,
                        "residue": sequence[pos - 1],
                        "label": int(pos in task_pos),
                        "ptm_descriptions": "; ".join(sorted(set(task_pos.get(pos, [])))),
                    }
                )

    proteins_df = pd.DataFrame(proteins).sort_values(["accession"]).reset_index(drop=True)
    sites_df = pd.DataFrame(site_rows).sort_values(["task", "accession", "position"]).reset_index(drop=True)
    proteins_df.to_csv(proteins_csv, index=False)
    sites_df.to_csv(sites_csv, index=False)

    task_summary = []
    for task, frame in sites_df.groupby("task"):
        pos = frame[frame["label"] == 1]
        task_summary.append(
            {
                "task": task,
                "proteins_with_candidates": int(frame["accession"].nunique()),
                "proteins_with_positive": int(pos["accession"].nunique()),
                "candidate_sites": int(len(frame)),
                "positive_sites": int(pos.shape[0]),
                "negative_candidate_sites": int((frame["label"] == 0).sum()),
                "positive_rate": float(pos.shape[0] / len(frame)) if len(frame) else 0.0,
            }
        )

    summary = {
        "source": "UniProtKB REST API",
        "query": (
            f"reviewed:true AND organism_id:{args.organism_id} "
            f"AND length:[{args.min_len} TO {args.max_len}] "
            "AND (ft_mod_res:* OR ft_carbohyd:*)"
        ),
        "entries_downloaded": len(entries),
        "proteins_kept_standard_aa": int(len(proteins_df)),
        "proteins_skipped_nonstandard_aa": int(skipped_nonstandard),
        "feature_type_counts": dict(feature_counter.most_common()),
        "top_feature_descriptions": [
            {"type": key[0], "description": key[1], "count": count}
            for key, count in description_counter.most_common(25)
        ],
        "tasks": task_summary,
        "outputs": {
            "raw_jsonl": str(raw_jsonl),
            "proteins_csv": str(proteins_csv),
            "sites_csv": str(sites_csv),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with summary_md.open("w", encoding="utf-8") as handle:
        handle.write("# UniProt human reviewed PTM site dataset\n\n")
        handle.write(f"- Entries downloaded: {len(entries)}\n")
        handle.write(f"- Proteins kept, standard AA only: {len(proteins_df)}\n")
        handle.write(f"- Proteins skipped due to non-standard AA: {skipped_nonstandard}\n")
        handle.write(f"- Proteins table: `{proteins_csv}`\n")
        handle.write(f"- Site table: `{sites_csv}`\n\n")
        handle.write("## Task summary\n\n")
        handle.write(pd.DataFrame(task_summary).to_markdown(index=False, floatfmt=".6f"))
        handle.write("\n\n## Top feature descriptions\n\n")
        handle.write(
            pd.DataFrame(summary["top_feature_descriptions"]).to_markdown(index=False)
        )
        handle.write("\n")

    print(f"Wrote {proteins_csv}")
    print(f"Wrote {sites_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
