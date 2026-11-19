#!/usr/bin/env python3
"""Evaluate one arbitrary LRContact checkpoint on all eight protein tasks."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TASKS = {
    "ec_level1": ("classification", "prepared_full/uniprot_ec_level1_task.csv", None),
    "subcellular_location": ("classification", "prepared_full/uniprot_subcellular_location_task.csv", None),
    "membrane_binary": ("classification", "prepared_membrane_binary/uniprot_membrane_binary_task.csv", None),
    "strict_membrane_binary_seed23": (
        "classification",
        "prepared_strict_membrane_binary_seed23/uniprot_strict_membrane_binary_task.csv",
        None,
    ),
    "go_molecular_function": ("multilabel", "prepared_go_pfam/uniprot_go_molecular_function_task.csv", ";"),
    "pfam_family": ("classification", "prepared_go_pfam/uniprot_pfam_family_task.csv", None),
    "pfam_clan_remote_homology": (
        "classification",
        "prepared_go_pfam/uniprot_pfam_clan_remote_homology_task.csv",
        None,
    ),
    "hou_subcell_level3": ("multilabel", "prepared_hou_subcell_level3/hou_subcell_level3_task.csv", ";"),
}


def prepare_baseline_cache(root: Path, destination: Path, task_name: str) -> None:
    branch = root / "structure_topology_branch"
    if task_name == "hou_subcell_level3":
        source = branch / "feature_cache/hou_subcell_level3_lrcontact_500k_epoch123_maxres512"
    else:
        source = (
            branch
            / "feature_cache/esm2_t30_xattn_lr_contact_af2_swissprot_50k_seed23_maxres512"
            / task_name
        )
    destination.mkdir(parents=True, exist_ok=True)
    for source_file in source.glob("ESM2_pretrained_esm2_t30_150M_n*_maxlen512.npy"):
        target = destination / source_file.name
        if not target.exists():
            target.symlink_to(source_file.resolve())


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    branch = root / "structure_topology_branch"
    task_root = Path(args.task_root) if args.task_root is not None else (
        root / "sequence_graph_pretraining/formal_downstream_tasks"
    )
    eval_root = Path(args.eval_root) if args.eval_root is not None else task_root / "evaluations"
    eval_root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    task_names = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown = sorted(set(task_names) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    seeds = [int(seed) for seed in args.eval_seeds.split(",") if seed.strip()]
    cache_root = branch / "feature_cache" / args.cache_tag

    for task_name in task_names:
        task_type, task_csv_rel, delimiter = TASKS[task_name]
        if task_name == "hou_subcell_level3" and args.hou_task_csv is not None:
            task_csv = Path(args.hou_task_csv)
        else:
            task_csv = task_root / task_csv_rel
        if not task_csv.is_file():
            raise FileNotFoundError(task_csv)
        task_cache = cache_root / task_name
        prepare_baseline_cache(root, task_cache, task_name)
        for seed in seeds:
            out_dir = eval_root / f"{task_name}_{args.tag}_evalseed{seed}"
            summary = out_dir / "protein_task_summary.csv"
            if summary.is_file() and not args.force:
                print(f"Skip existing {summary}", flush=True)
                continue
            cmd = [
                sys.executable,
                str(root / "code/evaluate_protein_task_embeddings.py"),
                "--task-csv",
                str(task_csv),
                "--out-dir",
                str(out_dir),
                "--task-type",
                task_type,
                "--seed",
                str(seed),
                "--feature-cache-dir",
                str(task_cache),
                "--sequence-col",
                "sequence",
                "--label-col",
                "label",
                "--max-len",
                "256",
                "--batch-size",
                str(args.classifier_batch_size),
                "--n-splits",
                "5",
                "--eval-split-mode",
                args.eval_split_mode,
                "--device",
                args.device,
                "--classifier",
                "sgd_log_loss",
                "--classifier-max-iter",
                str(args.classifier_max_iter),
                "--skip-default-baselines",
                "--esm2-pretrained-model",
                f"{args.esm2_feature_name}={args.model_name}",
                "--esm2-lora-model",
                f"{args.lora_feature_name}={checkpoint}",
                "--esm2-max-residues",
                str(args.esm2_max_residues),
                "--esm2-batch-size",
                str(args.esm2_batch_size),
                "--esm2-amp",
            ]
            if delimiter:
                cmd.extend(["--label-delimiter", delimiter])
            print("Run " + " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)

    summarize_cmd = [
        sys.executable,
        str(root / "code/summarize_lrcontact_candidate_8task_multiseed.py"),
        "--root",
        str(root),
        "--tag",
        args.tag,
        "--eval-seeds",
        args.eval_seeds,
        "--tasks",
        ",".join(task_names),
        "--esm2-feature-name",
        args.esm2_feature_name,
        "--lora-feature-name",
        args.lora_feature_name,
        "--eval-root",
        str(eval_root),
    ]
    subprocess.run(summarize_cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", required=True, help="Project root containing prepared task data and outputs.")
    parser.add_argument(
        "--task-root",
        type=Path,
        default=None,
        help="Dataset root; point this at formal_downstream_tasks_no_exact_leak for strict evaluation.",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="Evaluation output root; defaults to TASK_ROOT/evaluations.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--lora-feature-name", required=True)
    parser.add_argument("--model-name", default="esm2_t30_150M_UR50D")
    parser.add_argument("--esm2-feature-name", default="ESM2_pretrained_esm2_t30_150M")
    parser.add_argument("--eval-seeds", default="7,17,23,42,101")
    parser.add_argument(
        "--eval-split-mode",
        choices=("val", "test", "val_test"),
        default="val_test",
        help="Use val during hyperparameter selection; reserve test for final evaluation.",
    )
    parser.add_argument(
        "--hou-task-csv",
        type=Path,
        default=None,
        help="Optional train/val Hou CSV derived without using the official test rows.",
    )
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--esm2-max-residues", type=int, default=512)
    parser.add_argument("--esm2-batch-size", type=int, default=32)
    parser.add_argument("--classifier-batch-size", type=int, default=256)
    parser.add_argument("--classifier-max-iter", type=int, default=1000)
    parser.add_argument("--cache-tag", default="mlm_weight_search_full8_maxres512")
    parser.add_argument("--force", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
