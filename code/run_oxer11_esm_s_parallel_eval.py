#!/usr/bin/env python3
"""Parallel launcher for Oxer11 / ESM-S official dataset evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRANCH_ROOT = ROOT / "results"
EVAL_SCRIPT = ROOT / "code" / "evaluate_oxer11_esm_s_datasets.py"


MODEL_SPECS = {
    "ESM2_pretrained": "ESM2_pretrained=pretrained:esm2_t30_150M_UR50D",
    "ESM_S_150M": "ESM_S_150M=full_state:esm2_t30_150M_UR50D,models/esm_s_150m.pth",
    "LC_SEPLM_epoch10": "LC_SEPLM_epoch10=lora:models/lc_seplm_epoch10.pt",
}

TASK_GROUPS = {
    "fold": "fold",
    "ec": "ec",
    "go": "go_mf,go_bp,go_cc",
}

EXPECTED_ROWS = {
    "fold": 9,  # 3 seeds x 3 fold test splits
    "ec": 9,  # 3 seeds x 3 identity splits
    "go": 27,  # 3 GO levels x 3 seeds x 3 identity splits
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task"]), str(row["split"]), str(row["model"])), []).append(row)
    out_rows: list[dict[str, object]] = []
    skip = {"task", "split", "model", "seed"}
    for (task, split, model), group in grouped.items():
        out: dict[str, object] = {"task": task, "split": split, "model": model, "seeds": len(group)}
        for key, value in group[0].items():
            if key in skip:
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                continue
            values = [float(row[key]) for row in group if row.get(key, "") != ""]
            if not values:
                continue
            out[f"{key}_mean"] = sum(values) / len(values)
            mean = out[f"{key}_mean"]
            out[f"{key}_std"] = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
        out_rows.append(out)
    return sorted(out_rows, key=lambda row: (str(row["task"]), str(row["split"]), str(row["model"])))


def job_done(out_dir: Path, group_name: str) -> bool:
    rows = read_rows(out_dir / "oxer11_esm_s_eval_raw.csv")
    return len(rows) >= EXPECTED_ROWS[group_name]


def collect(args: argparse.Namespace) -> tuple[Path, Path, int]:
    all_rows: list[dict[str, object]] = []
    for model_name in MODEL_SPECS:
        for group_name in TASK_GROUPS:
            job_dir = args.out_root / f"{group_name}__{model_name}"
            all_rows.extend(read_rows(job_dir / "oxer11_esm_s_eval_raw.csv"))
    raw_path = args.out_root / "oxer11_esm_s_eval_raw_merged.csv"
    summary_path = args.out_root / "oxer11_esm_s_eval_summary_merged.csv"
    write_csv(raw_path, all_rows)
    write_csv(summary_path, summarize(all_rows))
    return raw_path, summary_path, len(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpus", default="9,8,3,2")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used for worker jobs.")
    parser.add_argument("--out-root", type=Path, default=BRANCH_ROOT / "oxer11_esm_s_official_eval_parallel_3seed")
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=BRANCH_ROOT / "feature_cache" / "oxer11_esm_s_official_maxres550",
    )
    parser.add_argument("--seeds", default="7,17,23")
    parser.add_argument("--percents", default="95,50,30")
    parser.add_argument("--max-residues", type=int, default=550)
    parser.add_argument("--embedding-batch-size", type=int, default=12)
    parser.add_argument("--embedding-log-every", type=int, default=2048)
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--probe-lr", type=float, default=1e-4)
    parser.add_argument("--probe-dropout", type=float, default=0.2)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--data-root", type=Path, required=True, help="External Oxer11 dataset root.")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "logs").mkdir(parents=True, exist_ok=True)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("No GPUs provided")

    jobs: list[dict[str, object]] = []
    for model_name, model_spec in MODEL_SPECS.items():
        for group_name, task_spec in TASK_GROUPS.items():
            out_dir = args.out_root / f"{group_name}__{model_name}"
            if job_done(out_dir, group_name):
                print(json.dumps({"event": "skip_done", "job": out_dir.name}), flush=True)
                continue
            jobs.append(
                {
                    "model_name": model_name,
                    "model_spec": model_spec,
                    "group_name": group_name,
                    "task_spec": task_spec,
                    "out_dir": out_dir,
                }
            )

    pending = jobs[::-1]
    running: list[dict[str, object]] = []
    available_gpus = gpus[:]
    failures: list[dict[str, object]] = []
    print(
        json.dumps(
            {
                "event": "launch_start",
                "num_jobs": len(jobs),
                "gpus": gpus,
                "out_root": str(args.out_root),
                "cache": str(args.feature_cache_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    try:
        while pending or running:
            while pending and available_gpus:
                gpu = available_gpus.pop(0)
                job = pending.pop()
                out_dir = Path(job["out_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                log_path = args.out_root / "logs" / f"{out_dir.name}.log"
                cmd = [
                    args.python,
                    str(EVAL_SCRIPT),
                    "--tasks",
                    str(job["task_spec"]),
                    "--data-root",
                    str(args.data_root),
                    "--model-spec",
                    str(job["model_spec"]),
                    "--seeds",
                    args.seeds,
                    "--percents",
                    args.percents,
                    "--device",
                    f"cuda:{gpu}",
                    "--out-dir",
                    str(out_dir),
                    "--feature-cache-dir",
                    str(args.feature_cache_dir),
                    "--max-residues",
                    str(args.max_residues),
                    "--embedding-batch-size",
                    str(args.embedding_batch_size),
                    "--embedding-log-every",
                    str(args.embedding_log_every),
                    "--probe-batch-size",
                    str(args.probe_batch_size),
                    "--eval-batch-size",
                    str(args.eval_batch_size),
                    "--probe-epochs",
                    str(args.probe_epochs),
                    "--early-stop-patience",
                    str(args.early_stop_patience),
                    "--probe-lr",
                    str(args.probe_lr),
                    "--probe-dropout",
                    str(args.probe_dropout),
                ]
                handle = log_path.open("a", encoding="utf-8")
                process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
                job.update({"gpu": gpu, "process": process, "log_path": log_path, "log_handle": handle, "start_time": time.time()})
                running.append(job)
                print(
                    json.dumps(
                        {
                            "event": "job_start",
                            "job": out_dir.name,
                            "gpu": gpu,
                            "pid": process.pid,
                            "log": str(log_path),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            time.sleep(args.poll_seconds)
            still_running: list[dict[str, object]] = []
            for job in running:
                process: subprocess.Popen = job["process"]  # type: ignore[assignment]
                ret = process.poll()
                if ret is None:
                    still_running.append(job)
                    continue
                job["log_handle"].close()  # type: ignore[index, union-attr]
                available_gpus.append(str(job["gpu"]))
                elapsed = time.time() - float(job["start_time"])
                event = "job_done" if ret == 0 else "job_failed"
                payload = {
                    "event": event,
                    "job": Path(job["out_dir"]).name,
                    "gpu": job["gpu"],
                    "returncode": ret,
                    "elapsed_sec": round(elapsed, 1),
                    "log": str(job["log_path"]),
                }
                if ret != 0:
                    failures.append(payload)
                print(json.dumps(payload, ensure_ascii=False), flush=True)
                raw_path, summary_path, n_rows = collect(args)
                print(
                    json.dumps(
                        {
                            "event": "merge_progress",
                            "raw_path": str(raw_path),
                            "summary_path": str(summary_path),
                            "merged_raw_rows": n_rows,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            running = still_running
    except KeyboardInterrupt:
        print(json.dumps({"event": "keyboard_interrupt", "running": len(running)}, ensure_ascii=False), flush=True)
        for job in running:
            process: subprocess.Popen = job["process"]  # type: ignore[assignment]
            if process.poll() is None:
                process.terminate()
            job["log_handle"].close()  # type: ignore[index, union-attr]
        raise

    raw_path, summary_path, n_rows = collect(args)
    final = {
        "event": "all_done" if not failures else "done_with_failures",
        "failures": failures,
        "raw_path": str(raw_path),
        "summary_path": str(summary_path),
        "merged_raw_rows": n_rows,
    }
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
