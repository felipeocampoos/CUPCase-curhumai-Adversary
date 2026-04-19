"""Repo-owned MedCalc-Bench-Verified runner for lm_eval."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


VALID_SPLITS = {"train", "test", "one_shot"}


def normalize_split_name(split: str) -> str:
    normalized = split.strip().lower()
    if normalized not in VALID_SPLITS:
        raise ValueError(f"Unsupported split '{split}'")
    return normalized


def slugify_model(model: str, model_args: str) -> str:
    base = model if not model_args else f"{model}_{model_args}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or "model"


def split_to_task_name(split: str) -> str:
    normalized = normalize_split_name(split)
    if normalized == "test":
        return "med_calc_bench"
    return f"med_calc_bench_{normalized}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedCalc-Bench-Verified via lm_eval")
    parser.add_argument("--model", type=str, default="hf")
    parser.add_argument("--model-args", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=str, default="auto")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="output/medcalc_bench")
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split = normalize_split_name(args.split)
    task_name = split_to_task_name(split)
    run_root = (
        Path(args.output_dir)
        / split
        / slugify_model(args.model, args.model_args)
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        "-m",
        "lm_eval",
        "--model",
        args.model,
        "--tasks",
        task_name,
        "--batch_size",
        args.batch_size,
        "--output_path",
        str(run_root),
        "--seed",
        str(args.seed),
    ]
    if args.model_args:
        cmd.extend(["--model_args", args.model_args])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.sample_size > 0:
        cmd.extend(["--limit", str(args.sample_size)])
    if args.log_samples:
        cmd.append("--log_samples")

    completed = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    if completed.returncode != 0:
        return completed.returncode

    print(f"Task:       {task_name}")
    print(f"Split:      {split}")
    print(f"Output dir: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
