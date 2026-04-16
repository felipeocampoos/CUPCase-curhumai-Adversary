"""Repo-owned MedConceptsQA runner for lm_eval."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


VOCABS = ["icd9cm", "icd10cm", "icd9proc", "icd10proc", "atc"]
LEVELS = ["easy", "medium", "hard"]


def normalize_subset_name(subset: str) -> str:
    normalized = subset.strip().lower()
    valid = {"all", *VOCABS}
    valid.update(f"{vocab}_{level}" for vocab in VOCABS for level in LEVELS)
    if normalized not in valid:
        raise ValueError(f"Unsupported subset '{subset}'")
    return normalized


def subset_to_task_name(subset: str) -> str:
    normalized = normalize_subset_name(subset)
    if normalized == "all":
        return "med_concepts_qa"
    if normalized in VOCABS:
        return f"med_concepts_qa_{normalized}"
    return f"med_concepts_qa_{normalized}"


def slugify_model(model: str, model_args: str) -> str:
    base = model if not model_args else f"{model}_{model_args}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedConceptsQA via lm_eval")
    parser.add_argument("--model", type=str, default="hf")
    parser.add_argument("--model-args", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=str, default="auto")
    parser.add_argument("--subset", type=str, default="all")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="output/medconceptsqa")
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subset = normalize_subset_name(args.subset)
    task_name = subset_to_task_name(subset)
    run_root = (
        Path(args.output_dir)
        / subset
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
    print(f"Output dir: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
