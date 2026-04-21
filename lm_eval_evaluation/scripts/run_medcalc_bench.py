"""Repo-owned MedCalc-Bench-Verified runner for lm_eval and MedCalc-specific methods."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


VALID_SPLITS = {"train", "test", "one_shot"}
VALID_METHODS = {
    "direct",
    "zero_shot_cot",
    "one_shot_cot",
    "medcalc_semantic_gate",
    "medcalc_uncertainty_consistency_gate",
}


def normalize_split_name(split: str) -> str:
    normalized = split.strip().lower()
    if normalized not in VALID_SPLITS:
        raise ValueError(f"Unsupported split '{split}'")
    return normalized


def normalize_method_name(method: str) -> str:
    normalized = method.strip().lower()
    if normalized not in VALID_METHODS:
        raise ValueError(f"Unsupported method '{method}'")
    return normalized


def slugify_model(model: str, model_args: str) -> str:
    base = model if not model_args else f"{model}_{model_args}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or "model"


def split_to_task_name(split: str, method: str) -> str:
    normalized = normalize_split_name(split)
    normalized_method = normalize_method_name(method)
    if normalized_method == "direct":
        return "med_calc_bench" if normalized == "test" else f"med_calc_bench_{normalized}"
    base = f"med_calc_bench_{normalized_method}"
    return base if normalized == "test" else f"{base}_{normalized}"


def parse_model_arg(model_args: str, key: str) -> str | None:
    for part in model_args.split(","):
        stripped = part.strip()
        if not stripped or "=" not in stripped:
            continue
        current_key, value = stripped.split("=", 1)
        if current_key.strip() == key:
            return value.strip()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedCalc-Bench-Verified via lm_eval")
    parser.add_argument("--model", type=str, default="hf")
    parser.add_argument("--model-args", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=str, default="auto")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--method", type=str, default="direct")
    parser.add_argument("--output-dir", type=str, default="output/medcalc_bench")
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--icl-example-id", type=str, default=None)
    parser.add_argument("--icl-seed", type=int, default=42)
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--candidate-temperature", type=float, default=0.7)
    parser.add_argument("--verifier-risk-threshold", type=float, default=0.5)
    return parser.parse_args()


def run_lm_eval(args: argparse.Namespace, split: str, method: str, run_root: Path) -> int:
    task_name = split_to_task_name(split, method)
    env = os.environ.copy()
    env["MEDCALC_DATASET_ID"] = "nsk7153/MedCalc-Bench-Verified"
    env["MEDCALC_ICL_SEED"] = str(args.icl_seed)
    if args.icl_example_id:
        env["MEDCALC_ICL_EXAMPLE_ID"] = args.icl_example_id
    else:
        env.pop("MEDCALC_ICL_EXAMPLE_ID", None)

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
        env=env,
    )
    if completed.returncode == 0:
        print(f"Task:       {task_name}")
        print(f"Split:      {split}")
        print(f"Method:     {method}")
        print(f"Output dir: {run_root}")
    return completed.returncode


def run_medcalc_gated_method(
    args: argparse.Namespace,
    split: str,
    run_root: Path,
    method: str,
) -> int:
    pretrained = parse_model_arg(args.model_args, "pretrained")
    if args.model != "hf" or not pretrained:
        raise ValueError(
            f"{method} on the local-model surface currently requires "
            "`--model hf --model-args pretrained=<hf-model>`"
        )

    api_surface_root = Path(__file__).resolve().parents[2] / "gpt_and_med_lm_evaluation"
    cmd = [
        args.python,
        "run_medcalc_bench_free_text.py",
        "--split",
        split,
        "--provider",
        "huggingface_local",
        "--model",
        pretrained,
        "--method",
        method,
        "--output-root",
        str(run_root),
        "--seed",
        str(args.seed),
        "--icl-seed",
        str(args.icl_seed),
        "--num-candidates",
        str(args.num_candidates),
        "--candidate-temperature",
        str(args.candidate_temperature),
        "--verifier-risk-threshold",
        str(args.verifier_risk_threshold),
        "--retry-attempts",
        "1",
        "--retry-delay",
        "0",
        "--api-delay",
        "0",
    ]
    if args.sample_size > 0:
        cmd.extend(["--sample-size", str(args.sample_size)])
    if args.icl_example_id:
        cmd.extend(["--icl-example-id", args.icl_example_id])

    env = os.environ.copy()
    env["HUGGINGFACE_LOCAL_MODEL"] = pretrained

    completed = subprocess.run(
        cmd,
        cwd=api_surface_root,
        check=False,
        env=env,
    )
    if completed.returncode == 0:
        print(f"Task:       {method}")
        print(f"Split:      {split}")
        print(f"Method:     {method}")
        print(f"Output dir: {run_root}")
    return completed.returncode


def main() -> int:
    args = parse_args()
    split = normalize_split_name(args.split)
    method = normalize_method_name(args.method)
    run_root = (
        Path(args.output_dir)
        / split
        / method
        / slugify_model(args.model, args.model_args)
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_root.mkdir(parents=True, exist_ok=True)

    if method in {"medcalc_semantic_gate", "medcalc_uncertainty_consistency_gate"}:
        return run_medcalc_gated_method(args, split, run_root, method)
    return run_lm_eval(args, split, method, run_root)


if __name__ == "__main__":
    raise SystemExit(main())
