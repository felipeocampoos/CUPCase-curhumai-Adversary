"""Smoke test a local Hugging Face judge path with the CUPCase evaluators.

This validates the native `huggingface_local` provider end to end. It performs
one 1-case evaluator run through either the MCQ or refined free-text runner.

Example:
    HUGGINGFACE_LOCAL_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
    python huggingface_local_smoke.py --task mcq
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from refinement.refiner import JudgeProvider


DEFAULT_MCQ_INPUT = "datasets/DiagnosisMedQA_eval_20_first10.csv"
DEFAULT_FREE_INPUT = "datasets/CUPCASE_RTEST_eval_20.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test Hugging Face local evaluator path")
    parser.add_argument(
        "--task",
        type=str,
        default="mcq",
        choices=["mcq", "free_text"],
        help="Which evaluator to exercise",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional custom input CSV",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("HUGGINGFACE_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        help="Hugging Face model id to load locally",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="baseline",
        help="Evaluator variant to run",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use for the evaluator subprocess",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep smoke outputs under output/huggingface_local_smoke",
    )
    return parser.parse_args()


def preflight_or_raise(model: str) -> None:
    print(f"Hugging Face local model: {model}")
    print("This smoke run will download the model into a writable local cache if needed.")


def build_command(args: argparse.Namespace, output_path: Path) -> list[str]:
    input_path = args.input
    if input_path is None:
        input_path = DEFAULT_MCQ_INPUT if args.task == "mcq" else DEFAULT_FREE_INPUT

    if args.task == "mcq":
        return [
            args.python,
            "gpt_qa_eval_refined.py",
            "--provider",
            JudgeProvider.HUGGINGFACE_LOCAL.value,
            "--model",
            args.model,
            "--variant",
            args.variant,
            "--input",
            input_path,
            "--output",
            str(output_path),
            "--n-batches",
            "1",
            "--batch-size",
            "1",
            "--api-delay",
            "0",
            "--retry-attempts",
            "1",
            "--retry-delay",
            "0",
        ]

    return [
        args.python,
        "gpt_free_text_eval_refined.py",
        "--provider",
        JudgeProvider.HUGGINGFACE_LOCAL.value,
        "--model",
        args.model,
        "--variant",
        args.variant,
        "--input",
        input_path,
        "--output-dir",
        str(output_path),
        "--n-batches",
        "1",
        "--batch-size",
        "1",
    ]


def main() -> int:
    args = parse_args()

    preflight_or_raise(args.model)

    output_root = Path("output/huggingface_local_smoke")
    output_path = (
        output_root / "mcq_results.csv"
        if args.task == "mcq"
        else output_root / "free_text"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    cmd = build_command(args, output_path)
    print("Running evaluator smoke command:")
    print(" ".join(cmd))

    env = os.environ.copy()
    env.setdefault("HUGGINGFACE_LOCAL_MODEL", args.model)

    completed = subprocess.run(cmd, check=False, env=env)
    if completed.returncode != 0:
        print(f"Smoke run failed with exit code {completed.returncode}")
        return completed.returncode

    if not args.keep_output:
        if output_path.is_file():
            output_path.unlink(missing_ok=True)
        elif output_path.is_dir():
            for child in output_path.iterdir():
                child.unlink(missing_ok=True)
            output_path.rmdir()

    print("Smoke run succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
