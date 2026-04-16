"""Smoke test MedConceptsQA through the CUPCase MCQ evaluator."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test MedConceptsQA integration for the CUPCase MCQ evaluator"
    )
    parser.add_argument("--subset", type=str, default="icd10cm_easy")
    parser.add_argument("--sample-size", type=int, default=1)
    parser.add_argument("--provider", type=str, default="huggingface_local")
    parser.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--variant", type=str, default="baseline")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--keep-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path("output/experiments/medconceptsqa_smoke")

    cmd = [
        args.python,
        "run_medconceptsqa_mcq.py",
        "--subset",
        args.subset,
        "--sample-size",
        str(args.sample_size),
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--variant",
        args.variant,
        "--output-root",
        str(output_root),
        "--retry-attempts",
        "1",
        "--retry-delay",
        "0",
        "--api-delay",
        "0",
    ]

    completed = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parent,
        check=False,
    )
    if completed.returncode != 0:
        print(f"Smoke run failed with exit code {completed.returncode}")
        return completed.returncode

    if not args.keep_output and output_root.exists():
        shutil.rmtree(output_root)

    print("Smoke run succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
