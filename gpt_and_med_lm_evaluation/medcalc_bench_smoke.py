"""Smoke test MedCalc-Bench-Verified through the API-based evaluation surface."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test MedCalc-Bench integration for the API-based evaluator"
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample-size", type=int, default=1)
    parser.add_argument("--provider", type=str, default="huggingface_local")
    parser.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--method", type=str, default="direct")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--keep-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path("output/experiments/medcalc_bench_smoke")

    cmd = [
        args.python,
        "run_medcalc_bench_free_text.py",
        "--split",
        args.split,
        "--sample-size",
        str(args.sample_size),
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--method",
        args.method,
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
