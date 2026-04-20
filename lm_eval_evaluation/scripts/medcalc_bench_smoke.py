"""Smoke test MedCalc-Bench-Verified via lm_eval."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test MedCalc-Bench integration for lm_eval"
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--model", type=str, default="hf")
    parser.add_argument(
        "--model-args",
        type=str,
        default="pretrained=sshleifer/tiny-gpt2",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--method", type=str, default="direct")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--keep-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path("output/medcalc_bench_smoke")
    cmd = [
        args.python,
        "scripts/run_medcalc_bench.py",
        "--model",
        args.model,
        "--model-args",
        args.model_args,
        "--device",
        args.device,
        "--split",
        args.split,
        "--method",
        args.method,
        "--sample-size",
        str(args.sample_size),
        "--batch-size",
        "1",
        "--output-dir",
        str(output_root),
        "--log-samples",
    ]

    completed = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
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
