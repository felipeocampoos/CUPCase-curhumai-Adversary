"""Smoke test an OpenAI-compatible judge endpoint with the CUPCase evaluators.

This validates the same provider/model path used for Qwen-style local or cluster
serving setups. It performs:

1. Endpoint preflight (`/models` via OpenAI client)
2. A 1-case evaluator run through either the MCQ or refined free-text runner

Example:
    OPENAI_COMPATIBLE_BASE_URL=http://127.0.0.1:8000/v1 \
    OPENAI_COMPATIBLE_MODEL=Qwen/Qwen3.5-0.8B \
    python openai_compatible_smoke.py --task mcq
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from refinement.refiner import JudgeProvider, create_client


DEFAULT_MCQ_INPUT = "datasets/DiagnosisMedQA_eval_20_first10.csv"
DEFAULT_FREE_INPUT = "datasets/CUPCASE_RTEST_eval_20.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test OpenAI-compatible evaluator path")
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
        default=os.environ.get("OPENAI_COMPATIBLE_MODEL", "openai-compatible-model"),
        help="Model name exposed by the OpenAI-compatible endpoint",
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
        help="Keep smoke outputs under output/openai_compatible_smoke",
    )
    return parser.parse_args()


def preflight_or_raise(model: str) -> None:
    base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
    if not base_url:
        raise ValueError("OPENAI_COMPATIBLE_BASE_URL is not set")

    client = create_client(provider=JudgeProvider.OPENAI_COMPATIBLE)

    try:
        models = client.models.list()
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        raise RuntimeError(
            f"Endpoint preflight failed for {base_url}: {exc}"
        ) from exc

    available = []
    for item in getattr(models, "data", []) or []:
        item_id = getattr(item, "id", None)
        if item_id:
            available.append(item_id)

    print(f"Endpoint OK: {base_url}")
    if available:
        print(f"Available models: {', '.join(available)}")
        if model not in available:
            print(f"Requested model not advertised by /models: {model}")
    else:
        print("Endpoint responded, but no models were listed.")


def build_command(args: argparse.Namespace, output_path: Path) -> list[str]:
    input_path = args.input
    if input_path is None:
        input_path = DEFAULT_MCQ_INPUT if args.task == "mcq" else DEFAULT_FREE_INPUT

    if args.task == "mcq":
        return [
            args.python,
            "gpt_qa_eval_refined.py",
            "--provider",
            JudgeProvider.OPENAI_COMPATIBLE.value,
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
        JudgeProvider.OPENAI_COMPATIBLE.value,
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

    output_root = Path("output/openai_compatible_smoke")
    output_path = (
        output_root / "mcq_results.csv"
        if args.task == "mcq"
        else output_root / "free_text"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    cmd = build_command(args, output_path)
    print("Running evaluator smoke command:")
    print(" ".join(cmd))

    completed = subprocess.run(cmd, check=False)
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
