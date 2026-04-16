"""Run MedConceptsQA through the CUPCase MCQ evaluator."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from gpt_qa_eval_refined import VALID_VARIANTS
from prepare_hf_medconceptsqa import (
    build_default_output_path,
    convert_rows,
    load_hf_rows,
    maybe_sample,
    normalize_subset_name,
    save_csv,
)
from refinement.refiner import JudgeProvider


def slugify_model(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MedConceptsQA via the CUPCase MCQ evaluator"
    )
    parser.add_argument("--dataset", type=str, default="ofir408/MedConceptsQA")
    parser.add_argument("--subset", type=str, default="all")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--provider",
        type=str,
        default=JudgeProvider.OPENAI.value,
        choices=[provider.value for provider in JudgeProvider],
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--variant",
        type=str,
        default="baseline",
        choices=VALID_VARIANTS,
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument(
        "--output-root",
        type=str,
        default="output/experiments/medconceptsqa",
        help="Base directory for evaluator artifacts",
    )
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=60.0)
    parser.add_argument("--api-delay", type=float, default=1.0)
    return parser.parse_args()


def build_results_path(
    output_root: str,
    subset: str,
    provider: str,
    variant: str,
    model: str,
    sample_size: int,
    seed: int,
) -> Path:
    sample_label = "all" if sample_size <= 0 else f"n{sample_size}_seed{seed}"
    return (
        Path(output_root)
        / normalize_subset_name(subset)
        / provider
        / "mcq"
        / variant
        / slugify_model(model)
        / sample_label
        / "results.csv"
    )


def main() -> int:
    args = parse_args()
    provider = JudgeProvider(args.provider)
    model = args.model or provider.default_model

    rows = load_hf_rows(args.dataset, args.subset, args.split)
    converted = convert_rows(rows)
    converted = maybe_sample(converted, args.sample_size, args.seed)
    if not converted:
        raise ValueError("No rows available to evaluate")

    generated_csv_path = build_default_output_path(
        subset=args.subset,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    save_csv(converted, generated_csv_path)

    results_path = build_results_path(
        output_root=args.output_root,
        subset=args.subset,
        provider=args.provider,
        variant=args.variant,
        model=model,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        "gpt_qa_eval_refined.py",
        "--dataset",
        "custom",
        "--input",
        str(generated_csv_path),
        "--output",
        str(results_path),
        "--provider",
        args.provider,
        "--model",
        model,
        "--variant",
        args.variant,
        "--n-batches",
        "1",
        "--batch-size",
        str(len(converted)),
        "--random-seed",
        str(args.seed),
        "--sampling-mode",
        "unique",
        "--retry-attempts",
        str(args.retry_attempts),
        "--retry-delay",
        str(args.retry_delay),
        "--api-delay",
        str(args.api_delay),
    ]

    completed = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parent,
        check=False,
    )
    if completed.returncode != 0:
        return completed.returncode

    print(f"Generated CSV: {generated_csv_path}")
    print(f"Results CSV:   {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
