"""
Prepare DiagnosisMedQA from Hugging Face for CUPCase evaluation scripts.

This script downloads/loads the dataset using the Hugging Face datasets loader
and converts it to the CSV schema expected by the evaluation runners.
"""

import argparse
import csv
import random
from pathlib import Path
from typing import List, Dict, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DiagnosisMedQA HF dataset to CUPCase eval CSV"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="oriel9p/DiagnosisMedQA",
        help="Hugging Face dataset id",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/DiagnosisMedQA_eval.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional sample size (0 means full split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed",
    )
    return parser.parse_args()


def load_hf_rows(dataset_id: str, split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install with: pip install datasets"
        ) from exc

    ds = load_dataset(dataset_id, split=split)
    return [dict(row) for row in ds]


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.split())


def convert_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    converted: List[Dict[str, str]] = []

    for row in rows:
        converted.append(
            {
                "id": str(row.get("id", "")),
                "clean text": clean_text(row.get("clean_case_presentation", "")),
                "final diagnosis": clean_text(row.get("correct_diagnosis", "")),
                "distractor1": clean_text(row.get("distractor1", "")),
                "distractor2": clean_text(row.get("distractor2", "")),
                "distractor3": clean_text(row.get("distractor3", "")),
            }
        )

    return converted


def maybe_sample(rows: List[Dict[str, str]], sample_size: int, seed: int) -> List[Dict[str, str]]:
    if sample_size <= 0 or sample_size >= len(rows):
        return rows

    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def save_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "clean text",
        "final diagnosis",
        "distractor1",
        "distractor2",
        "distractor3",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    rows = load_hf_rows(args.dataset, args.split)
    converted = convert_rows(rows)
    converted = maybe_sample(converted, args.sample_size, args.seed)

    output_path = Path(args.output)
    save_csv(converted, output_path)

    print(f"Loaded rows: {len(rows)}")
    print(f"Saved rows:  {len(converted)}")
    print(f"Output:      {output_path}")


if __name__ == "__main__":
    main()
