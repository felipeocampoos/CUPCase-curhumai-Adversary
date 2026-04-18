"""Prepare MedCalc-Bench-Verified from Hugging Face for local evaluation."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_DATASET_ID = "nsk7153/MedCalc-Bench-Verified"
VALID_SPLITS = {"train", "test", "one_shot"}
DEFAULT_OUTPUT = "datasets/generated/medcalc_bench/test_all.csv"


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.split())


def normalize_split_name(split: str) -> str:
    normalized = split.strip().lower()
    if normalized not in VALID_SPLITS:
        raise ValueError(f"Unsupported split '{split}'")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MedCalc-Bench-Verified HF dataset to a normalized CSV"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_ID,
        help="Hugging Face dataset id",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to load: train, test, or one_shot",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
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

    normalized_split = normalize_split_name(split)
    ds = load_dataset(dataset_id, split=normalized_split)
    rows = [dict(row) for row in ds]
    if not rows:
        raise ValueError(f"No MedCalc-Bench rows found for split '{normalized_split}'")
    return rows


def build_row_id(row: Dict[str, Any]) -> str:
    note_id = clean_text(row.get("Note ID", ""))
    row_number = clean_text(row.get("Row Number", ""))
    if note_id and row_number:
        return f"{note_id}:{row_number}"
    if row_number:
        return row_number
    return note_id


def convert_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    converted: List[Dict[str, str]] = []
    required = [
        "Patient Note",
        "Question",
        "Relevant Entities",
        "Ground Truth Answer",
        "Ground Truth Explanation",
        "Calculator Name",
        "Category",
        "Output Type",
        "Note Type",
    ]

    for row in rows:
        missing = [field for field in required if not clean_text(row.get(field, ""))]
        if missing:
            raise ValueError(f"Missing required MedCalc fields: {', '.join(missing)}")

        converted.append(
            {
                "id": build_row_id(row),
                "case_text": clean_text(row.get("Patient Note", "")),
                "question": clean_text(row.get("Question", "")),
                "relevant_entities": clean_text(row.get("Relevant Entities", "")),
                "ground_truth_answer": clean_text(row.get("Ground Truth Answer", "")),
                "lower_limit": clean_text(row.get("Lower Limit", "")),
                "upper_limit": clean_text(row.get("Upper Limit", "")),
                "ground_truth_explanation": clean_text(
                    row.get("Ground Truth Explanation", "")
                ),
                "calculator_name": clean_text(row.get("Calculator Name", "")),
                "category": clean_text(row.get("Category", "")),
                "output_type": clean_text(row.get("Output Type", "")),
                "note_type": clean_text(row.get("Note Type", "")),
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
        "case_text",
        "question",
        "relevant_entities",
        "ground_truth_answer",
        "lower_limit",
        "upper_limit",
        "ground_truth_explanation",
        "calculator_name",
        "category",
        "output_type",
        "note_type",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_default_output_path(split: str, sample_size: int, seed: int) -> Path:
    normalized_split = normalize_split_name(split)
    sample_label = "all" if sample_size <= 0 else f"n{sample_size}_seed{seed}"
    return Path(f"datasets/generated/medcalc_bench/{normalized_split}_{sample_label}.csv")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path == Path(DEFAULT_OUTPUT):
        output_path = build_default_output_path(
            split=args.split,
            sample_size=args.sample_size,
            seed=args.seed,
        )

    rows = load_hf_rows(args.dataset, args.split)
    converted = convert_rows(rows)
    sampled = maybe_sample(converted, args.sample_size, args.seed)
    save_csv(sampled, output_path)

    print(f"Wrote {len(sampled)} rows to {output_path}")


if __name__ == "__main__":
    main()
