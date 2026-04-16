"""Prepare MedConceptsQA from Hugging Face for the CUPCase MCQ evaluator."""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_DATASET_ID = "ofir408/MedConceptsQA"
VOCABS = ["icd9cm", "icd10cm", "icd9proc", "icd10proc", "atc"]
LEVELS = ["easy", "medium", "hard"]


def subset_to_dataset_config(subset: str) -> str:
    normalized = normalize_subset_name(subset)
    if normalized in {"all", *VOCABS}:
        return "all"
    return normalized


def filter_rows_for_subset(rows: List[Dict[str, Any]], subset: str) -> List[Dict[str, Any]]:
    normalized = normalize_subset_name(subset)
    if normalized == "all":
        return rows

    if normalized in VOCABS:
        target_vocab = normalized.upper()
        return [
            row for row in rows
            if clean_text(row.get("vocab", "")).upper() == target_vocab
        ]

    vocab_name, level = normalized.split("_", maxsplit=1)
    target_vocab = vocab_name.upper()
    target_level = level.lower()
    return [
        row for row in rows
        if clean_text(row.get("vocab", "")).upper() == target_vocab
        and clean_text(row.get("level", "")).lower() == target_level
    ]


def normalize_subset_name(subset: str) -> str:
    normalized = subset.strip().lower()
    valid = {"all", *VOCABS}
    valid.update(f"{vocab}_{level}" for vocab in VOCABS for level in LEVELS)
    if normalized not in valid:
        raise ValueError(f"Unsupported subset '{subset}'")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MedConceptsQA HF dataset to CUPCase MCQ CSV"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_ID,
        help="Hugging Face dataset id",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        help="Subset/config name: all, icd10cm, icd10cm_easy, atc_hard, etc.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to load",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/generated/medconceptsqa/all_test_all.csv",
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


def load_hf_rows(dataset_id: str, subset: str, split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install with: pip install datasets"
        ) from exc

    normalized_subset = normalize_subset_name(subset)
    ds = load_dataset(dataset_id, subset_to_dataset_config(normalized_subset), split=split)
    rows = [dict(row) for row in ds]
    filtered_rows = filter_rows_for_subset(rows, normalized_subset)
    if not filtered_rows:
        raise ValueError(f"No MedConceptsQA rows found for subset '{normalized_subset}'")
    return filtered_rows


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.split())


def extract_question_stem(question: str) -> str:
    cleaned = clean_text(question)
    match = re.search(r"\sA\.\s", cleaned)
    if not match:
        return cleaned
    return cleaned[: match.start()].strip()


def answer_letter_to_index(answer_id: str) -> int:
    normalized = clean_text(answer_id).upper()
    if normalized in {"A", "B", "C", "D"}:
        return ord(normalized) - ord("A")

    try:
        numeric = int(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported answer_id '{answer_id}'") from exc

    if numeric == 0:
        return 0
    if 1 <= numeric <= 4:
        return numeric - 1
    if 0 <= numeric <= 3:
        return numeric
    raise ValueError(f"Unsupported answer_id '{answer_id}'")


def convert_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    converted: List[Dict[str, str]] = []

    for row in rows:
        options = [
            clean_text(row.get("option1", "")),
            clean_text(row.get("option2", "")),
            clean_text(row.get("option3", "")),
            clean_text(row.get("option4", "")),
        ]
        if not all(options):
            raise ValueError("Each MedConceptsQA row must include option1-option4")

        answer_index = answer_letter_to_index(str(row.get("answer_id", "")))
        final_diagnosis = options[answer_index]
        distractors = [opt for idx, opt in enumerate(options) if idx != answer_index]

        converted.append(
            {
                "id": str(row.get("question_id", row.get("id", ""))),
                "clean text": extract_question_stem(str(row.get("question", ""))),
                "final diagnosis": final_diagnosis,
                "distractor1": distractors[0],
                "distractor2": distractors[1],
                "distractor3": distractors[2],
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

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_default_output_path(subset: str, split: str, sample_size: int, seed: int) -> Path:
    normalized_subset = normalize_subset_name(subset)
    sample_label = "all" if sample_size <= 0 else f"n{sample_size}_seed{seed}"
    return Path(
        f"datasets/generated/medconceptsqa/{normalized_subset}_{split}_{sample_label}.csv"
    )


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    default_output = Path("datasets/generated/medconceptsqa/all_test_all.csv")
    if output_path == default_output:
        output_path = build_default_output_path(
            subset=args.subset,
            split=args.split,
            sample_size=args.sample_size,
            seed=args.seed,
        )

    rows = load_hf_rows(args.dataset, args.subset, args.split)
    converted = convert_rows(rows)
    converted = maybe_sample(converted, args.sample_size, args.seed)
    save_csv(converted, output_path)

    print(f"Dataset:     {args.dataset}")
    print(f"Subset:      {normalize_subset_name(args.subset)}")
    print(f"Split:       {args.split}")
    print(f"Loaded rows: {len(rows)}")
    print(f"Saved rows:  {len(converted)}")
    print(f"Output:      {output_path}")


if __name__ == "__main__":
    main()
