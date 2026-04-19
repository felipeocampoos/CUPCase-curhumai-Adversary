from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional

import datasets
import pandas as pd


NUMERIC_OUTPUT_TYPES = {"decimal", "integer"}


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def decimal_from_string(value: str) -> Optional[Decimal]:
    cleaned = str(value).strip().replace(",", "")
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def canonicalize_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal("1")))
    return format(normalized, "f").rstrip("0").rstrip(".") or "0"


def extract_numeric_candidates(text: str) -> List[Decimal]:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    parsed_values: List[Decimal] = []
    for candidate in matches:
        parsed = decimal_from_string(candidate)
        if parsed is not None:
            parsed_values.append(parsed)
    return parsed_values


def extract_answer_numeric(text: str) -> Optional[Decimal]:
    answer_segments = []
    for pattern in (
        r"(?is)\bfinal answer\b\s*[:=-]?\s*(.+)",
        r"(?is)\banswer\b\s*[:=-]?\s*(.+)",
    ):
        match = re.search(pattern, text)
        if match:
            answer_segments.append(match.group(1).strip())

    answer_segments.append(text.strip())
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if first_line:
        answer_segments.append(first_line)

    for segment in answer_segments:
        candidates = extract_numeric_candidates(segment)
        if candidates:
            return candidates[0]

    return None


def extract_last_numeric(text: str) -> Optional[Decimal]:
    matches = extract_numeric_candidates(text)
    if not matches:
        return None
    return matches[-1]


def normalize_date_like(value: str) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return normalize_text(value)
    return parsed.strftime("%Y-%m-%d")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        return {
            "id": f"{doc.get('Note ID', '')}:{doc.get('Row Number', '')}".strip(":"),
            "case_text": doc["Patient Note"],
            "question": doc["Question"],
            "relevant_entities": doc.get("Relevant Entities", ""),
            "ground_truth_answer": str(doc["Ground Truth Answer"]),
            "lower_limit": str(doc.get("Lower Limit", "")),
            "upper_limit": str(doc.get("Upper Limit", "")),
            "ground_truth_explanation": doc.get("Ground Truth Explanation", ""),
            "calculator_name": doc.get("Calculator Name", ""),
            "category": doc.get("Category", ""),
            "output_type": normalize_text(doc.get("Output Type", "")),
            "note_type": doc.get("Note Type", ""),
        }

    return dataset.map(_process_doc)


def doc_to_text(doc: dict) -> str:
    instruction = (
        "Read the patient note and answer the medical calculation question. "
        "Return only the final answer with no explanation."
    )
    if doc["output_type"] in NUMERIC_OUTPUT_TYPES:
        instruction += " If the answer is numeric, return only the final numeric value."
    elif doc["output_type"] == "date":
        instruction += " If the answer is a date, return only the final date."

    words = str(doc["case_text"]).split()
    truncated_case_text = " ".join(words[:250]) if len(words) > 250 else doc["case_text"]
    entities_block = (
        f"Relevant entities:\n{doc['relevant_entities']}\n\n"
        if doc.get("relevant_entities")
        else ""
    )

    return (
        f"{instruction}\n\n"
        f"{entities_block}"
        f"Patient note:\n{truncated_case_text}\n\n"
        f"Question:\n{doc['question']}\n\n"
        "Final answer:"
    )


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    raw_prediction = results[0] if isinstance(results, list) else results
    output_type = doc["output_type"]

    if output_type in NUMERIC_OUTPUT_TYPES:
        parsed_prediction = extract_answer_numeric(raw_prediction)
        if parsed_prediction is None:
            return {"exact_match": 0}

        lower_decimal = decimal_from_string(doc["lower_limit"])
        upper_decimal = decimal_from_string(doc["upper_limit"])
        if lower_decimal is not None and upper_decimal is not None:
            correct = lower_decimal <= parsed_prediction <= upper_decimal
            return {"exact_match": int(correct)}

        target_decimal = decimal_from_string(doc["ground_truth_answer"])
        if target_decimal is not None:
            correct = canonicalize_decimal(parsed_prediction) == canonicalize_decimal(
                target_decimal
            )
            return {"exact_match": int(correct)}

    if output_type == "date":
        correct = normalize_date_like(raw_prediction) == normalize_date_like(
            doc["ground_truth_answer"]
        )
        return {"exact_match": int(correct)}

    correct = normalize_text(raw_prediction) == normalize_text(doc["ground_truth_answer"])
    return {"exact_match": int(correct)}
