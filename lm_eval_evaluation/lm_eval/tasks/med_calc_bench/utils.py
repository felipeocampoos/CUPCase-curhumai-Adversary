from __future__ import annotations

import os
import random
import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Dict, List, Optional

import datasets
import pandas as pd


NUMERIC_OUTPUT_TYPES = {"decimal", "integer"}
DEFAULT_DATASET_ID = "nsk7153/MedCalc-Bench-Verified"


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


def extract_answer_segment(text: str) -> str:
    stripped = str(text).strip()
    for pattern in (
        r"(?is)\bfinal answer\b\s*[:=-]\s*(.+)",
        r"(?is)\banswer\b\s*[:=-]\s*(.+)",
    ):
        match = re.search(pattern, stripped)
        if match:
            return match.group(1).strip()
    conversational_match = re.search(r"(?is)\bthe answer is\b\s+(.+)", stripped)
    if conversational_match:
        return conversational_match.group(1).strip()
    first_line = stripped.splitlines()[0].strip() if stripped else ""
    return first_line or stripped


def extract_answer_numeric(text: str) -> Optional[Decimal]:
    answer_segments = [extract_answer_segment(text), str(text).strip()]
    for segment in answer_segments:
        candidates = extract_numeric_candidates(segment)
        if candidates:
            return candidates[0]
    return None


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


def format_case_block(doc: dict) -> str:
    words = str(doc["case_text"]).split()
    truncated_case_text = " ".join(words[:250]) if len(words) > 250 else doc["case_text"]
    entities_block = (
        f"Relevant entities:\n{doc['relevant_entities']}\n\n"
        if doc.get("relevant_entities")
        else ""
    )

    return (
        f"{entities_block}"
        f"Patient note:\n{truncated_case_text}\n\n"
        f"Question:\n{doc['question']}"
    )


def build_instruction(output_type: str, *, chain_of_thought: bool) -> str:
    if chain_of_thought:
        instruction = (
            "Read the patient note and solve the medical calculation question. "
            "Reason step by step, then end with a line starting exactly with 'Final answer:'."
        )
    else:
        instruction = (
            "Read the patient note and answer the medical calculation question. "
            "Return only the final answer with no explanation."
        )
    if output_type in NUMERIC_OUTPUT_TYPES:
        suffix = (
            " If the answer is numeric, the final answer line must contain only the final numeric value."
            if chain_of_thought
            else " If the answer is numeric, return only the final numeric value."
        )
        instruction += suffix
    elif output_type == "date":
        suffix = (
            " If the answer is a date, the final answer line must contain only the final date."
            if chain_of_thought
            else " If the answer is a date, return only the final date."
        )
        instruction += suffix
    return instruction


@lru_cache(maxsize=8)
def _cached_one_shot_pool(dataset_id: str) -> List[dict]:
    dataset = datasets.load_dataset(dataset_id, split="one_shot")
    rows = []
    for doc in dataset:
        rows.append(
            {
                "id": f"{doc.get('Note ID', '')}:{doc.get('Row Number', '')}".strip(":"),
                "case_text": doc["Patient Note"],
                "question": doc["Question"],
                "relevant_entities": doc.get("Relevant Entities", ""),
                "ground_truth_answer": str(doc["Ground Truth Answer"]),
                "ground_truth_explanation": doc.get("Ground Truth Explanation", ""),
            }
        )
    return rows


def select_one_shot_example(*, exclude_id: str | None = None) -> dict:
    dataset_id = os.environ.get("MEDCALC_DATASET_ID", DEFAULT_DATASET_ID)
    examples = _cached_one_shot_pool(dataset_id)
    if exclude_id:
        examples = [example for example in examples if example["id"] != exclude_id]
    if not examples:
        raise ValueError("No one-shot examples available after exclusion")
    explicit_id = os.environ.get("MEDCALC_ICL_EXAMPLE_ID")
    if explicit_id:
        for example in examples:
            if example["id"] == explicit_id:
                return example
        raise ValueError(f"Unknown or excluded MEDCALC_ICL_EXAMPLE_ID '{explicit_id}'")
    seed = int(os.environ.get("MEDCALC_ICL_SEED", "42"))
    rng = random.Random(seed)
    return examples[rng.randrange(len(examples))]


def format_one_shot_example(example: dict) -> str:
    example_doc = dict(example)
    example_doc["case_text"] = " ".join(str(example["case_text"]).split()[:24])
    explanation = (
        " ".join(str(example.get("ground_truth_explanation", "")).strip().split()[:40])
        or "Use the relevant calculator inputs from the note and solve directly."
    )
    return (
        "Worked example:\n"
        f"{format_case_block(example_doc)}\n\n"
        f"Reasoning:\n{explanation}\n\n"
        f"Final answer: {str(example['ground_truth_answer']).strip()}"
    )


def doc_to_text(doc: dict) -> str:
    return f"{build_instruction(doc['output_type'], chain_of_thought=False)}\n\n{format_case_block(doc)}\n\nFinal answer:"


def doc_to_text_zero_shot_cot(doc: dict) -> str:
    return f"{build_instruction(doc['output_type'], chain_of_thought=True)}\n\n{format_case_block(doc)}\n\nReasoning:\n"


def doc_to_text_one_shot_cot(doc: dict) -> str:
    example = select_one_shot_example(exclude_id=str(doc.get("id", "")) or None)
    shortened_doc = dict(doc)
    shortened_doc["case_text"] = " ".join(str(doc["case_text"]).split()[:120])
    return (
        f"{build_instruction(doc['output_type'], chain_of_thought=True)}\n\n"
        f"{format_one_shot_example(example)}\n\n"
        "Now solve the real case.\n\n"
        f"{format_case_block(shortened_doc)}\n\n"
        "Reasoning:\n"
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
        correct = normalize_date_like(extract_answer_segment(raw_prediction)) == normalize_date_like(
            doc["ground_truth_answer"]
        )
        return {"exact_match": int(correct)}

    correct = normalize_text(extract_answer_segment(raw_prediction)) == normalize_text(
        doc["ground_truth_answer"]
    )
    return {"exact_match": int(correct)}
