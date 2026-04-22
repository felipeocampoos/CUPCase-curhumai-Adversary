"""Run MedCalc-Bench-Verified through the repo's API-based evaluation surface."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from eval_batching import build_eval_batches
from prepare_hf_medcalc_bench import (
    DEFAULT_DATASET_ID,
    build_default_output_path,
    convert_rows,
    load_hf_rows,
    maybe_sample,
    normalize_split_name,
    save_csv,
)
from refinement.refiner import JudgeProvider, create_client
from refinement.run_manifest import create_run_manifest, save_run_manifest


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

NUMERIC_OUTPUT_TYPES = {"decimal", "integer"}
DATE_OUTPUT_TYPE = "date"
VALID_METHODS = [
    "direct",
    "zero_shot_cot",
    "one_shot_cot",
    "medcalc_semantic_gate",
    "medcalc_uncertainty_consistency_gate",
]


@dataclass
class ScoreResult:
    is_correct: bool
    scoring_mode: str
    parsed_prediction: str
    normalized_prediction: str
    failure_reason: str


@dataclass
class ParsedCandidate:
    raw_prediction: str
    parsed_prediction: str
    normalized_prediction: str


@dataclass
class MethodExecutionResult:
    raw_prediction: str
    parsed_prediction: str
    normalized_prediction: str
    metadata: Dict[str, Any]


def slugify_model(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MedCalc-Bench-Verified via the API-based experiment surface"
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_ID)
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
        "--method",
        type=str,
        default="direct",
        choices=VALID_METHODS,
        help="Prompting/evaluation method to run for MedCalc-Bench",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="output/experiments/medcalc_bench",
        help="Base directory for MedCalc-Bench evaluation artifacts",
    )
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=60.0)
    parser.add_argument("--api-delay", type=float, default=1.0)
    parser.add_argument("--n-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument(
        "--max-case-words",
        type=int,
        default=250,
        help="Maximum number of patient-note words to include in prompts",
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="unique",
        choices=["unique", "bootstrap"],
        help="Batch sampling mode for evaluator execution",
    )
    parser.add_argument(
        "--icl-example-id",
        type=str,
        default=None,
        help="Use a fixed MedCalc one-shot example id for one_shot_cot",
    )
    parser.add_argument(
        "--icl-seed",
        type=int,
        default=42,
        help="Seed for deterministic one-shot example selection when no fixed id is provided",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=3,
        help="Number of candidate generations for medcalc_semantic_gate",
    )
    parser.add_argument(
        "--candidate-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for candidate generations in medcalc_semantic_gate",
    )
    parser.add_argument(
        "--verifier-risk-threshold",
        type=float,
        default=0.5,
        help="Risk threshold used by medcalc_uncertainty_consistency_gate",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=192,
        help="Maximum completion tokens for MedCalc prompts",
    )
    return parser.parse_args()


def truncate_words(text: str, max_words: int) -> str:
    words = str(text).split()
    if max_words <= 0 or len(words) <= max_words:
        return str(text)
    return " ".join(words[:max_words])


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


UNCERTAINTY_HEDGE_PATTERNS = (
    r"\buncertain\b",
    r"\bunsure\b",
    r"\bnot sure\b",
    r"\bnot certain\b",
    r"\bcannot determine\b",
    r"\bcan'?t determine\b",
    r"\binsufficient information\b",
    r"\binsufficient data\b",
    r"\bneed more (?:information|data)\b",
    r"\bmay be\b",
    r"\bmight be\b",
    r"\bpossible(?:ly)?\b",
)


@dataclass
class VerifierResult:
    label: str
    risk_score: float
    rationale: str
    signal_triggered: bool


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


def parse_candidate_prediction(raw_prediction: str, output_type: str) -> ParsedCandidate:
    normalized_output_type = normalize_text(output_type)
    if normalized_output_type in NUMERIC_OUTPUT_TYPES:
        parsed_prediction = extract_answer_numeric(raw_prediction)
        if parsed_prediction is None:
            return ParsedCandidate(
                raw_prediction=raw_prediction,
                parsed_prediction="",
                normalized_prediction="",
            )
        normalized_prediction = canonicalize_decimal(parsed_prediction)
        return ParsedCandidate(
            raw_prediction=raw_prediction,
            parsed_prediction=normalized_prediction,
            normalized_prediction=normalized_prediction,
        )

    answer_segment = extract_answer_segment(raw_prediction)
    if normalized_output_type == DATE_OUTPUT_TYPE:
        normalized_prediction = normalize_date_like(answer_segment)
        return ParsedCandidate(
            raw_prediction=raw_prediction,
            parsed_prediction=normalized_prediction,
            normalized_prediction=normalized_prediction,
        )

    normalized_prediction = normalize_text(answer_segment)
    return ParsedCandidate(
        raw_prediction=raw_prediction,
        parsed_prediction=normalized_prediction,
        normalized_prediction=normalized_prediction,
    )


def build_base_instruction(method: str, output_type: str) -> str:
    if method == "direct":
        instruction = (
            "Read the patient note and answer the medical calculation question. "
            "Return only the final answer with no explanation."
        )
    else:
        instruction = (
            "Read the patient note and solve the medical calculation question. "
            "Reason step by step, then end with a line starting exactly with 'Final answer:'."
        )

    if output_type in NUMERIC_OUTPUT_TYPES:
        instruction += " If the answer is numeric, the final answer line must contain only the final numeric value."
    elif output_type == DATE_OUTPUT_TYPE:
        instruction += " If the answer is a date, the final answer line must contain only the final date."
    return instruction


def format_case_block(
    case_text: str,
    question: str,
    relevant_entities: str,
    max_case_words: int,
) -> str:
    truncated_case_text = truncate_words(case_text, max_case_words)
    entities_block = (
        f"Relevant entities:\n{relevant_entities}\n\n" if str(relevant_entities).strip() else ""
    )
    return (
        f"{entities_block}"
        f"Patient note:\n{truncated_case_text}\n\n"
        f"Question:\n{question}"
    )


def format_one_shot_example(example: Dict[str, Any], max_case_words: int) -> str:
    example_case_words = min(max_case_words, 24)
    case_block = format_case_block(
        case_text=str(example["case_text"]),
        question=str(example["question"]),
        relevant_entities=str(example.get("relevant_entities", "")),
        max_case_words=example_case_words,
    )
    explanation = truncate_words(
        str(example.get("ground_truth_explanation", "")).strip()
        or "Use the calculator inputs from the note and solve directly.",
        40,
    )
    final_answer = str(example["ground_truth_answer"]).strip()
    return (
        "Worked example:\n"
        f"{case_block}\n\n"
        f"Reasoning:\n{explanation}\n\n"
        f"Final answer: {final_answer}"
    )


def build_prompt(
    *,
    method: str,
    case_text: str,
    question: str,
    output_type: str,
    relevant_entities: str,
    max_case_words: int,
    icl_example: Optional[Dict[str, Any]] = None,
) -> str:
    instruction = build_base_instruction(method, output_type)
    case_word_limit = min(max_case_words, 120) if method == "one_shot_cot" else max_case_words
    case_block = format_case_block(
        case_text=case_text,
        question=question,
        relevant_entities=relevant_entities,
        max_case_words=case_word_limit,
    )

    if method == "one_shot_cot":
        if icl_example is None:
            raise ValueError("one_shot_cot requires an ICL example")
        return (
            f"{instruction}\n\n"
            f"{format_one_shot_example(icl_example, max_case_words)}\n\n"
            "Now solve the real case.\n\n"
            f"{case_block}\n\n"
            "Reasoning:\n"
        )

    if method in {
        "zero_shot_cot",
        "medcalc_semantic_gate",
        "medcalc_uncertainty_consistency_gate",
    }:
        return f"{instruction}\n\n{case_block}\n\nReasoning:\n"

    return f"{instruction}\n\n{case_block}\n\nFinal answer:"


def call_model(
    client: Any,
    model: str,
    prompt: str,
    retry_attempts: int,
    retry_delay: float,
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(retry_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # pragma: no cover - network behavior
            last_error = exc
            logger.warning(
                "MedCalc API call attempt %s/%s failed: %s",
                attempt + 1,
                retry_attempts,
                exc,
            )
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
    assert last_error is not None
    raise last_error


def score_prediction(
    *,
    raw_prediction: str,
    output_type: str,
    ground_truth_answer: str,
    lower_limit: str,
    upper_limit: str,
) -> ScoreResult:
    normalized_output_type = normalize_text(output_type)
    normalized_gt = normalize_text(ground_truth_answer)

    if normalized_output_type in NUMERIC_OUTPUT_TYPES:
        parsed_prediction = extract_answer_numeric(raw_prediction)
        if parsed_prediction is None:
            return ScoreResult(
                is_correct=False,
                scoring_mode="numeric_parse_failed",
                parsed_prediction="",
                normalized_prediction=normalize_text(raw_prediction),
                failure_reason="numeric_parse_failed",
            )

        parsed_prediction_text = canonicalize_decimal(parsed_prediction)
        lower_decimal = decimal_from_string(lower_limit)
        upper_decimal = decimal_from_string(upper_limit)

        if lower_decimal is not None and upper_decimal is not None:
            correct = lower_decimal <= parsed_prediction <= upper_decimal
            failure_reason = "" if correct else "outside_tolerance"
            return ScoreResult(
                is_correct=correct,
                scoring_mode="numeric_interval",
                parsed_prediction=parsed_prediction_text,
                normalized_prediction=parsed_prediction_text,
                failure_reason=failure_reason,
            )

        gt_decimal = decimal_from_string(ground_truth_answer)
        if gt_decimal is not None:
            gt_text = canonicalize_decimal(gt_decimal)
            correct = parsed_prediction_text == gt_text
            return ScoreResult(
                is_correct=correct,
                scoring_mode="numeric_exact",
                parsed_prediction=parsed_prediction_text,
                normalized_prediction=parsed_prediction_text,
                failure_reason="" if correct else "numeric_mismatch",
            )

    if normalized_output_type == DATE_OUTPUT_TYPE:
        normalized_prediction = normalize_date_like(extract_answer_segment(raw_prediction))
        normalized_target = normalize_date_like(ground_truth_answer)
        correct = normalized_prediction == normalized_target
        return ScoreResult(
            is_correct=correct,
            scoring_mode="date_exact",
            parsed_prediction=normalized_prediction,
            normalized_prediction=normalized_prediction,
            failure_reason="" if correct else "date_mismatch",
        )

    normalized_prediction = normalize_text(extract_answer_segment(raw_prediction))
    correct = normalized_prediction == normalized_gt
    return ScoreResult(
        is_correct=correct,
        scoring_mode="normalized_exact",
        parsed_prediction=normalized_prediction,
        normalized_prediction=normalized_prediction,
        failure_reason="" if correct else "text_mismatch",
    )


def build_results_dir(
    output_root: str,
    split: str,
    provider: str,
    method: str,
    model: str,
    sample_size: int,
    seed: int,
) -> Path:
    sample_label = "all" if sample_size <= 0 else f"n{sample_size}_seed{seed}"
    return (
        Path(output_root)
        / normalize_split_name(split)
        / provider
        / "free_text"
        / method
        / slugify_model(model)
        / sample_label
    )


def create_summary_report(
    results_df: pd.DataFrame,
    *,
    split: str,
    provider: str,
    model: str,
    method: str,
    runtime_seconds: float,
) -> Dict[str, Any]:
    n_cases = len(results_df)
    accuracy = float(results_df["is_correct"].mean()) if n_cases else 0.0
    numeric_rows = results_df["output_type"].isin(sorted(NUMERIC_OUTPUT_TYPES))
    numeric_parse_rate = (
        float((results_df.loc[numeric_rows, "parsed_prediction"] != "").mean())
        if numeric_rows.any()
        else None
    )
    accuracy_by_type = {
        output_type: float(group["is_correct"].mean())
        for output_type, group in results_df.groupby("output_type", dropna=False)
    }
    scoring_mode_counts = {
        str(mode): int(count)
        for mode, count in results_df["scoring_mode"].value_counts(dropna=False).items()
    }
    failure_reason_counts = {
        str(reason): int(count)
        for reason, count in results_df["failure_reason"].value_counts(dropna=False).items()
        if str(reason)
    }
    selection_source_counts = {
        str(source): int(count)
        for source, count in results_df["selection_source"].value_counts(dropna=False).items()
        if str(source)
    } if "selection_source" in results_df.columns else {}

    return {
        "timestamp": datetime.now().isoformat(),
        "n_cases": n_cases,
        "split": split,
        "provider": provider,
        "model": model,
        "method": method,
        "metrics": {
            "accuracy": accuracy,
            "numeric_parse_rate": numeric_parse_rate,
            "accuracy_by_output_type": accuracy_by_type,
            "scoring_mode_counts": scoring_mode_counts,
            "failure_reason_counts": failure_reason_counts,
            "selection_source_counts": selection_source_counts,
        },
        "runtime_seconds": runtime_seconds,
    }


def select_icl_example(
    examples: Sequence[Dict[str, Any]],
    *,
    example_id: Optional[str],
    seed: int,
    exclude_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    excluded = {str(example_id_value) for example_id_value in (exclude_ids or [])}
    eligible_examples = [
        example for example in examples if str(example.get("id", "")) not in excluded
    ]
    if not eligible_examples:
        raise ValueError("No one-shot examples available")
    if example_id:
        for example in eligible_examples:
            if str(example.get("id")) == example_id:
                return example
        raise ValueError(
            f"Unknown or excluded ICL example id '{example_id}'"
        )
    rng = random.Random(seed)
    return eligible_examples[rng.randrange(len(eligible_examples))]


def build_candidate_counter(candidates: Sequence[ParsedCandidate]) -> Counter[str]:
    return Counter(
        candidate.normalized_prediction
        for candidate in candidates
        if candidate.normalized_prediction
    )


def choose_consensus_candidate(candidates: Sequence[ParsedCandidate]) -> Optional[ParsedCandidate]:
    counts = build_candidate_counter(candidates)
    if not counts:
        return None
    top_prediction, top_count = counts.most_common(1)[0]
    if top_count < 2:
        return None
    for candidate in candidates:
        if candidate.normalized_prediction == top_prediction:
            return candidate
    return None


def candidate_agreement_signal(candidates: Sequence[ParsedCandidate]) -> bool:
    if len(candidates) < 2:
        return False
    counts = build_candidate_counter(candidates)
    if not counts:
        return True
    top_prediction, top_count = counts.most_common(1)[0]
    if top_count < 2:
        return True
    if any(not candidate.normalized_prediction for candidate in candidates):
        return True
    non_empty = [candidate.normalized_prediction for candidate in candidates if candidate.normalized_prediction]
    return any(prediction != top_prediction for prediction in non_empty)


def has_uncertainty_signal(raw_prediction: str) -> bool:
    stripped = str(raw_prediction).strip()
    if not stripped:
        return True
    if not re.search(r"(?im)^\s*final answer\s*[:=-]", stripped):
        return True
    answer_portion = str(extract_answer_segment(stripped)).lower()
    return any(re.search(pattern, answer_portion) for pattern in UNCERTAINTY_HEDGE_PATTERNS)


def build_verifier_prompt(
    *,
    case_text: str,
    question: str,
    relevant_entities: str,
    proposed_answer: str,
    max_case_words: int,
) -> str:
    case_block = format_case_block(
        case_text=case_text,
        question=question,
        relevant_entities=relevant_entities,
        max_case_words=max_case_words,
    )
    return (
        "You are verifying whether a proposed answer to a medical calculator question is supported "
        "by the patient note.\n"
        "Respond in exactly three lines:\n"
        "Label: supported|unsupported|insufficient_information\n"
        "Risk: <number between 0 and 1>\n"
        "Rationale: <one short sentence>\n\n"
        f"{case_block}\n\n"
        f"Proposed answer:\n{proposed_answer}"
    )


def parse_verifier_output(raw_output: str, risk_threshold: float) -> VerifierResult:
    stripped = str(raw_output).strip()
    label_match = re.search(
        r"(?im)^\s*label\s*:\s*(supported|unsupported|insufficient_information)\b",
        stripped,
    )
    risk_match = re.search(r"(?im)^\s*risk\s*:\s*([01](?:\.\d+)?)\b", stripped)
    rationale_match = re.search(r"(?im)^\s*rationale\s*:\s*(.+)$", stripped)

    label = label_match.group(1).lower() if label_match else ""
    if not label:
        lowered = stripped.lower()
        if "unsupported" in lowered:
            label = "unsupported"
        elif "insufficient_information" in lowered or "insufficient information" in lowered:
            label = "insufficient_information"
        elif "supported" in lowered:
            label = "supported"
        else:
            label = "insufficient_information"

    risk_score = 1.0
    if risk_match:
        try:
            risk_score = max(0.0, min(1.0, float(risk_match.group(1))))
        except ValueError:
            risk_score = 1.0
    else:
        if label == "supported":
            risk_score = 0.0
        elif label == "unsupported":
            risk_score = 1.0
        else:
            risk_score = 0.75

    rationale = rationale_match.group(1).strip() if rationale_match else stripped.splitlines()[0].strip()
    signal_triggered = label != "supported" or risk_score >= risk_threshold
    return VerifierResult(
        label=label,
        risk_score=risk_score,
        rationale=rationale,
        signal_triggered=signal_triggered,
    )


def build_guided_retry_prompt(
    *,
    case_text: str,
    question: str,
    output_type: str,
    relevant_entities: str,
    max_case_words: int,
    baseline_answer: str,
    verifier_result: Optional[VerifierResult],
    candidates: Sequence[ParsedCandidate],
) -> str:
    case_block = format_case_block(
        case_text=case_text,
        question=question,
        relevant_entities=relevant_entities,
        max_case_words=max_case_words,
    )
    candidate_summary = ", ".join(
        candidate.normalized_prediction for candidate in candidates if candidate.normalized_prediction
    ) or "No stable parsed candidates."
    verifier_summary = (
        f"Verifier label: {verifier_result.label}. Risk: {verifier_result.risk_score:.2f}. "
        f"Rationale: {verifier_result.rationale}"
        if verifier_result is not None
        else "Verifier feedback unavailable."
    )
    instruction = build_base_instruction("zero_shot_cot", output_type)
    return (
        f"{instruction}\n\n"
        f"{case_block}\n\n"
        "Revise the answer carefully using the verifier feedback and candidate disagreement summary.\n"
        f"Current draft:\n{baseline_answer}\n\n"
        f"Candidate summary:\n{candidate_summary}\n\n"
        f"{verifier_summary}\n\n"
        "Reasoning:\n"
    )


def build_adjudication_prompt(
    *,
    case_text: str,
    question: str,
    output_type: str,
    relevant_entities: str,
    max_case_words: int,
    candidates: Sequence[ParsedCandidate],
) -> str:
    case_block = format_case_block(
        case_text=case_text,
        question=question,
        relevant_entities=relevant_entities,
        max_case_words=max_case_words,
    )
    candidate_lines = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_lines.append(f"Candidate {index}: {candidate.raw_prediction}")
        if candidate.normalized_prediction:
            candidate_lines.append(f"Parsed final answer {index}: {candidate.normalized_prediction}")
    candidate_block = "\n".join(candidate_lines)
    instruction = build_base_instruction("direct", output_type)
    return (
        "You are adjudicating multiple candidate answers to the same medical calculator question.\n"
        "Choose the single best final answer grounded in the case facts and candidate outputs.\n"
        f"{instruction}\n\n"
        f"{case_block}\n\n"
        f"Candidate answers:\n{candidate_block}\n\n"
        "Final answer:"
    )


def build_uncertainty_adjudication_prompt(
    *,
    case_text: str,
    question: str,
    output_type: str,
    relevant_entities: str,
    max_case_words: int,
    baseline_answer: str,
    candidates: Sequence[ParsedCandidate],
    verifier_result: Optional[VerifierResult],
    uncertainty_triggered: bool,
) -> str:
    case_block = format_case_block(
        case_text=case_text,
        question=question,
        relevant_entities=relevant_entities,
        max_case_words=max_case_words,
    )
    candidate_lines = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_lines.append(f"Candidate {index}: {candidate.raw_prediction}")
        candidate_lines.append(
            f"Parsed answer {index}: {candidate.normalized_prediction or '[unparsed]'}"
        )
    verifier_block = (
        f"Verifier label: {verifier_result.label}\n"
        f"Verifier risk: {verifier_result.risk_score:.2f}\n"
        f"Verifier rationale: {verifier_result.rationale}"
        if verifier_result is not None
        else "Verifier feedback unavailable."
    )
    instruction = build_base_instruction("direct", output_type)
    return (
        "You are adjudicating a medical calculator answer after multiple instability signals fired.\n"
        "Choose the single best final answer grounded only in the case facts.\n"
        f"{instruction}\n\n"
        f"{case_block}\n\n"
        f"Baseline answer:\n{baseline_answer}\n\n"
        f"Candidate answers:\n{chr(10).join(candidate_lines)}\n\n"
        f"{verifier_block}\n\n"
        f"Uncertainty triggered: {'yes' if uncertainty_triggered else 'no'}\n\n"
        "Final answer:"
    )


def execute_method(
    *,
    client: Any,
    model: str,
    row: Dict[str, Any],
    args: argparse.Namespace,
    icl_examples: Optional[Sequence[Dict[str, Any]]],
) -> MethodExecutionResult:
    icl_example: Optional[Dict[str, Any]] = None
    if args.method == "one_shot_cot":
        if not icl_examples:
            raise ValueError("one_shot_cot requires a one-shot example pool")
        icl_example = select_icl_example(
            icl_examples,
            example_id=args.icl_example_id,
            seed=args.icl_seed,
            exclude_ids=[str(row["id"])],
        )

    prompt = build_prompt(
        method=args.method,
        case_text=str(row["case_text"]),
        question=str(row["question"]),
        output_type=str(row["output_type"]),
        relevant_entities=str(row.get("relevant_entities", "")),
        max_case_words=args.max_case_words,
        icl_example=icl_example,
    )

    if args.method not in {
        "medcalc_semantic_gate",
        "medcalc_uncertainty_consistency_gate",
    }:
        raw_prediction = call_model(
            client=client,
            model=model,
            prompt=prompt,
            retry_attempts=args.retry_attempts,
            retry_delay=args.retry_delay,
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
        parsed = parse_candidate_prediction(raw_prediction, str(row["output_type"]))
        return MethodExecutionResult(
            raw_prediction=raw_prediction,
            parsed_prediction=parsed.parsed_prediction,
            normalized_prediction=parsed.normalized_prediction,
            metadata={
                "icl_example_id": icl_example["id"] if icl_example else "",
                "gate_triggered": False,
                "adjudication_invoked": False,
                "retry_invoked": False,
                "selection_source": "single_pass",
                "candidate_predictions": "[]",
                "candidate_raw_predictions": "[]",
                "agreement_signal_triggered": False,
                "uncertainty_signal_triggered": False,
                "verifier_signal_triggered": False,
                "verifier_label": "",
                "verifier_rationale": "",
                "verifier_risk_score": "",
                "signal_count": 0,
                "escalation_tier": "stable",
            },
        )

    candidates: List[ParsedCandidate] = []
    for _ in range(max(args.num_candidates, 1)):
        raw_prediction = call_model(
            client=client,
            model=model,
            prompt=prompt,
            retry_attempts=args.retry_attempts,
            retry_delay=args.retry_delay,
            temperature=args.candidate_temperature,
            max_tokens=args.max_tokens,
        )
        candidates.append(parse_candidate_prediction(raw_prediction, str(row["output_type"])))

    consensus = choose_consensus_candidate(candidates)
    metadata = {
        "icl_example_id": icl_example["id"] if icl_example else "",
        "candidate_predictions": json.dumps(
            [candidate.normalized_prediction for candidate in candidates],
            ensure_ascii=True,
        ),
        "candidate_raw_predictions": json.dumps(
            [candidate.raw_prediction for candidate in candidates],
            ensure_ascii=True,
        ),
    }
    if args.method == "medcalc_semantic_gate" and consensus is not None:
        metadata.update(
            {
                "gate_triggered": False,
                "adjudication_invoked": False,
                "retry_invoked": False,
                "selection_source": "candidate_consensus",
                "agreement_signal_triggered": False,
                "uncertainty_signal_triggered": False,
                "verifier_signal_triggered": False,
                "verifier_label": "",
                "verifier_rationale": "",
                "verifier_risk_score": "",
                "signal_count": 0,
                "escalation_tier": "stable",
            }
        )
        return MethodExecutionResult(
            raw_prediction=consensus.raw_prediction,
            parsed_prediction=consensus.parsed_prediction,
            normalized_prediction=consensus.normalized_prediction,
            metadata=metadata,
        )

    if args.method == "medcalc_semantic_gate":
        adjudication_prompt = build_adjudication_prompt(
            case_text=str(row["case_text"]),
            question=str(row["question"]),
            output_type=str(row["output_type"]),
            relevant_entities=str(row.get("relevant_entities", "")),
            max_case_words=args.max_case_words,
            candidates=candidates,
        )
        raw_prediction = call_model(
            client=client,
            model=model,
            prompt=adjudication_prompt,
            retry_attempts=args.retry_attempts,
            retry_delay=args.retry_delay,
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
        parsed = parse_candidate_prediction(raw_prediction, str(row["output_type"]))
        metadata.update(
            {
                "gate_triggered": True,
                "adjudication_invoked": True,
                "retry_invoked": False,
                "selection_source": "adjudication",
                "agreement_signal_triggered": True,
                "uncertainty_signal_triggered": False,
                "verifier_signal_triggered": False,
                "verifier_label": "",
                "verifier_rationale": "",
                "verifier_risk_score": "",
                "signal_count": 1,
                "escalation_tier": "adjudication",
            }
        )
        return MethodExecutionResult(
            raw_prediction=raw_prediction,
            parsed_prediction=parsed.parsed_prediction,
            normalized_prediction=parsed.normalized_prediction,
            metadata=metadata,
        )

    baseline_raw_prediction = call_model(
        client=client,
        model=model,
        prompt=prompt,
        retry_attempts=args.retry_attempts,
        retry_delay=args.retry_delay,
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    baseline_parsed = parse_candidate_prediction(
        baseline_raw_prediction,
        str(row["output_type"]),
    )
    agreement_triggered = candidate_agreement_signal(candidates)
    uncertainty_triggered = has_uncertainty_signal(baseline_raw_prediction)

    verifier_result: Optional[VerifierResult] = None
    verifier_error = ""
    try:
        verifier_prompt = build_verifier_prompt(
            case_text=str(row["case_text"]),
            question=str(row["question"]),
            relevant_entities=str(row.get("relevant_entities", "")),
            proposed_answer=baseline_raw_prediction,
            max_case_words=args.max_case_words,
        )
        verifier_raw = call_model(
            client=client,
            model=model,
            prompt=verifier_prompt,
            retry_attempts=args.retry_attempts,
            retry_delay=args.retry_delay,
            temperature=0.0,
            max_tokens=max(96, min(args.max_tokens, 160)),
        )
        verifier_result = parse_verifier_output(verifier_raw, args.verifier_risk_threshold)
    except Exception as exc:  # pragma: no cover - network/model behavior
        verifier_error = str(exc)
        verifier_result = VerifierResult(
            label="insufficient_information",
            risk_score=1.0,
            rationale=f"verifier_error: {exc}",
            signal_triggered=True,
        )

    signal_count = sum(
        int(flag)
        for flag in (
            agreement_triggered,
            uncertainty_triggered,
            verifier_result.signal_triggered if verifier_result is not None else True,
        )
    )
    metadata.update(
        {
            "agreement_signal_triggered": agreement_triggered,
            "uncertainty_signal_triggered": uncertainty_triggered,
            "verifier_signal_triggered": (
                verifier_result.signal_triggered if verifier_result is not None else True
            ),
            "verifier_label": verifier_result.label if verifier_result is not None else "",
            "verifier_rationale": verifier_result.rationale if verifier_result is not None else "",
            "verifier_risk_score": (
                f"{verifier_result.risk_score:.2f}" if verifier_result is not None else ""
            ),
            "signal_count": signal_count,
        }
    )
    if verifier_error:
        metadata["verifier_error"] = verifier_error

    if signal_count == 0:
        metadata.update(
            {
                "gate_triggered": False,
                "adjudication_invoked": False,
                "retry_invoked": False,
                "selection_source": "baseline_generator",
                "escalation_tier": "stable",
            }
        )
        return MethodExecutionResult(
            raw_prediction=baseline_raw_prediction,
            parsed_prediction=baseline_parsed.parsed_prediction,
            normalized_prediction=baseline_parsed.normalized_prediction,
            metadata=metadata,
        )

    high_instability = (
        signal_count >= 2
        or (
            verifier_result is not None
            and verifier_result.label in {"unsupported", "insufficient_information"}
        )
    )
    if not high_instability:
        retry_prompt = build_guided_retry_prompt(
            case_text=str(row["case_text"]),
            question=str(row["question"]),
            output_type=str(row["output_type"]),
            relevant_entities=str(row.get("relevant_entities", "")),
            max_case_words=args.max_case_words,
            baseline_answer=baseline_raw_prediction,
            verifier_result=verifier_result,
            candidates=candidates,
        )
        try:
            retry_raw_prediction = call_model(
                client=client,
                model=model,
                prompt=retry_prompt,
                retry_attempts=args.retry_attempts,
                retry_delay=args.retry_delay,
                temperature=0.0,
                max_tokens=args.max_tokens,
            )
            retry_parsed = parse_candidate_prediction(retry_raw_prediction, str(row["output_type"]))
            metadata.update(
                {
                    "gate_triggered": True,
                    "adjudication_invoked": False,
                    "retry_invoked": True,
                    "selection_source": "guided_retry",
                    "escalation_tier": "guided_retry",
                }
            )
            return MethodExecutionResult(
                raw_prediction=retry_raw_prediction,
                parsed_prediction=retry_parsed.parsed_prediction,
                normalized_prediction=retry_parsed.normalized_prediction,
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - network/model behavior
            metadata["guided_retry_error"] = str(exc)
            metadata.update(
                {
                    "gate_triggered": True,
                    "adjudication_invoked": False,
                    "retry_invoked": True,
                    "selection_source": "baseline_fallback_guided_retry_error",
                    "escalation_tier": "guided_retry",
                }
            )
            return MethodExecutionResult(
                raw_prediction=baseline_raw_prediction,
                parsed_prediction=baseline_parsed.parsed_prediction,
                normalized_prediction=baseline_parsed.normalized_prediction,
                metadata=metadata,
            )

    adjudication_prompt = build_uncertainty_adjudication_prompt(
        case_text=str(row["case_text"]),
        question=str(row["question"]),
        output_type=str(row["output_type"]),
        relevant_entities=str(row.get("relevant_entities", "")),
        max_case_words=args.max_case_words,
        baseline_answer=baseline_raw_prediction,
        candidates=candidates,
        verifier_result=verifier_result,
        uncertainty_triggered=uncertainty_triggered,
    )
    try:
        raw_prediction = call_model(
            client=client,
            model=model,
            prompt=adjudication_prompt,
            retry_attempts=args.retry_attempts,
            retry_delay=args.retry_delay,
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
        parsed = parse_candidate_prediction(raw_prediction, str(row["output_type"]))
        metadata.update(
            {
                "gate_triggered": True,
                "adjudication_invoked": True,
                "retry_invoked": False,
                "selection_source": "adjudication",
                "escalation_tier": "adjudication",
            }
        )
        return MethodExecutionResult(
            raw_prediction=raw_prediction,
            parsed_prediction=parsed.parsed_prediction,
            normalized_prediction=parsed.normalized_prediction,
            metadata=metadata,
        )
    except Exception as exc:  # pragma: no cover - network/model behavior
        metadata["adjudication_error"] = str(exc)
        retry_prompt = build_guided_retry_prompt(
            case_text=str(row["case_text"]),
            question=str(row["question"]),
            output_type=str(row["output_type"]),
            relevant_entities=str(row.get("relevant_entities", "")),
            max_case_words=args.max_case_words,
            baseline_answer=baseline_raw_prediction,
            verifier_result=verifier_result,
            candidates=candidates,
        )
        try:
            retry_raw_prediction = call_model(
                client=client,
                model=model,
                prompt=retry_prompt,
                retry_attempts=args.retry_attempts,
                retry_delay=args.retry_delay,
                temperature=0.0,
                max_tokens=args.max_tokens,
            )
            retry_parsed = parse_candidate_prediction(retry_raw_prediction, str(row["output_type"]))
            metadata.update(
                {
                    "gate_triggered": True,
                    "adjudication_invoked": True,
                    "retry_invoked": True,
                    "selection_source": "guided_retry_fallback_after_adjudication_error",
                    "escalation_tier": "adjudication",
                }
            )
            return MethodExecutionResult(
                raw_prediction=retry_raw_prediction,
                parsed_prediction=retry_parsed.parsed_prediction,
                normalized_prediction=retry_parsed.normalized_prediction,
                metadata=metadata,
            )
        except Exception as retry_exc:  # pragma: no cover - network/model behavior
            metadata["guided_retry_error"] = str(retry_exc)
            metadata.update(
                {
                    "gate_triggered": True,
                    "adjudication_invoked": True,
                    "retry_invoked": True,
                    "selection_source": "baseline_fallback_adjudication_and_retry_error",
                    "escalation_tier": "adjudication",
                }
            )
            return MethodExecutionResult(
                raw_prediction=baseline_raw_prediction,
                parsed_prediction=baseline_parsed.parsed_prediction,
                normalized_prediction=baseline_parsed.normalized_prediction,
                metadata=metadata,
            )


def main() -> int:
    args = parse_args()
    provider = JudgeProvider(args.provider)
    model = args.model or provider.default_model

    rows = load_hf_rows(args.dataset, args.split)
    converted = convert_rows(rows)
    converted = maybe_sample(converted, args.sample_size, args.seed)
    if not converted:
        raise ValueError("No rows available to evaluate")

    generated_csv_path = build_default_output_path(
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    save_csv(converted, generated_csv_path)

    output_dir = build_results_dir(
        output_root=args.output_root,
        split=args.split,
        provider=args.provider,
        method=args.method,
        model=model,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    client = create_client(provider=provider)
    icl_examples: Optional[List[Dict[str, Any]]] = None
    if args.method == "one_shot_cot":
        icl_examples = convert_rows(load_hf_rows(args.dataset, "one_shot"))

    ds = pd.DataFrame(converted)
    batch_size = args.batch_size or len(ds)
    batches = build_eval_batches(
        ds=ds,
        n_batches=args.n_batches,
        batch_size=batch_size,
        random_seed=args.seed,
        sampling_mode=args.sampling_mode,
    )

    results: List[Dict[str, Any]] = []
    start_time = time.time()

    for batch_index, batch in enumerate(batches, start=1):
        logger.info("Processing batch %s/%s with %s rows", batch_index, len(batches), len(batch))
        for _, row in batch.iterrows():
            method_result = execute_method(
                client=client,
                model=model,
                row=dict(row),
                args=args,
                icl_examples=icl_examples,
            )
            score = score_prediction(
                raw_prediction=method_result.raw_prediction,
                output_type=str(row["output_type"]),
                ground_truth_answer=str(row["ground_truth_answer"]),
                lower_limit=str(row["lower_limit"]),
                upper_limit=str(row["upper_limit"]),
            )

            result_row = dict(row)
            result_row.update(
                {
                    "method": args.method,
                    "icl_example_id": (
                        method_result.metadata.get("icl_example_id", "")
                    ),
                    "raw_prediction": method_result.raw_prediction,
                    "parsed_prediction": score.parsed_prediction,
                    "normalized_prediction": score.normalized_prediction,
                    "is_correct": score.is_correct,
                    "scoring_mode": score.scoring_mode,
                    "failure_reason": score.failure_reason,
                }
            )
            result_row.update(method_result.metadata)
            results.append(result_row)

            if args.api_delay > 0:
                time.sleep(args.api_delay)

    runtime = time.time() - start_time
    results_df = pd.DataFrame(results)

    results_csv_path = output_dir / "results.csv"
    results_df.to_csv(results_csv_path, index=False)

    summary_report = create_summary_report(
        results_df,
        split=args.split,
        provider=args.provider,
        model=model,
        method=args.method,
        runtime_seconds=runtime,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"summary_report_{timestamp}.json"
    summary_path.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")

    manifest_path = output_dir / f"run_manifest_{timestamp}.json"
    manifest = create_run_manifest(
        script_path=__file__,
        output_paths={
            "generated_csv": generated_csv_path,
            "results_csv": results_csv_path,
            "summary_report": summary_path,
            "run_manifest": manifest_path,
        },
        config={
            "split": args.split,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "retry_attempts": args.retry_attempts,
            "retry_delay": args.retry_delay,
            "api_delay": args.api_delay,
            "n_batches": args.n_batches,
            "batch_size": batch_size,
            "sampling_mode": args.sampling_mode,
            "max_case_words": args.max_case_words,
            "method": args.method,
            "icl_example_id": args.icl_example_id,
            "icl_seed": args.icl_seed,
            "num_candidates": args.num_candidates,
            "candidate_temperature": args.candidate_temperature,
            "max_tokens": args.max_tokens,
        },
        dataset={
            "dataset_id": args.dataset,
            "split": args.split,
            "input_path": str(generated_csv_path),
            "n_rows_loaded": len(converted),
            "n_cases_processed": len(results_df),
        },
        task="free_text",
        provider=args.provider,
        model=model,
        runtime_seconds=runtime,
        extra={
            "benchmark": "medcalc_bench",
            "method": args.method,
            "icl_example_id": (
                args.icl_example_id if args.icl_example_id else None
            ),
        },
    )
    save_run_manifest(manifest, manifest_path)

    logger.info("Generated CSV: %s", generated_csv_path)
    logger.info("Results CSV: %s", results_csv_path)
    logger.info("Summary report: %s", summary_path)
    logger.info("Run manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
