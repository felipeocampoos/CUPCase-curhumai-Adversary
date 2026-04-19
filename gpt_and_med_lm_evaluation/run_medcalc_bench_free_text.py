"""Run MedCalc-Bench-Verified through the repo's API-based evaluation surface."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional

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


@dataclass
class ScoreResult:
    is_correct: bool
    scoring_mode: str
    parsed_prediction: str
    normalized_prediction: str
    failure_reason: str


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
    return parser.parse_args()


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if max_words <= 0 or len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def load_prompt(
    case_text: str,
    question: str,
    output_type: str,
    relevant_entities: str,
    max_case_words: int,
) -> str:
    instruction = (
        "Read the patient note and answer the medical calculation question. "
        "Return only the final answer with no explanation."
    )
    if output_type in NUMERIC_OUTPUT_TYPES:
        instruction += " If the answer is numeric, return only the final numeric value."
    elif output_type == DATE_OUTPUT_TYPE:
        instruction += " If the answer is a date, return only the final date."

    truncated_case_text = truncate_words(case_text, max_case_words)
    entities_block = (
        f"Relevant entities:\n{relevant_entities}\n\n" if relevant_entities else ""
    )

    return (
        f"{instruction}\n\n"
        f"{entities_block}"
        f"Patient note:\n{truncated_case_text}\n\n"
        f"Question:\n{question}\n\n"
        "Final answer:"
    )


def call_model(
    client: Any,
    model: str,
    prompt: str,
    retry_attempts: int,
    retry_delay: float,
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(retry_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=96,
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


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def decimal_from_string(value: str) -> Optional[Decimal]:
    cleaned = value.strip().replace(",", "")
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
        normalized_prediction = normalize_date_like(raw_prediction)
        normalized_target = normalize_date_like(ground_truth_answer)
        correct = normalized_prediction == normalized_target
        return ScoreResult(
            is_correct=correct,
            scoring_mode="date_exact",
            parsed_prediction=normalized_prediction,
            normalized_prediction=normalized_prediction,
            failure_reason="" if correct else "date_mismatch",
        )

    normalized_prediction = normalize_text(raw_prediction)
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
        / slugify_model(model)
        / sample_label
    )


def create_summary_report(
    results_df: pd.DataFrame,
    *,
    split: str,
    provider: str,
    model: str,
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

    return {
        "timestamp": datetime.now().isoformat(),
        "n_cases": n_cases,
        "split": split,
        "provider": provider,
        "model": model,
        "metrics": {
            "accuracy": accuracy,
            "numeric_parse_rate": numeric_parse_rate,
            "accuracy_by_output_type": accuracy_by_type,
            "scoring_mode_counts": scoring_mode_counts,
            "failure_reason_counts": failure_reason_counts,
        },
        "runtime_seconds": runtime_seconds,
    }


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
        model=model,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    client = create_client(provider=provider)
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
            prompt = load_prompt(
                case_text=str(row["case_text"]),
                question=str(row["question"]),
                output_type=str(row["output_type"]),
                relevant_entities=str(row.get("relevant_entities", "")),
                max_case_words=args.max_case_words,
            )
            raw_prediction = call_model(
                client=client,
                model=model,
                prompt=prompt,
                retry_attempts=args.retry_attempts,
                retry_delay=args.retry_delay,
            )
            score = score_prediction(
                raw_prediction=raw_prediction,
                output_type=str(row["output_type"]),
                ground_truth_answer=str(row["ground_truth_answer"]),
                lower_limit=str(row["lower_limit"]),
                upper_limit=str(row["upper_limit"]),
            )

            result_row = dict(row)
            result_row.update(
                {
                    "raw_prediction": raw_prediction,
                    "parsed_prediction": score.parsed_prediction,
                    "normalized_prediction": score.normalized_prediction,
                    "is_correct": score.is_correct,
                    "scoring_mode": score.scoring_mode,
                    "failure_reason": score.failure_reason,
                }
            )
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
