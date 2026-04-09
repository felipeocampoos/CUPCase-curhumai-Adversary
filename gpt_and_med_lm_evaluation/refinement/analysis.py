"""Case-level refinement failure analysis utilities."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .io import hash_case_text, load_refinement_traces
from .schema import RefinementTrace


DIAGNOSIS_ALIASES: Dict[str, Sequence[str]] = {
    "lambert eaton syndrome": (
        "lambert eaton syndrome",
        "lambert eaton myasthenic syndrome",
        "lambert-eaton syndrome",
        "lambert-eaton myasthenic syndrome",
        "lems",
    ),
    "myasthenia gravis": ("myasthenia gravis",),
    "chronic obstructive pulmonary disease": (
        "chronic obstructive pulmonary disease",
        "copd",
    ),
    "congestive heart failure": ("congestive heart failure", "chf", "heart failure"),
    "pulmonary fibrosis": ("pulmonary fibrosis",),
}

UNCLEAR_ANSWER_MARKERS = {
    "",
    "unclear",
    "unknown",
    "not provided",
    "insufficient information",
    "cannot determine",
}

GENERIC_QUESTION_WORDS = {
    "response",
    "improve",
    "improved",
    "improvement",
    "patient",
    "history",
    "symptom",
    "symptoms",
    "weakness",
    "muscle",
    "change",
    "changes",
}


@dataclass
class RunArtifact:
    """Resolved trace input plus optional sibling metrics."""

    trace_path: Path
    variant_name: str
    traces: List[RefinementTrace]
    bertscore_by_case_key: Dict[Tuple[str, int], float]


def normalize_diagnosis(text: Optional[str]) -> str:
    """Normalize diagnosis text for lightweight correctness checks."""
    if not text:
        return ""

    normalized = text.lower()
    normalized = re.sub(r"\([^)]*\)", " ", normalized)
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[,;:]", " ", normalized)

    for marker in (
        " likely ",
        " probable ",
        " secondary to ",
        " due to ",
        " associated with ",
        " in setting of ",
        " caused by ",
        " with underlying ",
    ):
        if marker in normalized:
            normalized = normalized.split(marker, 1)[0]
            break

    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    for canonical, aliases in DIAGNOSIS_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return canonical
    return normalized


def diagnosis_matches(predicted: Optional[str], truth: Optional[str]) -> bool:
    """Return True when predicted and truth normalize to the same diagnosis."""
    predicted_norm = normalize_diagnosis(predicted)
    truth_norm = normalize_diagnosis(truth)
    return bool(predicted_norm and truth_norm and predicted_norm == truth_norm)


def collect_run_artifacts(inputs: Sequence[str]) -> List[RunArtifact]:
    """Resolve CLI inputs into trace artifacts with optional sibling metrics."""
    trace_paths: List[Path] = []
    for raw_input in inputs:
        path = Path(raw_input)
        if path.is_dir():
            trace_paths.extend(sorted(path.rglob("refinement_traces_*.jsonl")))
        elif path.is_file():
            trace_paths.append(path)

    artifacts: List[RunArtifact] = []
    for trace_path in sorted(set(trace_paths)):
        traces = load_refinement_traces(trace_path)
        if not traces:
            continue
        artifacts.append(
            RunArtifact(
                trace_path=trace_path,
                variant_name=traces[0].variant_name,
                traces=traces,
                bertscore_by_case_key=_load_sibling_bertscores(trace_path),
            )
        )
    return artifacts


def _load_sibling_bertscores(trace_path: Path) -> Dict[Tuple[str, int], float]:
    """Load case-level BERTScore from a sibling CSV if one is present."""
    trace_dir = trace_path.parent
    timestamp = trace_path.stem.removeprefix("refinement_traces_")
    csv_candidates = sorted(trace_dir.glob(f"*_{timestamp}.csv"))
    if not csv_candidates:
        csv_candidates = sorted(trace_dir.glob("gpt4_free_text_refined_*.csv"))
    if len(csv_candidates) != 1:
        return {}

    score_map: Dict[Tuple[str, int], float] = {}
    case_counts: Dict[str, int] = defaultdict(int)
    with open(csv_candidates[0], "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_text = row.get("Case presentation", "")
            score = row.get("BERTScore F1")
            if not case_text or not score:
                continue
            try:
                case_hash = hash_case_text(case_text)
                occurrence_index = case_counts[case_hash]
                score_map[(case_hash, occurrence_index)] = float(score)
                case_counts[case_hash] += 1
            except ValueError:
                continue
    return score_map


def analyze_runs(artifacts: Sequence[RunArtifact]) -> Dict[str, Any]:
    """Build a structured case-level report from one or more runs."""
    case_records: List[Dict[str, Any]] = []
    grouped_by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for artifact in artifacts:
        run_key = f"{artifact.variant_name}:{artifact.trace_path}"
        case_counts: Dict[str, int] = defaultdict(int)
        for trace in artifact.traces:
            case_hash = hash_case_text(trace.case_text)
            occurrence_index = case_counts[case_hash]
            case_counts[case_hash] += 1
            record = _build_case_record(
                trace,
                artifact,
                run_key,
                case_hash,
                occurrence_index,
            )
            case_records.append(record)
            grouped_by_hash[case_hash].append(record)

    comparative_cases: List[Dict[str, Any]] = []
    best_evidence_cases: List[Dict[str, Any]] = []
    for case_hash, records in grouped_by_hash.items():
        baseline_record = next((record for record in records if record["variant"] == "baseline"), None)
        for record in records:
            _apply_comparative_labels(record, baseline_record)

        comparative = _build_comparative_case(case_hash, records)
        comparative_cases.append(comparative)
        if comparative["best_evidence"]:
            best_evidence_cases.append(
                {
                    "case_hash": case_hash,
                    "true_diagnosis": comparative["true_diagnosis"],
                    "best_evidence": comparative["best_evidence"],
                }
            )

    aggregate = _build_aggregate_summary(case_records)
    by_variant = _build_variant_summary(case_records)

    return {
        "generated_at": datetime.now().isoformat(),
        "inputs": [str(artifact.trace_path) for artifact in artifacts],
        "runs": [
            {
                "trace_path": str(artifact.trace_path),
                "variant_name": artifact.variant_name,
                "n_cases": len(artifact.traces),
            }
            for artifact in artifacts
        ],
        "aggregate": aggregate,
        "by_variant": by_variant,
        "best_evidence_cases": best_evidence_cases,
        "case_records": case_records,
        "comparative_cases": comparative_cases,
    }


def _build_case_record(
    trace: RefinementTrace,
    artifact: RunArtifact,
    run_key: str,
    case_hash: str,
    occurrence_index: int,
) -> Dict[str, Any]:
    """Create a structured case record from a trace."""
    variant_stage_metadata = trace.variant_stage_metadata or trace.variant_metadata
    diagnosis_trajectory = trace.diagnosis_trajectory or [
        iteration.response.final_diagnosis
        for iteration in trace.iterations
    ] or [trace.extracted_final_diagnosis]
    variant_initial_diagnosis = (
        trace.variant_initial_diagnosis
        or (trace.iterations[0].response.final_diagnosis if trace.iterations else None)
        or trace.extracted_final_diagnosis
    )
    variant_initial_correct = diagnosis_matches(variant_initial_diagnosis, trace.true_diagnosis)
    final_correct = diagnosis_matches(trace.extracted_final_diagnosis, trace.true_diagnosis)
    hard_fail_any_iteration = trace.hard_fail_any_iteration or any(
        iteration.critic_result.hard_fail.failed
        for iteration in trace.iterations
        if iteration.critic_result is not None
    )
    first_failure_iteration = trace.first_failure_iteration
    if first_failure_iteration is None:
        for iteration in trace.iterations:
            critic = iteration.critic_result
            if critic is None:
                continue
            if critic.hard_fail.failed or not critic.is_compliant(quality_threshold=3):
                first_failure_iteration = iteration.iteration
                break

    editor_recovered_case = trace.editor_recovered_case
    if (
        not editor_recovered_case
        and trace.is_compliant
        and first_failure_iteration is not None
        and trace.iterations_to_compliance is not None
        and trace.iterations_to_compliance > first_failure_iteration
    ):
        editor_recovered_case = True

    labels: List[str] = []
    if not trace.true_diagnosis or not variant_initial_diagnosis:
        labels.append("analysis_insufficient_data")
    if not variant_initial_correct:
        labels.append("variant_wrong_initial_diagnosis")
    if editor_recovered_case:
        labels.append("critic_detected_and_recovered")
    if hard_fail_any_iteration and not final_correct:
        labels.append("critic_failed_to_recover")
    if trace.variant_name == "progressive_disclosure" and not variant_initial_correct:
        labels.append("progressive_disclosure_anchor_error")
    if trace.variant_name == "discriminative_question":
        answer = str(variant_stage_metadata.get("extracted_answer", "")).strip().lower()
        if answer in UNCLEAR_ANSWER_MARKERS and final_correct:
            labels.append("variant_added_no_new_information")
        if _question_looks_ungrounded(variant_stage_metadata, trace.case_text):
            labels.append("question_generation_ungrounded")
    if trace.variant_name == "domain_routed":
        predicted_domain = variant_stage_metadata.get("predicted_domain")
        if predicted_domain and predicted_domain == "general_medicine" and not final_correct:
            labels.append("routing_misfire")

    primary_failure_mode = _choose_primary_failure_mode(labels, final_correct)
    return {
        "run_key": run_key,
        "trace_path": str(artifact.trace_path),
        "case_id": trace.case_id,
        "case_hash": case_hash,
        "case_occurrence_index": occurrence_index,
        "variant": trace.variant_name,
        "true_diagnosis": trace.true_diagnosis,
        "variant_initial_diagnosis": variant_initial_diagnosis,
        "final_diagnosis": trace.extracted_final_diagnosis,
        "variant_initial_correct": variant_initial_correct,
        "final_correct": final_correct,
        "is_compliant": trace.is_compliant,
        "iterations_to_compliance": trace.iterations_to_compliance,
        "first_failure_iteration": first_failure_iteration,
        "hard_fail": trace.hard_fail,
        "hard_fail_any_iteration": hard_fail_any_iteration,
        "editor_recovered_case": editor_recovered_case,
        "diagnosis_trajectory": diagnosis_trajectory,
        "variant_selection_source": variant_stage_metadata.get("final_selection_source"),
        "variant_stage_metadata": variant_stage_metadata,
        "variant_metadata": trace.variant_metadata,
        "clinical_quality_score": trace.clinical_quality_score,
        "bertscore_f1": artifact.bertscore_by_case_key.get((case_hash, occurrence_index)),
        "labels": sorted(set(labels)),
        "primary_failure_mode": primary_failure_mode,
    }


def _apply_comparative_labels(record: Dict[str, Any], baseline_record: Optional[Dict[str, Any]]) -> None:
    """Add baseline-relative labels when a matching baseline case exists."""
    if baseline_record is None or record is baseline_record:
        return

    labels = set(record["labels"])
    if (
        record["variant"] == "semantic_similarity_gated"
        and record["variant_stage_metadata"].get("gate_triggered")
        and baseline_record["final_correct"]
        and record["final_correct"]
    ):
        labels.add("similarity_gate_unnecessary")

    baseline_score = baseline_record.get("bertscore_f1")
    variant_score = record.get("bertscore_f1")
    if (
        baseline_record["final_correct"]
        and record["final_correct"]
        and baseline_score is not None
        and variant_score is not None
        and variant_score + 1e-9 < baseline_score
    ):
        labels.add("variant_correct_but_metric_loss")

    record["labels"] = sorted(labels)
    record["primary_failure_mode"] = _choose_primary_failure_mode(record["labels"], record["final_correct"])


def _build_comparative_case(case_hash: str, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a baseline-relative view of a single case across variants."""
    baseline_record = next((record for record in records if record["variant"] == "baseline"), None)
    variants = {
        record["variant"]: {
            "variant_initial_correct": record["variant_initial_correct"],
            "final_correct": record["final_correct"],
            "editor_recovered_case": record["editor_recovered_case"],
            "primary_failure_mode": record["primary_failure_mode"],
            "bertscore_f1": record["bertscore_f1"],
        }
        for record in records
    }

    best_evidence: List[str] = []
    if baseline_record is not None:
        for record in records:
            if record is baseline_record:
                continue
            if not baseline_record["final_correct"] and record["variant_initial_correct"]:
                best_evidence.append(
                    f"{record['variant']} fixed the case at variant stage while baseline remained wrong"
                )
            elif not baseline_record["final_correct"] and record["final_correct"]:
                best_evidence.append(
                    f"{record['variant']} only succeeded end-to-end after refinement while baseline remained wrong"
                )

    true_diagnosis = records[0]["true_diagnosis"] if records else ""
    return {
        "case_hash": case_hash,
        "true_diagnosis": true_diagnosis,
        "variants": variants,
        "best_evidence": best_evidence,
    }


def _build_aggregate_summary(case_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate record counts across all analyzed cases."""
    label_counts = Counter(
        label
        for record in case_records
        for label in record["labels"]
    )
    return {
        "n_case_records": len(case_records),
        "variant_stage_successes": sum(record["variant_initial_correct"] for record in case_records),
        "end_to_end_successes": sum(record["final_correct"] for record in case_records),
        "editor_recoveries": sum(record["editor_recovered_case"] for record in case_records),
        "failure_mode_counts": dict(sorted(label_counts.items())),
    }


def _build_variant_summary(case_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize results per variant."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in case_records:
        grouped[record["variant"]].append(record)

    summary: Dict[str, Any] = {}
    for variant, records in grouped.items():
        scores = [record["bertscore_f1"] for record in records if record["bertscore_f1"] is not None]
        summary[variant] = {
            "n_cases": len(records),
            "variant_stage_successes": sum(record["variant_initial_correct"] for record in records),
            "end_to_end_successes": sum(record["final_correct"] for record in records),
            "editor_recoveries": sum(record["editor_recovered_case"] for record in records),
            "mean_bertscore_f1": (sum(scores) / len(scores)) if scores else None,
        }
    return dict(sorted(summary.items()))


def render_markdown_report(report: Dict[str, Any]) -> str:
    """Render a concise Markdown summary for the analysis report."""
    aggregate = report["aggregate"]
    lines = [
        "# Refinement Failure Analysis",
        "",
        "## Headline totals",
        "",
        f"- Case records analyzed: `{aggregate['n_case_records']}`",
        f"- Variant-stage successes: `{aggregate['variant_stage_successes']}`",
        f"- End-to-end successes: `{aggregate['end_to_end_successes']}`",
        f"- Editor recoveries: `{aggregate['editor_recoveries']}`",
        "",
        "## Failure modes",
        "",
    ]

    failure_counts = aggregate["failure_mode_counts"]
    if failure_counts:
        for label, count in failure_counts.items():
            lines.append(f"- `{label}`: `{count}`")
    else:
        lines.append("- No failure labels were assigned.")

    lines.extend(
        [
            "",
            "## Variant summary",
            "",
        ]
    )
    for variant, summary in report["by_variant"].items():
        lines.append(
            f"- `{variant}`: stage `{summary['variant_stage_successes']}/{summary['n_cases']}`, "
            f"final `{summary['end_to_end_successes']}/{summary['n_cases']}`, "
            f"recoveries `{summary['editor_recoveries']}`"
        )

    lines.extend(
        [
            "",
            "## Best evidence cases",
            "",
        ]
    )
    if report["best_evidence_cases"]:
        for case in report["best_evidence_cases"]:
            lines.append(f"- `{case['case_hash']}` ({case['true_diagnosis']}):")
            for evidence in case["best_evidence"]:
                lines.append(f"  - {evidence}")
    else:
        lines.append("- No case showed a clear variant win over baseline in the analyzed inputs.")

    lines.extend(
        [
            "",
            "## Cases requiring review",
            "",
        ]
    )
    flagged = [
        record for record in report["case_records"]
        if record["labels"] and "analysis_insufficient_data" not in record["labels"]
    ]
    if not flagged:
        lines.append("- No labeled failure cases in the analyzed inputs.")
    else:
        for record in flagged:
            lines.append(
                f"- `{record['variant']}` case `{record['case_hash']}`: "
                f"`{record['primary_failure_mode']}`; "
                f"trajectory `{record['diagnosis_trajectory']}`"
            )

    return "\n".join(lines) + "\n"


def _choose_primary_failure_mode(labels: Iterable[str], final_correct: bool) -> str:
    """Select one primary label from the assigned label set."""
    labels = set(labels)
    if not labels:
        return "end_to_end_success" if final_correct else "unlabeled_failure"

    priority = [
        "analysis_insufficient_data",
        "progressive_disclosure_anchor_error",
        "question_generation_ungrounded",
        "critic_failed_to_recover",
        "variant_wrong_initial_diagnosis",
        "critic_detected_and_recovered",
        "variant_correct_but_metric_loss",
        "variant_added_no_new_information",
        "similarity_gate_unnecessary",
        "routing_misfire",
    ]
    for label in priority:
        if label in labels:
            return label
    return sorted(labels)[0]


def _question_looks_ungrounded(metadata: Dict[str, Any], case_text: str) -> bool:
    """Heuristic for whether the generated discriminative question is unsupported by the case."""
    question = str(metadata.get("discriminative_question", "")).strip().lower()
    target = str(metadata.get("question_target_variable", "")).strip().lower()
    answer = str(metadata.get("extracted_answer", "")).strip().lower()
    if answer not in UNCLEAR_ANSWER_MARKERS:
        return False
    if not question and not target:
        return False

    case_words = set(re.findall(r"[a-z0-9]+", case_text.lower()))
    target_words = {
        word for word in re.findall(r"[a-z0-9]+", f"{question} {target}")
        if len(word) > 4 and word not in GENERIC_QUESTION_WORDS
    }
    if not target_words:
        return False
    return any(word not in case_words for word in target_words)
