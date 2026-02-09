"""Shared helpers for progressive disclosure and explicit belief revision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .schema import (
    DiagnosticResponse,
    extract_json_from_response,
    parse_diagnostic_response,
)


@dataclass
class EarlyCandidate:
    label: str
    confidence: float
    rationale: str = ""


@dataclass
class EarlyDifferential:
    candidates: List[EarlyCandidate]
    raw_response: str


@dataclass
class RevisionDecisionFreeText:
    response: DiagnosticResponse
    final_choice: str
    final_confidence: float
    revision_summary: str
    kept_hypotheses: List[str]
    dropped_hypotheses: List[str]
    added_hypotheses: List[str]
    contradiction_found: bool
    rationale: str
    raw_response: str


@dataclass
class RevisionDecisionMCQ:
    final_choice_index: int
    final_confidence: float
    revision_summary: str
    kept_indices: List[int]
    dropped_indices: List[int]
    added_indices: List[int]
    contradiction_found: bool
    rationale: str
    raw_response: str


@dataclass
class BeliefRevisionScores:
    anchoring_flag: bool
    confidence_instability_score: float
    revision_delta: float
    penalty_score: float


@dataclass
class EarlyRankedOptions:
    ranked_indices: List[int]
    confidences: List[float]
    rationale: str
    raw_response: str


def truncate_case_by_fraction(case_text: str, fraction: float = 0.2) -> str:
    words = case_text.split()
    if not words:
        return ""
    cutoff = max(1, int(len(words) * fraction))
    return " ".join(words[:cutoff])


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    return default


def parse_early_differential_free_text(text: str) -> EarlyDifferential:
    data = extract_json_from_response(text)
    items = data.get("candidates", [])
    if not isinstance(items, list) or len(items) < 2:
        raise ValueError("Expected at least two candidates")

    candidates: List[EarlyCandidate] = []
    for item in items[:3]:
        if not isinstance(item, dict):
            raise ValueError("candidate item must be object")
        label = str(item.get("label", "")).strip()
        confidence = float(item.get("confidence", 0.0))
        rationale = str(item.get("rationale", "")).strip()
        if not label:
            raise ValueError("candidate label is required")
        candidates.append(EarlyCandidate(label=label, confidence=confidence, rationale=rationale))

    return EarlyDifferential(candidates=candidates, raw_response=text)


def parse_early_ranked_option_indices_mcq(text: str, num_options: int) -> EarlyRankedOptions:
    data = extract_json_from_response(text)
    ranked = data.get("ranked_indices", [])
    if not isinstance(ranked, list) or len(ranked) < 2:
        raise ValueError("ranked_indices must contain at least 2 entries")

    parsed: List[int] = []
    for item in ranked[:3]:
        idx = int(item) - 1
        if idx < 0 or idx >= num_options:
            raise ValueError(f"ranked index out of range: {item}")
        if idx not in parsed:
            parsed.append(idx)
    if len(parsed) < 2:
        raise ValueError("ranked_indices must contain at least 2 unique entries")

    confidences_raw = data.get("confidences", [])
    confidences: List[float] = []
    if isinstance(confidences_raw, list):
        for value in confidences_raw[: len(parsed)]:
            try:
                confidences.append(float(value))
            except (TypeError, ValueError):
                confidences.append(0.0)

    while len(confidences) < len(parsed):
        confidences.append(0.0)

    rationale = str(data.get("rationale", "")).strip()
    return EarlyRankedOptions(
        ranked_indices=parsed,
        confidences=confidences,
        rationale=rationale,
        raw_response=text,
    )


def parse_revision_decision_free_text(text: str) -> RevisionDecisionFreeText:
    data = extract_json_from_response(text)

    response_data = data.get("response", {})
    if isinstance(response_data, dict):
        response_obj = DiagnosticResponse.from_dict(response_data)
    else:
        response_obj = parse_diagnostic_response(str(response_data))
    final_choice = str(data.get("final_choice", response_obj.final_diagnosis)).strip()
    if not final_choice:
        raise ValueError("final_choice is required")

    final_confidence = float(data.get("final_confidence", 0.0))
    revision_summary = str(data.get("revision_summary", "")).strip()
    kept_hypotheses = [str(x).strip() for x in data.get("kept_hypotheses", []) if str(x).strip()]
    dropped_hypotheses = [str(x).strip() for x in data.get("dropped_hypotheses", []) if str(x).strip()]
    added_hypotheses = [str(x).strip() for x in data.get("added_hypotheses", []) if str(x).strip()]
    contradiction_found = parse_bool(data.get("contradiction_found"), default=False)
    rationale = str(data.get("rationale", "")).strip()

    response_obj.final_diagnosis = final_choice

    return RevisionDecisionFreeText(
        response=response_obj,
        final_choice=final_choice,
        final_confidence=final_confidence,
        revision_summary=revision_summary,
        kept_hypotheses=kept_hypotheses,
        dropped_hypotheses=dropped_hypotheses,
        added_hypotheses=added_hypotheses,
        contradiction_found=contradiction_found,
        rationale=rationale,
        raw_response=text,
    )


def parse_revision_decision_mcq(
    text: str,
    num_options: int,
    candidate_indices: Sequence[int] | None = None,
) -> RevisionDecisionMCQ:
    data = extract_json_from_response(text)

    raw_final_idx = int(data.get("final_choice_index", 0)) - 1
    if candidate_indices:
        # Prefer rank-relative decoding because prompt presents only ranked subset.
        if 0 <= raw_final_idx < len(candidate_indices):
            final_idx = int(candidate_indices[raw_final_idx])
        elif 0 <= raw_final_idx < num_options:
            final_idx = raw_final_idx
        else:
            raise ValueError("final_choice_index out of range")
    else:
        final_idx = raw_final_idx
        if final_idx < 0 or final_idx >= num_options:
            raise ValueError("final_choice_index out of range")

    def parse_indices(key: str) -> List[int]:
        values = data.get(key, [])
        if not isinstance(values, list):
            return []
        parsed: List[int] = []
        for value in values:
            raw_idx = int(value) - 1
            idx: int | None = None
            if candidate_indices and 0 <= raw_idx < len(candidate_indices):
                idx = int(candidate_indices[raw_idx])
            elif 0 <= raw_idx < num_options:
                idx = raw_idx
            if idx is not None and idx not in parsed:
                parsed.append(idx)
        return parsed

    return RevisionDecisionMCQ(
        final_choice_index=final_idx,
        final_confidence=float(data.get("final_confidence", 0.0)),
        revision_summary=str(data.get("revision_summary", "")).strip(),
        kept_indices=parse_indices("kept_indices"),
        dropped_indices=parse_indices("dropped_indices"),
        added_indices=parse_indices("added_indices"),
        contradiction_found=parse_bool(data.get("contradiction_found"), default=False),
        rationale=str(data.get("rationale", "")).strip(),
        raw_response=text,
    )


def compute_belief_revision_scores(
    early_top_label: str,
    final_label: str,
    early_top_confidence: float,
    final_confidence: float,
    contradiction_found: bool,
    kept_labels: Sequence[str],
) -> BeliefRevisionScores:
    changed = early_top_label.strip().lower() != final_label.strip().lower()

    anchoring_flag = contradiction_found and not changed

    instability = 0.0
    if changed and early_top_confidence >= 0.8:
        instability = min(1.0, max(0.0, early_top_confidence - final_confidence))

    revision_delta = 1.0 if changed else 0.0
    if not changed and early_top_label in kept_labels:
        revision_delta = 0.2

    unexplained_revision = 1.0 if changed and not contradiction_found else 0.0

    penalty = (0.5 * float(anchoring_flag)) + (0.3 * instability) + (0.2 * unexplained_revision)
    penalty = max(0.0, min(1.0, penalty))

    return BeliefRevisionScores(
        anchoring_flag=anchoring_flag,
        confidence_instability_score=instability,
        revision_delta=revision_delta,
        penalty_score=penalty,
    )


def format_early_candidates_for_prompt(candidates: Sequence[EarlyCandidate]) -> str:
    lines: List[str] = []
    for i, c in enumerate(candidates, start=1):
        lines.append(f"{i}. {c.label} (confidence={c.confidence:.3f})")
        if c.rationale:
            lines.append(f"   rationale: {c.rationale}")
    return "\n".join(lines)


def format_early_options_for_prompt(options: Sequence[str], ranked_indices: Sequence[int], confidences: Sequence[float]) -> str:
    lines: List[str] = []
    for pos, idx in enumerate(ranked_indices, start=1):
        conf = confidences[pos - 1] if pos - 1 < len(confidences) else 0.0
        lines.append(f"{pos}. option {idx + 1}: {options[idx]} (confidence={conf:.3f})")
    return "\n".join(lines)
