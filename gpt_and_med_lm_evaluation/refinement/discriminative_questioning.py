"""Shared utilities for self-generated discriminative questioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schema import DiagnosticResponse, extract_json_from_response


@dataclass
class RankedCandidate:
    """A ranked diagnosis candidate."""

    label: str
    confidence: Optional[float] = None
    rationale: Optional[str] = None


@dataclass
class RankedCandidateSet:
    """Top ranked candidates for a case."""

    candidates: List[RankedCandidate]
    raw_response: str


@dataclass
class DiscriminativeQuestion:
    """Single question to discriminate top candidates."""

    question: str
    target_variable: str
    rationale: str
    raw_response: str


@dataclass
class QuestionAnswerEvidence:
    """Extracted answer and evidence from case text."""

    answer: str
    confidence: float
    evidence_spans: List[str]
    rationale: str
    raw_response: str


@dataclass
class IntegratedDecision:
    """Decision object describing answer integration effect."""

    final_choice: str
    integration_summary: str
    rationale: str
    raw_response: str


@dataclass
class FreeTextIntegratedOutput:
    """Free-text integration output with structured response."""

    response: DiagnosticResponse
    decision: IntegratedDecision


def parse_ranked_candidates(text: str) -> RankedCandidateSet:
    """Parse top candidate JSON from model output."""
    data = extract_json_from_response(text)
    payload = data.get("candidates", data.get("top_candidates", []))

    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("Expected at least 2 ranked candidates")

    candidates: List[RankedCandidate] = []
    for item in payload:
        if isinstance(item, dict):
            label = str(item.get("label", "")).strip()
            if not label:
                raise ValueError("Candidate label cannot be empty")
            confidence = item.get("confidence")
            confidence_value = float(confidence) if confidence is not None else None
            rationale = item.get("rationale") or item.get("evidence")
            candidates.append(
                RankedCandidate(
                    label=label,
                    confidence=confidence_value,
                    rationale=str(rationale).strip() if rationale is not None else None,
                )
            )
        else:
            label = str(item).strip()
            if not label:
                raise ValueError("Candidate label cannot be empty")
            candidates.append(RankedCandidate(label=label))

    return RankedCandidateSet(candidates=candidates, raw_response=text)


def parse_ranked_option_indices(text: str, num_options: int) -> Tuple[List[int], str]:
    """Parse ranked MCQ option indices (1-based in prompt, 0-based returned)."""
    data = extract_json_from_response(text)
    ranked = data.get("ranked_indices", data.get("candidate_indices", []))
    rationale = str(data.get("rationale", "")).strip()

    if not isinstance(ranked, list):
        raise ValueError("ranked_indices must be a list")

    parsed: List[int] = []
    for item in ranked:
        idx = int(item) - 1
        if idx < 0 or idx >= num_options:
            raise ValueError(f"ranked index out of range: {item}")
        if idx not in parsed:
            parsed.append(idx)

    if len(parsed) < 2:
        raise ValueError("Need at least 2 unique ranked indices")

    return parsed, rationale


def parse_discriminative_question(text: str) -> DiscriminativeQuestion:
    """Parse single discriminative question payload."""
    data = extract_json_from_response(text)

    question = str(data.get("question", "")).strip()
    target_variable = str(data.get("target_variable", "")).strip()
    rationale = str(data.get("rationale", "")).strip()

    if not question:
        raise ValueError("Discriminative question missing `question`")
    if not target_variable:
        raise ValueError("Discriminative question missing `target_variable`")

    return DiscriminativeQuestion(
        question=question,
        target_variable=target_variable,
        rationale=rationale,
        raw_response=text,
    )


def parse_answer_extraction(text: str) -> QuestionAnswerEvidence:
    """Parse answer + evidence extraction payload."""
    data = extract_json_from_response(text)

    answer = str(data.get("answer", "")).strip()
    if not answer:
        raise ValueError("Answer extraction missing `answer`")

    confidence_raw = data.get("confidence", 0.0)
    confidence = float(confidence_raw)
    confidence = max(0.0, min(1.0, confidence))

    evidence_spans = data.get("evidence_spans", [])
    if not isinstance(evidence_spans, list):
        evidence_spans = [str(evidence_spans)]

    cleaned_spans = [str(item).strip() for item in evidence_spans if str(item).strip()]
    rationale = str(data.get("rationale", "")).strip()

    return QuestionAnswerEvidence(
        answer=answer,
        confidence=confidence,
        evidence_spans=cleaned_spans,
        rationale=rationale,
        raw_response=text,
    )


def parse_integrated_decision_free_text(text: str) -> FreeTextIntegratedOutput:
    """Parse free-text integration output payload."""
    data = extract_json_from_response(text)

    response_payload: Dict[str, Any]
    if isinstance(data.get("response"), dict):
        response_payload = data["response"]
    else:
        response_payload = data

    response = DiagnosticResponse.from_dict(response_payload)
    if not response.final_diagnosis:
        raise ValueError("Integrated free-text output missing final_diagnosis")

    final_choice = str(
        data.get("final_choice", response.final_diagnosis)
    ).strip() or response.final_diagnosis

    integration_summary = str(data.get("integration_summary", "")).strip()
    rationale = str(data.get("rationale", "")).strip()

    decision = IntegratedDecision(
        final_choice=final_choice,
        integration_summary=integration_summary,
        rationale=rationale,
        raw_response=text,
    )

    return FreeTextIntegratedOutput(response=response, decision=decision)


def parse_integrated_decision_mcq(text: str, num_options: int) -> Tuple[int, str, str]:
    """Parse integrated MCQ decision payload."""
    data = extract_json_from_response(text)

    final_choice = int(data.get("final_choice_index", -1)) - 1
    if final_choice < 0 or final_choice >= num_options:
        raise ValueError("Invalid integrated final_choice_index")

    integration_summary = str(data.get("integration_summary", "")).strip()
    rationale = str(data.get("rationale", "")).strip()

    return final_choice, integration_summary, rationale


def format_ranked_candidates_for_prompt(candidates: Sequence[RankedCandidate]) -> str:
    """Render ranked candidates for prompt context."""
    lines: List[str] = []
    for idx, candidate in enumerate(candidates, start=1):
        lines.append(f"{idx}. {candidate.label}")
        if candidate.confidence is not None:
            lines.append(f"   confidence: {candidate.confidence:.3f}")
        if candidate.rationale:
            lines.append(f"   rationale: {candidate.rationale}")
    return "\n".join(lines)


def format_evidence_for_prompt(answer: QuestionAnswerEvidence) -> str:
    """Render extracted answer evidence for integration prompt."""
    lines = [
        f"answer: {answer.answer}",
        f"confidence: {answer.confidence:.3f}",
    ]
    if answer.evidence_spans:
        lines.append("evidence_spans:")
        for span in answer.evidence_spans:
            lines.append(f"- {span}")
    if answer.rationale:
        lines.append(f"rationale: {answer.rationale}")
    return "\n".join(lines)
