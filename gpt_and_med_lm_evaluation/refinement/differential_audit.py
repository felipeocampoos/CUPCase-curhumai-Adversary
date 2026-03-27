"""Helpers for differential-audit candidate expansion and comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from .schema import DiagnosticResponse, extract_json_from_response


@dataclass
class CounterHypothesisSet:
    """Alternative diagnoses proposed against a seed candidate."""

    hypotheses: List[str]
    raw_response: str


@dataclass
class ComparativeDecision:
    """Comparative evaluation result over a pooled differential."""

    final_choice: str
    rationale: str
    evidence_for: Dict[str, List[str]]
    evidence_against: Dict[str, List[str]]
    missing_information: Dict[str, List[str]]
    raw_response: str


@dataclass
class ComparativeFreeTextOutput:
    """Structured free-text output from comparative evaluation."""

    response: DiagnosticResponse
    decision: ComparativeDecision


def _coerce_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def parse_counter_hypotheses(text: str) -> CounterHypothesisSet:
    """Parse a counter-hypothesis generation payload."""
    data = extract_json_from_response(text)
    payload = data.get("counter_hypotheses", data.get("alternatives", []))
    if not isinstance(payload, list):
        payload = [payload]

    hypotheses: List[str] = []
    seen: set[str] = set()
    for item in payload:
        label = ""
        if isinstance(item, dict):
            label = str(item.get("label", item.get("diagnosis", ""))).strip()
        else:
            label = str(item).strip()
        if not label:
            continue
        key = label.casefold()
        if key in seen:
            continue
        hypotheses.append(label)
        seen.add(key)
        if len(hypotheses) == 2:
            break

    if not hypotheses:
        raise ValueError("Expected at least 1 counter-hypothesis")

    return CounterHypothesisSet(hypotheses=hypotheses, raw_response=text)


def parse_comparative_evaluation_free_text(text: str) -> ComparativeFreeTextOutput:
    """Parse comparative differential evaluation output."""
    data = extract_json_from_response(text)

    response_payload: Dict[str, Any]
    if isinstance(data.get("response"), dict):
        response_payload = data["response"]
    else:
        response_payload = data

    response = DiagnosticResponse.from_dict(response_payload)
    if not response.final_diagnosis:
        raise ValueError("Comparative output missing final_diagnosis")

    final_choice = str(data.get("final_choice", response.final_diagnosis)).strip()
    if not final_choice:
        final_choice = response.final_diagnosis

    def parse_evidence_map(key: str) -> Dict[str, List[str]]:
        value = data.get(key, {})
        if not isinstance(value, dict):
            return {}
        return {
            str(label).strip(): _coerce_string_list(items)
            for label, items in value.items()
            if str(label).strip()
        }

    decision = ComparativeDecision(
        final_choice=final_choice,
        rationale=str(data.get("rationale", "")).strip(),
        evidence_for=parse_evidence_map("evidence_for"),
        evidence_against=parse_evidence_map("evidence_against"),
        missing_information=parse_evidence_map("missing_information"),
        raw_response=text,
    )

    response.final_diagnosis = final_choice
    return ComparativeFreeTextOutput(response=response, decision=decision)


def merge_differential_pool(
    seed_candidates: Sequence[str],
    counter_hypotheses_by_seed: Dict[str, Sequence[str]],
    max_total: int = 9,
) -> List[str]:
    """Merge seed candidates with deduplicated counter-hypotheses."""
    pooled: List[str] = []
    seen: set[str] = set()

    for label in seed_candidates:
        clean = str(label).strip()
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        pooled.append(clean)
        seen.add(key)
        if len(pooled) >= max_total:
            return pooled

    for seed in seed_candidates:
        seed_key = str(seed).strip().casefold()
        for label in counter_hypotheses_by_seed.get(seed, []):
            clean = str(label).strip()
            if not clean:
                continue
            key = clean.casefold()
            if key == seed_key or key in seen:
                continue
            pooled.append(clean)
            seen.add(key)
            if len(pooled) >= max_total:
                return pooled

    return pooled


def format_seed_candidates_for_prompt(seed_candidates: Sequence[str]) -> str:
    """Format ranked seed candidates for prompt context."""
    return "\n".join(
        f"{idx}. {label}" for idx, label in enumerate(seed_candidates, start=1)
    )


def format_pooled_differential_for_prompt(
    seed_candidates: Sequence[str],
    counter_hypotheses_by_seed: Dict[str, Sequence[str]],
    pooled_differential: Sequence[str],
) -> str:
    """Format the pooled differential for comparative evaluation prompts."""
    lines: List[str] = ["Seed candidates:"]
    for idx, label in enumerate(seed_candidates, start=1):
        lines.append(f"{idx}. {label}")
        alternatives = counter_hypotheses_by_seed.get(label, [])
        if alternatives:
            for alt in alternatives:
                lines.append(f"   counter-hypothesis: {alt}")

    lines.append("")
    lines.append("All diagnoses to compare:")
    for idx, label in enumerate(pooled_differential, start=1):
        lines.append(f"{idx}. {label}")

    return "\n".join(lines)
