"""Semantic similarity gating utilities for diagnostic candidate reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schema import DiagnosticResponse, extract_json_from_response


@dataclass
class Candidate:
    """Single diagnosis candidate."""

    label: str
    confidence: Optional[float] = None
    evidence: Optional[str] = None


@dataclass
class CandidateSet:
    """Ranked candidate output from the candidate-generation pass."""

    candidates: List[Candidate]
    raw_response: str


@dataclass
class SimilarityResult:
    """Pairwise similarity analysis over top candidates."""

    pairwise_cosine: Dict[str, float]
    mean_cosine: float
    gate_triggered: bool
    threshold: float


@dataclass
class DiscriminatorResult:
    """Output from a discriminator reasoning pass."""

    final_choice: str
    differentiators: List[str]
    rationale: str
    raw_response: str


@dataclass
class FreeTextDiscriminatorOutput:
    """Parsed free-text discriminator output."""

    response: DiagnosticResponse
    discriminator: DiscriminatorResult


class JinaEmbeddingService:
    """Embedding service wrapper for jinaai/jina-embeddings-v2-base-en."""

    _model = None

    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            from transformers import AutoModel

            cls._model = AutoModel.from_pretrained(
                "jinaai/jina-embeddings-v2-base-en",
                trust_remote_code=True,
            )
        return cls._model

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode text list to dense vectors (n_texts, dim)."""
        normalized = [" ".join(str(t).split()) for t in texts]
        vectors: List[Optional[np.ndarray]] = [None] * len(normalized)
        missing_texts: List[str] = []
        missing_positions: List[int] = []

        for idx, text in enumerate(normalized):
            if text in self._cache:
                vectors[idx] = self._cache[text]
            else:
                missing_texts.append(text)
                missing_positions.append(idx)

        if missing_texts:
            model = self._load_model()
            encoded = model.encode(missing_texts)
            for idx, vec in zip(missing_positions, encoded):
                arr = np.array(vec, dtype=np.float64)
                vectors[idx] = arr
                self._cache[normalized[idx]] = arr

        return np.vstack([v for v in vectors if v is not None])


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


def compute_similarity_for_top3(
    candidate_labels: Sequence[str],
    threshold: float = 0.65,
    embedding_service: Optional[JinaEmbeddingService] = None,
) -> SimilarityResult:
    """Compute pairwise cosine similarity and gate decision over top-3 labels."""
    if len(candidate_labels) < 3:
        raise ValueError("Need at least 3 candidates for similarity gating")

    service = embedding_service or JinaEmbeddingService()
    labels = list(candidate_labels[:3])
    vectors = service.encode_texts(labels)

    pairs = [(0, 1), (0, 2), (1, 2)]
    pairwise: Dict[str, float] = {}
    values: List[float] = []

    for left, right in pairs:
        score = _cosine_similarity(vectors[left], vectors[right])
        key = f"{left}-{right}"
        pairwise[key] = score
        values.append(score)

    mean_cosine = float(np.mean(values)) if values else 0.0
    return SimilarityResult(
        pairwise_cosine=pairwise,
        mean_cosine=mean_cosine,
        gate_triggered=mean_cosine >= threshold,
        threshold=threshold,
    )


def parse_candidate_set(text: str) -> CandidateSet:
    """Parse ranked top-candidate JSON payload."""
    data = extract_json_from_response(text)
    payload = data.get("candidates", data.get("top_candidates", []))

    if not isinstance(payload, list) or len(payload) < 3:
        raise ValueError("Candidate payload must include at least 3 candidates")

    candidates: List[Candidate] = []
    for item in payload[:3]:
        if isinstance(item, dict):
            label = str(item.get("label", "")).strip()
            if not label:
                raise ValueError("Candidate label cannot be empty")
            confidence = item.get("confidence")
            confidence_value = float(confidence) if confidence is not None else None
            evidence = item.get("evidence")
            candidates.append(
                Candidate(
                    label=label,
                    confidence=confidence_value,
                    evidence=str(evidence) if evidence is not None else None,
                )
            )
        else:
            label = str(item).strip()
            if not label:
                raise ValueError("Candidate label cannot be empty")
            candidates.append(Candidate(label=label))

    return CandidateSet(candidates=candidates, raw_response=text)


def parse_discriminator_result(text: str) -> DiscriminatorResult:
    """Parse generic discriminator JSON payload."""
    data = extract_json_from_response(text)

    final_choice = str(
        data.get("final_choice", data.get("final_diagnosis", ""))
    ).strip()
    if not final_choice:
        raise ValueError("Discriminator output missing final choice")

    differentiators = data.get("differentiators", [])
    if not isinstance(differentiators, list):
        differentiators = [str(differentiators)]

    rationale = str(data.get("rationale", "")).strip()

    return DiscriminatorResult(
        final_choice=final_choice,
        differentiators=[str(item).strip() for item in differentiators if str(item).strip()],
        rationale=rationale,
        raw_response=text,
    )


def parse_free_text_discriminator_output(text: str) -> FreeTextDiscriminatorOutput:
    """Parse discriminator output for free-text variant."""
    data = extract_json_from_response(text)

    response_payload: Dict[str, Any]
    if isinstance(data.get("response"), dict):
        response_payload = data["response"]
    else:
        response_payload = data

    response = DiagnosticResponse.from_dict(response_payload)
    if not response.final_diagnosis:
        raise ValueError("Discriminator response missing final_diagnosis")

    discriminator = parse_discriminator_result(text)

    return FreeTextDiscriminatorOutput(
        response=response,
        discriminator=discriminator,
    )


def format_candidates_for_prompt(candidates: Sequence[Candidate]) -> str:
    """Format candidate list for prompts."""
    lines = []
    for idx, candidate in enumerate(candidates, start=1):
        lines.append(f"{idx}. {candidate.label}")
        if candidate.confidence is not None:
            lines.append(f"   confidence: {candidate.confidence:.3f}")
        if candidate.evidence:
            lines.append(f"   evidence: {candidate.evidence}")
    return "\n".join(lines)


def format_similarity_for_prompt(result: SimilarityResult) -> str:
    """Format similarity metrics for prompts."""
    lines = [f"Mean cosine similarity: {result.mean_cosine:.4f}"]
    for key, value in result.pairwise_cosine.items():
        lines.append(f"pair {key}: {value:.4f}")
    return "\n".join(lines)
