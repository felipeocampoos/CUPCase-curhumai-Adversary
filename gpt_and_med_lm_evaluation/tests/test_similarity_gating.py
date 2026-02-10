"""Unit tests for semantic similarity gating utilities."""

import numpy as np
import pytest

from refinement.similarity_gating import (
    Candidate,
    compute_similarity_for_top3,
    parse_candidate_set,
    parse_discriminator_result,
    parse_free_text_discriminator_output,
)


class FakeEmbeddingService:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode_texts(self, texts):
        return np.vstack([self.mapping[text] for text in texts])


def test_compute_similarity_triggers_gate_above_threshold():
    service = FakeEmbeddingService(
        {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.9, 0.1]),
            "c": np.array([0.85, 0.15]),
        }
    )

    result = compute_similarity_for_top3(["a", "b", "c"], threshold=0.65, embedding_service=service)

    assert result.gate_triggered is True
    assert result.mean_cosine >= 0.65
    assert set(result.pairwise_cosine.keys()) == {"0-1", "0-2", "1-2"}


def test_compute_similarity_does_not_trigger_gate_below_threshold():
    service = FakeEmbeddingService(
        {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
            "c": np.array([1.0, 1.0]),
        }
    )

    result = compute_similarity_for_top3(["a", "b", "c"], threshold=0.95, embedding_service=service)

    assert result.gate_triggered is False


def test_compute_similarity_requires_three_candidates():
    with pytest.raises(ValueError):
        compute_similarity_for_top3(["a", "b"])  # pragma: no cover


def test_parse_candidate_set_json():
    text = """
    {
      "candidates": [
        {"label": "Dx A", "confidence": 0.7, "evidence": "Reason A"},
        {"label": "Dx B", "confidence": 0.2, "evidence": "Reason B"},
        {"label": "Dx C", "confidence": 0.1, "evidence": "Reason C"}
      ]
    }
    """

    result = parse_candidate_set(text)

    assert len(result.candidates) == 3
    assert result.candidates[0].label == "Dx A"
    assert result.candidates[1].confidence == 0.2


def test_parse_discriminator_result_json():
    text = """
    {
      "final_choice": "Dx A",
      "differentiators": ["Feature 1", "Feature 2"],
      "rationale": "Dx A best matches feature set"
    }
    """

    result = parse_discriminator_result(text)

    assert result.final_choice == "Dx A"
    assert len(result.differentiators) == 2
    assert "feature" in result.rationale.lower()


def test_parse_free_text_discriminator_output():
    text = """
    {
      "response": {
        "final_diagnosis": "Dx A",
        "differential": ["Dx B", "Dx C"],
        "conditional_reasoning": "If x then y",
        "next_steps": ["step 1"]
      },
      "final_choice": "Dx A",
      "differentiators": ["A has marker M"],
      "rationale": "Marker M supports Dx A"
    }
    """

    parsed = parse_free_text_discriminator_output(text)

    assert parsed.response.final_diagnosis == "Dx A"
    assert parsed.discriminator.final_choice == "Dx A"
    assert parsed.discriminator.differentiators == ["A has marker M"]
