"""Unit tests for discriminative-question shared utilities."""

import pytest

from refinement.discriminative_questioning import (
    parse_answer_extraction,
    parse_discriminative_question,
    parse_integrated_decision_free_text,
    parse_integrated_decision_mcq,
    parse_ranked_candidates,
    parse_ranked_option_indices,
)


def test_parse_ranked_candidates():
    text = """
    {
      "candidates": [
        {"label": "Dx A", "confidence": 0.6, "rationale": "fits x"},
        {"label": "Dx B", "confidence": 0.3, "rationale": "fits y"}
      ]
    }
    """
    parsed = parse_ranked_candidates(text)

    assert len(parsed.candidates) == 2
    assert parsed.candidates[0].label == "Dx A"


def test_parse_ranked_option_indices():
    parsed, rationale = parse_ranked_option_indices(
        '{"ranked_indices": [2, 4, 1], "rationale": "rank reason"}',
        num_options=4,
    )
    assert parsed[:2] == [1, 3]
    assert rationale == "rank reason"


def test_parse_discriminative_question():
    text = '{"question": "Is there orthopnea?", "target_variable": "orthopnea", "rationale": "key separator"}'
    parsed = parse_discriminative_question(text)

    assert parsed.question == "Is there orthopnea?"
    assert parsed.target_variable == "orthopnea"


def test_parse_answer_extraction():
    text = """
    {
      "answer": "yes",
      "confidence": 0.82,
      "evidence_spans": ["reports orthopnea at night"],
      "rationale": "explicit symptom mention"
    }
    """
    parsed = parse_answer_extraction(text)

    assert parsed.answer == "yes"
    assert parsed.confidence == pytest.approx(0.82)
    assert parsed.evidence_spans == ["reports orthopnea at night"]


def test_parse_integrated_decision_free_text():
    text = """
    {
      "response": {
        "final_diagnosis": "Heart failure exacerbation",
        "conditional_reasoning": "Orthopnea favors heart failure",
        "next_steps": ["BNP", "echo"]
      },
      "final_choice": "Heart failure exacerbation",
      "integration_summary": "Orthopnea increased confidence for cardiac cause",
      "rationale": "Integrated answer supports cardiac etiology"
    }
    """
    parsed = parse_integrated_decision_free_text(text)

    assert parsed.response.final_diagnosis == "Heart failure exacerbation"
    assert "Orthopnea" in parsed.decision.integration_summary


def test_parse_integrated_decision_mcq():
    text = '{"final_choice_index": 3, "integration_summary": "answer favors option 3", "rationale": "better fit"}'
    final_idx, summary, rationale = parse_integrated_decision_mcq(text, num_options=4)

    assert final_idx == 2
    assert summary == "answer favors option 3"
    assert rationale == "better fit"
