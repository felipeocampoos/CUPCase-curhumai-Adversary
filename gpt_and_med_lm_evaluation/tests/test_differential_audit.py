"""Unit tests for differential-audit helpers."""

from refinement.differential_audit import (
    merge_differential_pool,
    parse_comparative_evaluation_free_text,
    parse_counter_hypotheses,
)


def test_parse_counter_hypotheses():
    text = """
    {
      "counter_hypotheses": [
        {"label": "Primary myelofibrosis", "rationale": "dry tap and teardrop cells"},
        {"label": "Essential thrombocythemia", "rationale": "myeloproliferative differential"}
      ]
    }
    """
    parsed = parse_counter_hypotheses(text)

    assert parsed.hypotheses == ["Primary myelofibrosis", "Essential thrombocythemia"]


def test_parse_counter_hypotheses_deduplicates():
    text = """
    {
      "counter_hypotheses": [
        {"label": "Primary myelofibrosis"},
        {"label": "Primary myelofibrosis"},
        {"label": "Polycythemia vera"}
      ]
    }
    """
    parsed = parse_counter_hypotheses(text)

    assert parsed.hypotheses == ["Primary myelofibrosis", "Polycythemia vera"]


def test_parse_comparative_evaluation_free_text():
    text = """
    {
      "response": {
        "final_diagnosis": "Primary myelofibrosis",
        "conditional_reasoning": "dry tap and teardrop cells favor myelofibrosis",
        "next_steps": ["Bone marrow biopsy"]
      },
      "final_choice": "Primary myelofibrosis",
      "evidence_for": {
        "Primary myelofibrosis": ["dry tap", "teardrop cells"]
      },
      "evidence_against": {
        "Polycythemia vera": ["anemia rather than erythrocytosis"]
      },
      "missing_information": {
        "Chronic myeloid leukemia": ["BCR-ABL status"]
      },
      "rationale": "myelofibrosis best explains the smear and marrow findings"
    }
    """
    parsed = parse_comparative_evaluation_free_text(text)

    assert parsed.response.final_diagnosis == "Primary myelofibrosis"
    assert parsed.decision.evidence_for["Primary myelofibrosis"] == ["dry tap", "teardrop cells"]
    assert parsed.decision.evidence_against["Polycythemia vera"] == ["anemia rather than erythrocytosis"]


def test_merge_differential_pool_preserves_seed_order():
    pooled = merge_differential_pool(
        seed_candidates=["Dx A", "Dx B"],
        counter_hypotheses_by_seed={
            "Dx A": ["Dx C", "Dx D"],
            "Dx B": ["Dx E"],
        },
    )

    assert pooled == ["Dx A", "Dx B", "Dx C", "Dx D", "Dx E"]
