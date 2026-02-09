"""Unit tests for progressive disclosure helpers."""

from refinement.progressive_disclosure import (
    compute_belief_revision_scores,
    parse_early_differential_free_text,
    parse_early_ranked_option_indices_mcq,
    parse_revision_decision_mcq,
    parse_revision_decision_free_text,
    truncate_case_by_fraction,
)


def test_truncate_case_by_fraction_non_empty():
    text = "one two three four five six seven eight nine ten"
    result = truncate_case_by_fraction(text, fraction=0.2)
    assert result == "one two"


def test_parse_early_differential_free_text():
    parsed = parse_early_differential_free_text(
        '{"candidates":[{"label":"Dx A","confidence":0.6,"rationale":"r1"},{"label":"Dx B","confidence":0.3,"rationale":"r2"}]}'
    )
    assert len(parsed.candidates) == 2
    assert parsed.candidates[0].label == "Dx A"


def test_parse_early_ranked_option_indices_mcq():
    parsed = parse_early_ranked_option_indices_mcq(
        '{"ranked_indices":[2,1,3],"confidences":[0.55,0.3,0.15],"rationale":"early"}',
        4,
    )
    assert parsed.ranked_indices == [1, 0, 2]
    assert parsed.confidences[0] == 0.55


def test_parse_early_ranked_option_indices_mcq_rejects_duplicates():
    try:
        parse_early_ranked_option_indices_mcq('{"ranked_indices":[1,1]}', 4)
        assert False, "Expected ValueError for duplicate ranked indices"
    except ValueError as exc:
        assert "unique" in str(exc)


def test_parse_revision_decision_free_text():
    parsed = parse_revision_decision_free_text(
        '{"response":{"final_diagnosis":"Dx B","next_steps":["step"]},"final_choice":"Dx B","final_confidence":0.7,"revision_summary":"changed","kept_hypotheses":["Dx A"],"dropped_hypotheses":["Dx C"],"added_hypotheses":["Dx B"],"contradiction_found":true,"rationale":"full case"}'
    )
    assert parsed.final_choice == "Dx B"
    assert parsed.contradiction_found is True


def test_parse_revision_decision_mcq():
    parsed = parse_revision_decision_mcq(
        '{"final_choice_index":3,"final_confidence":0.66,"revision_summary":"changed","kept_indices":[2],"dropped_indices":[1],"added_indices":[3],"contradiction_found":true,"rationale":"full"}',
        4,
    )
    assert parsed.final_choice_index == 2
    assert parsed.contradiction_found is True


def test_parse_revision_decision_mcq_string_false():
    parsed = parse_revision_decision_mcq(
        '{"final_choice_index":3,"final_confidence":0.66,"revision_summary":"changed","kept_indices":[2],"dropped_indices":[1],"added_indices":[3],"contradiction_found":"false","rationale":"full"}',
        4,
    )
    assert parsed.contradiction_found is False


def test_parse_revision_decision_mcq_prefers_absolute_index_with_candidates():
    parsed = parse_revision_decision_mcq(
        '{"final_choice_index":1,"final_confidence":0.66,"revision_summary":"changed","kept_indices":[1,2],"dropped_indices":[3],"added_indices":[2],"contradiction_found":true,"rationale":"full"}',
        4,
        candidate_indices=[3, 1, 2],
    )
    assert parsed.final_choice_index == 0
    assert parsed.kept_indices == [0, 1]
    assert parsed.dropped_indices == [2]
    assert parsed.added_indices == [1]


def test_parse_revision_decision_mcq_falls_back_to_rank_mapping():
    parsed = parse_revision_decision_mcq(
        '{"final_choice_index":3,"final_confidence":0.66,"revision_summary":"changed","kept_indices":[3],"dropped_indices":[2],"added_indices":[3],"contradiction_found":true,"rationale":"full"}',
        2,
        candidate_indices=[1, 0, 1],
    )
    assert parsed.final_choice_index == 1
    assert parsed.kept_indices == [1]
    assert parsed.dropped_indices == [1]
    assert parsed.added_indices == [1]


def test_parse_revision_decision_free_text_string_false():
    parsed = parse_revision_decision_free_text(
        '{"response":{"final_diagnosis":"Dx B","next_steps":["step"]},"final_choice":"Dx B","final_confidence":0.7,"revision_summary":"changed","kept_hypotheses":["Dx A"],"dropped_hypotheses":["Dx C"],"added_hypotheses":["Dx B"],"contradiction_found":"false","rationale":"full case"}'
    )
    assert parsed.contradiction_found is False


def test_compute_belief_revision_scores_flags_instability():
    scores = compute_belief_revision_scores(
        early_top_label="Dx A",
        final_label="Dx B",
        early_top_confidence=0.9,
        final_confidence=0.3,
        contradiction_found=True,
        kept_labels=[],
    )
    assert scores.anchoring_flag is False
    assert scores.confidence_instability_score > 0
    assert scores.penalty_score > 0
