"""Tests for MCQ refined helper logic."""

import pandas as pd
import pytest

from gpt_qa_eval_refined import (
    VALID_VARIANTS,
    extract_distractors,
    parse_discriminator_choice,
    parse_predicted_index,
    parse_ranked_indices,
)


def test_parse_predicted_index_valid():
    assert parse_predicted_index("3", 4) == 2


def test_parse_predicted_index_invalid():
    assert parse_predicted_index("x", 4) == -1


def test_parse_ranked_indices_json():
    indices, rationale = parse_ranked_indices(
        '{"ranked_indices": [2, 4, 1], "rationale": "test"}',
        4,
    )
    assert indices == [1, 3, 0]
    assert rationale == "test"


def test_parse_ranked_indices_requires_three():
    with pytest.raises(ValueError):
        parse_ranked_indices('{"ranked_indices": [1, 2]}', 4)


def test_parse_discriminator_choice_json():
    final_idx, differentiators, rationale = parse_discriminator_choice(
        '{"final_choice_index": 2, "differentiators": ["f1"], "rationale": "why"}',
        4,
    )
    assert final_idx == 1
    assert differentiators == ["f1"]
    assert rationale == "why"


def test_extract_distractors_prefers_unique_values():
    row = pd.Series(
        {
            "distractor1": "Dx B",
            "distractor2": "Dx C",
            "distractor3": "Dx D",
            "distractor4": "Dx D",
        }
    )
    distractors = extract_distractors(row, true_diagnosis="Dx A")
    assert distractors == ["Dx B", "Dx C", "Dx D"]


def test_valid_variants_include_discriminative_question():
    assert "discriminative_question" in VALID_VARIANTS
