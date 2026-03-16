"""Tests for MCQ refined helper logic."""

import pandas as pd
import pytest

from gpt_qa_eval_refined import (
    VALID_VARIANTS,
    extract_distractors,
    parse_args,
    parse_discriminator_choice,
    parse_predicted_index,
    parse_ranked_indices,
)


def test_parse_predicted_index_valid():
    assert parse_predicted_index("3", 4) == 2


def test_parse_predicted_index_invalid():
    assert parse_predicted_index("x", 4) == -1


def test_parse_predicted_index_ignores_incidental_numbers():
    text = "This 53-year-old patient is most consistent with option 2."
    assert parse_predicted_index(text, 4) == 1


def test_parse_predicted_index_from_structured_output():
    text = '{"final_choice_index": 4, "rationale": "best fit"}'
    assert parse_predicted_index(text, 4) == 3


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


def test_parse_discriminator_choice_rank_relative_mapping():
    final_idx, _, _ = parse_discriminator_choice(
        '{"final_choice_index": 1, "differentiators": [], "rationale": "why"}',
        4,
        candidate_indices=[3, 1, 2],
    )
    assert final_idx == 3


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


def test_extract_distractors_rejects_missing_values():
    row = pd.Series(
        {
            "distractor1": "Dx B",
            "distractor2": float("nan"),
            "distractor3": "",
            "distractor4": "Dx D",
        }
    )
    with pytest.raises(ValueError, match="Need at least 3 distractors"):
        extract_distractors(row, true_diagnosis="Dx A")


def test_valid_variants_include_discriminative_question():
    assert "discriminative_question" in VALID_VARIANTS


def test_valid_variants_include_progressive_disclosure():
    assert "progressive_disclosure" in VALID_VARIANTS


def test_parse_args_uses_provider_default_model_for_deepseek(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--provider", "deepseek"],
    )
    args = parse_args()
    assert args.model == "deepseek-chat"


def test_parse_args_preserves_explicit_model(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--provider", "deepseek", "--model", "custom-model"],
    )
    args = parse_args()
    assert args.model == "custom-model"


def test_parse_args_uses_env_default_model_for_openai_compatible(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL", "Qwen/Qwen3.5-35B")
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--provider", "openai_compatible"],
    )
    args = parse_args()
    assert args.model == "Qwen/Qwen3.5-35B"


def test_parse_args_uses_env_default_model_for_huggingface_local(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--provider", "huggingface_local"],
    )
    args = parse_args()
    assert args.model == "Qwen/Qwen2.5-0.5B-Instruct"
