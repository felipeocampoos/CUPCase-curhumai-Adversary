"""Regression tests for semantic-similarity wrapper CLI handling."""

from gpt_free_text_eval_refined_semantic_similarity import (
    apply_default_variant,
    has_variant_override,
)


def test_has_variant_override_detects_split_flag_form():
    assert has_variant_override(["--variant", "baseline"]) is True


def test_has_variant_override_detects_equals_form():
    assert has_variant_override(["--variant=baseline"]) is True


def test_apply_default_variant_when_missing():
    argv = ["gpt_free_text_eval_refined_semantic_similarity.py", "--batch-size", "10"]
    result = apply_default_variant(argv)

    assert result[-2:] == ["--variant", "semantic_similarity_gated"]


def test_apply_default_variant_respects_override():
    argv = [
        "gpt_free_text_eval_refined_semantic_similarity.py",
        "--variant",
        "baseline",
    ]
    result = apply_default_variant(argv)

    assert result.count("--variant") == 1
    assert result[-1] == "baseline"
