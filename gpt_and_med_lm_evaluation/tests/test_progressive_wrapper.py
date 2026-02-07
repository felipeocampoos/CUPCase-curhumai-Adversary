"""Tests for progressive-disclosure wrapper variant handling."""

from gpt_free_text_eval_refined_progressive_disclosure import (
    apply_default_variant,
    has_variant_override,
)


def test_has_variant_override_detects_split_flag_form():
    assert has_variant_override(["--variant", "baseline"]) is True


def test_has_variant_override_detects_equals_form():
    assert has_variant_override(["--variant=baseline"]) is True


def test_apply_default_variant_when_missing():
    argv = ["gpt_free_text_eval_refined_progressive_disclosure.py", "--batch-size", "10"]
    result = apply_default_variant(argv)
    assert result[-2:] == ["--variant", "progressive_disclosure"]


def test_apply_default_variant_respects_override():
    argv = [
        "gpt_free_text_eval_refined_progressive_disclosure.py",
        "--variant",
        "baseline",
    ]
    result = apply_default_variant(argv)
    assert result.count("--variant") == 1
