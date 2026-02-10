"""Regression tests for the domain-routed wrapper CLI argument handling."""

import types

from gpt_free_text_eval_refined_domain_routed import (
    apply_default_variant,
    has_variant_override,
    run,
)


def test_has_variant_override_detects_split_flag_form():
    assert has_variant_override(["--variant", "baseline"]) is True


def test_has_variant_override_detects_equals_form():
    assert has_variant_override(["--variant=baseline"]) is True


def test_apply_default_variant_when_missing():
    argv = ["gpt_free_text_eval_refined_domain_routed.py", "--batch-size", "10"]

    result = apply_default_variant(argv)

    assert result[-2:] == ["--variant", "domain_routed"]


def test_apply_default_variant_respects_split_override():
    argv = [
        "gpt_free_text_eval_refined_domain_routed.py",
        "--variant",
        "baseline",
    ]

    result = apply_default_variant(argv)

    assert result.count("--variant") == 1
    assert result[-1] == "baseline"


def test_apply_default_variant_respects_equals_override():
    argv = [
        "gpt_free_text_eval_refined_domain_routed.py",
        "--variant=baseline",
    ]

    result = apply_default_variant(argv)

    assert "--variant" not in result
    assert "--variant=baseline" in result


def test_run_respects_explicit_argv(monkeypatch):
    captured = {}

    def fake_main():
        import sys

        captured["argv"] = list(sys.argv)

    monkeypatch.setitem(
        __import__("sys").modules,
        "gpt_free_text_eval_refined",
        types.SimpleNamespace(main=fake_main),
    )

    run(["prog", "--batch-size", "10"])

    assert captured["argv"][-2:] == ["--variant", "domain_routed"]
