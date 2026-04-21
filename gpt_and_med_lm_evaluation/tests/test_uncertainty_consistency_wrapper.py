"""Regression tests for uncertainty-consistency wrapper CLI handling."""

from pathlib import Path
import types
import sys


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from gpt_free_text_eval_refined_uncertainty_consistency import (
    apply_default_variant,
    has_variant_override,
    run,
)


def test_has_variant_override_detects_split_flag_form():
    assert has_variant_override(["--variant", "baseline"]) is True


def test_has_variant_override_detects_equals_form():
    assert has_variant_override(["--variant=baseline"]) is True


def test_apply_default_variant_when_missing():
    argv = ["gpt_free_text_eval_refined_uncertainty_consistency.py", "--batch-size", "10"]
    result = apply_default_variant(argv)

    assert result[-2:] == ["--variant", "uncertainty_consistency_gated"]


def test_apply_default_variant_respects_override():
    argv = [
        "gpt_free_text_eval_refined_uncertainty_consistency.py",
        "--variant",
        "baseline",
    ]
    result = apply_default_variant(argv)

    assert result.count("--variant") == 1
    assert result[-1] == "baseline"


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

    assert captured["argv"][-2:] == ["--variant", "uncertainty_consistency_gated"]
