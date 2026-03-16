"""Tests for provider configuration and comparison report formatting."""

from __future__ import annotations

import sys
import types

sys.modules.setdefault("bert_score", types.SimpleNamespace(score=None))

from compare_baseline_vs_refined import format_text_report
from gpt_free_text_eval_refined import parse_args as parse_free_text_args
from refinement.refiner import JudgeProvider, create_client


def test_free_text_parse_args_uses_provider_default_model_for_deepseek(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--provider", "deepseek"],
    )
    args = parse_free_text_args()
    assert args.model == "deepseek-chat"


def test_free_text_parse_args_uses_env_default_for_openai_compatible(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL", "Qwen/Qwen3.5-0.8B")
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--provider", "openai_compatible"],
    )
    args = parse_free_text_args()
    assert args.model == "Qwen/Qwen3.5-0.8B"


def test_free_text_parse_args_uses_env_default_for_huggingface_local(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--provider", "huggingface_local"],
    )
    args = parse_free_text_args()
    assert args.model == "Qwen/Qwen2.5-0.5B-Instruct"


def test_create_client_openai_compatible_requires_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)

    try:
        create_client(provider=JudgeProvider.OPENAI_COMPATIBLE)
    except ValueError as exc:
        assert "OPENAI_COMPATIBLE_BASE_URL" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ValueError when base URL is missing")


def test_create_client_openai_compatible_uses_dummy_key(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, api_key, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.setattr("refinement.refiner.OpenAI", FakeOpenAI)

    create_client(provider=JudgeProvider.OPENAI_COMPATIBLE)

    assert captured == {
        "api_key": "dummy",
        "base_url": "http://localhost:8000/v1",
    }


def test_format_text_report_handles_declined_summary():
    report = {
        "timestamp": "2026-03-16T12:00:00",
        "inputs": {
            "baseline_path": "baseline.csv",
            "refined_path": "refined.csv",
            "n_aligned": 2,
        },
        "comparisons": {
            "BERTScore_F1": {
                "baseline_mean": 0.6,
                "refined_mean": 0.5,
                "bootstrap": {
                    "mean_difference": -0.1,
                    "ci_lower": -0.2,
                    "ci_upper": -0.01,
                },
                "permutation": {
                    "p_value": 0.04,
                },
            }
        },
        "summary": {
            "BERTScore_F1": {
                "direction": "declined",
                "delta": -0.1,
                "significant": True,
            }
        },
    }

    formatted = format_text_report(report)

    assert "DECLINED by 0.1000" in formatted
    assert "significantly DECLINED" in formatted
