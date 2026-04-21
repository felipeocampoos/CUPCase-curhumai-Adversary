from pathlib import Path
import sys
from types import SimpleNamespace


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from prepare_hf_medcalc_bench import build_default_output_path, convert_rows, maybe_sample
from run_medcalc_bench_free_text import (
    build_prompt,
    build_results_dir,
    candidate_agreement_signal,
    choose_consensus_candidate,
    execute_method,
    extract_answer_segment,
    extract_answer_numeric,
    has_uncertainty_signal,
    parse_candidate_prediction,
    parse_verifier_output,
    score_prediction,
    select_icl_example,
)


def test_convert_rows_maps_hf_fields_to_normalized_schema():
    rows = [
        {
            "Row Number": 1,
            "Note ID": "pmc-1",
            "Patient Note": "Patient note text",
            "Question": "What is the value?",
            "Relevant Entities": "{'age': [16, 'years']}",
            "Ground Truth Answer": "141.0403",
            "Lower Limit": "133.98828",
            "Upper Limit": "148.09232",
            "Ground Truth Explanation": "Explanation text",
            "Calculator Name": "Creatinine Clearance",
            "Category": "lab test",
            "Output Type": "decimal",
            "Note Type": "Extracted",
        }
    ]

    assert convert_rows(rows) == [
        {
            "id": "pmc-1:1",
            "case_text": "Patient note text",
            "question": "What is the value?",
            "relevant_entities": "{'age': [16, 'years']}",
            "ground_truth_answer": "141.0403",
            "lower_limit": "133.98828",
            "upper_limit": "148.09232",
            "ground_truth_explanation": "Explanation text",
            "calculator_name": "Creatinine Clearance",
            "category": "lab test",
            "output_type": "decimal",
            "note_type": "Extracted",
        }
    ]


def test_maybe_sample_is_deterministic():
    rows = [{"id": str(i)} for i in range(10)]
    assert maybe_sample(rows, 3, 11) == maybe_sample(rows, 3, 11)


def test_build_default_output_path_includes_split_and_sampling():
    path = build_default_output_path("test", 5, 9)
    assert path == Path("datasets/generated/medcalc_bench/test_n5_seed9.csv")


def test_build_results_dir_includes_method():
    path = build_results_dir("output/experiments/medcalc_bench", "test", "huggingface_local", "zero_shot_cot", "model/name", 5, 9)
    assert path == Path(
        "output/experiments/medcalc_bench/test/huggingface_local/free_text/zero_shot_cot/model_name/n5_seed9"
    )


def test_score_prediction_uses_numeric_interval():
    result = score_prediction(
        raw_prediction="Final answer: 141.04 mL/min",
        output_type="decimal",
        ground_truth_answer="141.0403",
        lower_limit="133.98828",
        upper_limit="148.09232",
    )

    assert result.is_correct is True
    assert result.scoring_mode == "numeric_interval"
    assert result.parsed_prediction == "141.04"


def test_score_prediction_uses_numeric_exact_when_bounds_missing():
    result = score_prediction(
        raw_prediction="79.333330",
        output_type="decimal",
        ground_truth_answer="79.33333",
        lower_limit="",
        upper_limit="",
    )

    assert result.is_correct is True
    assert result.scoring_mode == "numeric_exact"


def test_score_prediction_handles_date_exact_match():
    result = score_prediction(
        raw_prediction="January 2, 2024",
        output_type="date",
        ground_truth_answer="2024-01-02",
        lower_limit="",
        upper_limit="",
    )

    assert result.is_correct is True
    assert result.scoring_mode == "date_exact"


def test_extract_answer_numeric_ignores_trailing_unit_number():
    parsed = extract_answer_numeric("75 mL/min/1.73 m²")
    assert parsed is not None
    assert str(parsed) == "75"


def test_extract_answer_numeric_prefers_final_answer_segment():
    parsed = extract_answer_numeric("Range 70 to 80. Final answer: 75 mL/min/1.73 m²")
    assert parsed is not None
    assert str(parsed) == "75"


def test_extract_answer_segment_handles_chat_style_answer():
    assert (
        extract_answer_segment("The answer is Glanzmann's thrombasthenia")
        == "Glanzmann's thrombasthenia"
    )


def test_select_icl_example_is_deterministic():
    examples = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert select_icl_example(examples, example_id=None, seed=11) == select_icl_example(
        examples,
        example_id=None,
        seed=11,
    )


def test_select_icl_example_excludes_current_row():
    examples = [{"id": "a"}, {"id": "b"}]
    selected = select_icl_example(
        examples,
        example_id=None,
        seed=0,
        exclude_ids=["a"],
    )
    assert selected == {"id": "b"}


def test_build_prompt_one_shot_includes_worked_example():
    example = {
        "id": "demo-1",
        "case_text": "Example note",
        "question": "Example question",
        "relevant_entities": "",
        "ground_truth_explanation": "Example reasoning",
        "ground_truth_answer": "42",
    }
    prompt = build_prompt(
        method="one_shot_cot",
        case_text="Real note",
        question="Real question",
        output_type="decimal",
        relevant_entities="",
        max_case_words=250,
        icl_example=example,
    )

    assert "Worked example:" in prompt
    assert "Final answer: 42" in prompt
    assert "Now solve the real case." in prompt


def test_choose_consensus_candidate_returns_majority_numeric_answer():
    candidates = [
        parse_candidate_prediction("Final answer: 75 mL/min/1.73 m²", "decimal"),
        parse_candidate_prediction("75", "decimal"),
        parse_candidate_prediction("Final answer: 80", "decimal"),
    ]
    consensus = choose_consensus_candidate(candidates)
    assert consensus is not None
    assert consensus.normalized_prediction == "75"


def test_candidate_agreement_signal_triggers_on_disagreement():
    candidates = [
        parse_candidate_prediction("Final answer: 75", "decimal"),
        parse_candidate_prediction("Final answer: 80", "decimal"),
        parse_candidate_prediction("Final answer: 75", "decimal"),
    ]

    assert candidate_agreement_signal(candidates) is True


def test_has_uncertainty_signal_detects_hedge_and_missing_final_answer():
    assert has_uncertainty_signal("I am not sure, maybe 75") is True
    assert has_uncertainty_signal("Final answer: 75") is False


def test_has_uncertainty_signal_ignores_likely_in_reasoning_when_final_answer_is_stable():
    assert (
        has_uncertainty_signal("This likely reflects the expected value.\nFinal answer: 75")
        is False
    )


def test_parse_verifier_output_handles_structured_response():
    result = parse_verifier_output(
        "Label: supported\nRisk: 0.20\nRationale: The note supports the value.",
        0.5,
    )

    assert result.label == "supported"
    assert result.signal_triggered is False
    assert result.risk_score == 0.2


def test_execute_method_uncertainty_gate_uses_guided_retry(monkeypatch):
    call_prompts = []
    responses = iter(
        [
            "Final answer: 75",
            "Final answer: 75",
            "Final answer: 75",
            "Maybe 75",
            "Label: supported\nRisk: 0.00\nRationale: Supported by the note.",
            "Final answer: 75",
        ]
    )

    def fake_call_model(**kwargs):
        call_prompts.append(kwargs["prompt"])
        return next(responses)

    monkeypatch.setattr("run_medcalc_bench_free_text.call_model", fake_call_model)

    args = SimpleNamespace(
        method="medcalc_uncertainty_consistency_gate",
        max_case_words=250,
        retry_attempts=1,
        retry_delay=0,
        num_candidates=3,
        candidate_temperature=0.7,
        max_tokens=128,
        verifier_risk_threshold=0.5,
        icl_example_id=None,
        icl_seed=42,
    )
    row = {
        "id": "demo-1",
        "case_text": "Patient note text",
        "question": "What is the value?",
        "output_type": "decimal",
        "relevant_entities": "",
    }

    result = execute_method(
        client=object(),
        model="demo",
        row=row,
        args=args,
        icl_examples=None,
    )

    assert result.raw_prediction == "Final answer: 75"
    assert result.metadata["selection_source"] == "guided_retry"
    assert result.metadata["retry_invoked"] is True
    assert result.metadata["adjudication_invoked"] is False
    assert result.metadata["signal_count"] == 1
    assert "Revise the answer carefully" in call_prompts[-1]


def test_execute_method_uncertainty_gate_uses_adjudication(monkeypatch):
    responses = iter(
        [
            "Final answer: 75",
            "Final answer: 80",
            "Final answer: 90",
            "Final answer: 75",
            "Label: unsupported\nRisk: 0.90\nRationale: Candidate values conflict.",
            "Final answer: 80",
        ]
    )

    monkeypatch.setattr(
        "run_medcalc_bench_free_text.call_model",
        lambda **kwargs: next(responses),
    )

    args = SimpleNamespace(
        method="medcalc_uncertainty_consistency_gate",
        max_case_words=250,
        retry_attempts=1,
        retry_delay=0,
        num_candidates=3,
        candidate_temperature=0.7,
        max_tokens=128,
        verifier_risk_threshold=0.5,
        icl_example_id=None,
        icl_seed=42,
    )
    row = {
        "id": "demo-2",
        "case_text": "Patient note text",
        "question": "What is the value?",
        "output_type": "decimal",
        "relevant_entities": "",
    }

    result = execute_method(
        client=object(),
        model="demo",
        row=row,
        args=args,
        icl_examples=None,
    )

    assert result.raw_prediction == "Final answer: 80"
    assert result.metadata["selection_source"] == "adjudication"
    assert result.metadata["adjudication_invoked"] is True
    assert result.metadata["signal_count"] >= 2


def test_execute_method_uncertainty_gate_stable_path_keeps_verified_baseline(monkeypatch):
    responses = iter(
        [
            "Final answer: 80",
            "Final answer: 80",
            "Final answer: 80",
            "Final answer: 75",
            "Label: supported\nRisk: 0.00\nRationale: Supported by the note.",
        ]
    )

    monkeypatch.setattr(
        "run_medcalc_bench_free_text.call_model",
        lambda **kwargs: next(responses),
    )

    args = SimpleNamespace(
        method="medcalc_uncertainty_consistency_gate",
        max_case_words=250,
        retry_attempts=1,
        retry_delay=0,
        num_candidates=3,
        candidate_temperature=0.7,
        max_tokens=128,
        verifier_risk_threshold=0.5,
        icl_example_id=None,
        icl_seed=42,
    )
    row = {
        "id": "demo-3",
        "case_text": "Patient note text",
        "question": "What is the value?",
        "output_type": "decimal",
        "relevant_entities": "",
    }

    result = execute_method(
        client=object(),
        model="demo",
        row=row,
        args=args,
        icl_examples=None,
    )

    assert result.raw_prediction == "Final answer: 75"
    assert result.metadata["selection_source"] == "baseline_generator"
    assert result.metadata["gate_triggered"] is False


def test_candidate_agreement_signal_is_neutral_for_single_candidate():
    candidate = parse_candidate_prediction("Final answer: 75", "decimal")
    assert candidate_agreement_signal([candidate]) is False
