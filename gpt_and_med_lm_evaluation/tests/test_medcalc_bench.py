from pathlib import Path
import sys


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from prepare_hf_medcalc_bench import build_default_output_path, convert_rows, maybe_sample
from run_medcalc_bench_free_text import (
    build_prompt,
    build_results_dir,
    choose_consensus_candidate,
    extract_answer_segment,
    extract_answer_numeric,
    parse_candidate_prediction,
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
