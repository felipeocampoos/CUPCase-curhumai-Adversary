from pathlib import Path
import sys


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from prepare_hf_medcalc_bench import build_default_output_path, convert_rows, maybe_sample
from run_medcalc_bench_free_text import extract_answer_numeric, score_prediction


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
