from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types


sys.modules.setdefault("datasets", types.SimpleNamespace(Dataset=object))


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "lm_eval"
    / "tasks"
    / "med_calc_bench"
    / "utils.py"
)
SPEC = spec_from_file_location("med_calc_bench_utils", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
utils = module_from_spec(SPEC)
SPEC.loader.exec_module(utils)


def test_doc_to_text_builds_medcalc_prompt():
    doc = {
        "case_text": "Patient note text",
        "question": "What is the value?",
        "output_type": "decimal",
    }

    prompt = utils.doc_to_text(doc)

    assert "Patient note:" in prompt
    assert "Question:" in prompt
    assert "If the answer is numeric, return only the final numeric value." in prompt


def test_process_results_uses_interval_for_numeric_tasks():
    doc = {
        "output_type": "decimal",
        "ground_truth_answer": "141.0403",
        "lower_limit": "133.98828",
        "upper_limit": "148.09232",
    }

    result = utils.process_results(doc, ["Final answer: 141.04 mL/min"])

    assert result == {"exact_match": 1}


def test_process_results_uses_date_normalization():
    doc = {
        "output_type": "date",
        "ground_truth_answer": "2024-01-02",
        "lower_limit": "",
        "upper_limit": "",
    }

    result = utils.process_results(doc, ["January 2, 2024"])

    assert result == {"exact_match": 1}


def test_process_results_ignores_trailing_unit_number():
    doc = {
        "output_type": "decimal",
        "ground_truth_answer": "75",
        "lower_limit": "75",
        "upper_limit": "75",
    }

    result = utils.process_results(doc, ["75 mL/min/1.73 m²"])

    assert result == {"exact_match": 1}
