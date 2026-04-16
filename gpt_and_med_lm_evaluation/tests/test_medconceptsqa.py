from pathlib import Path

from prepare_hf_medconceptsqa import (
    answer_letter_to_index,
    build_default_output_path,
    convert_rows,
    extract_question_stem,
    filter_rows_for_subset,
    maybe_sample,
    normalize_subset_name,
    subset_to_dataset_config,
)
from run_medconceptsqa_mcq import build_results_path


def test_normalize_subset_name_accepts_vocab_and_level():
    assert normalize_subset_name("ICD10CM_EASY") == "icd10cm_easy"
    assert normalize_subset_name("atc") == "atc"


def test_extract_question_stem_strips_inline_options():
    question = (
        "What is the description of the medical code R91.1 in ICD10CM? "
        "A. Solitary pulmonary nodule B. Esophagostomy hemorrhage "
        "C. Poisoning by antiasthmatics, assault, sequela D. Corporo-venous occlusive erectile dysfunction"
    )
    assert (
        extract_question_stem(question)
        == "What is the description of the medical code R91.1 in ICD10CM?"
    )


def test_answer_letter_to_index_supports_letters_and_numbers():
    assert answer_letter_to_index("C") == 2
    assert answer_letter_to_index("4") == 3
    assert answer_letter_to_index("1") == 0
    assert answer_letter_to_index("0") == 0


def test_subset_to_dataset_config_uses_all_for_vocab_only_subset():
    assert subset_to_dataset_config("icd10cm") == "all"
    assert subset_to_dataset_config("atc") == "all"
    assert subset_to_dataset_config("icd10cm_easy") == "icd10cm_easy"


def test_filter_rows_for_subset_handles_vocab_only_subset():
    rows = [
        {"vocab": "ICD10CM", "level": "easy", "id": 1},
        {"vocab": "ICD10CM", "level": "hard", "id": 2},
        {"vocab": "ATC", "level": "easy", "id": 3},
    ]
    filtered = filter_rows_for_subset(rows, "icd10cm")
    assert [row["id"] for row in filtered] == [1, 2]


def test_filter_rows_for_subset_handles_vocab_and_level_subset():
    rows = [
        {"vocab": "ICD10CM", "level": "easy", "id": 1},
        {"vocab": "ICD10CM", "level": "hard", "id": 2},
        {"vocab": "ATC", "level": "easy", "id": 3},
    ]
    filtered = filter_rows_for_subset(rows, "icd10cm_easy")
    assert [row["id"] for row in filtered] == [1]


def test_convert_rows_uses_correct_answer_and_distractors():
    rows = [
        {
            "question_id": 7,
            "question": (
                "What is the description of code X? "
                "A. wrong one B. right answer C. wrong three D. wrong four"
            ),
            "answer_id": "B",
            "option1": "wrong one",
            "option2": "right answer",
            "option3": "wrong three",
            "option4": "wrong four",
        }
    ]

    converted = convert_rows(rows)

    assert converted == [
        {
            "id": "7",
            "clean text": "What is the description of code X?",
            "final diagnosis": "right answer",
            "distractor1": "wrong one",
            "distractor2": "wrong three",
            "distractor3": "wrong four",
        }
    ]


def test_maybe_sample_is_deterministic():
    rows = [{"id": str(i)} for i in range(10)]
    assert maybe_sample(rows, 3, 11) == maybe_sample(rows, 3, 11)


def test_build_default_output_path_includes_subset_split_and_sampling():
    path = build_default_output_path("icd10cm_easy", "dev", 5, 9)
    assert path == Path("datasets/generated/medconceptsqa/icd10cm_easy_dev_n5_seed9.csv")


def test_build_results_path_uses_standardized_layout():
    path = build_results_path(
        output_root="output/experiments/medconceptsqa",
        subset="icd10cm_easy",
        provider="huggingface_local",
        variant="baseline",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        sample_size=10,
        seed=42,
    )
    assert path == Path(
        "output/experiments/medconceptsqa/icd10cm_easy/huggingface_local/mcq/baseline/Qwen_Qwen2.5-0.5B-Instruct/n10_seed42/results.csv"
    )
