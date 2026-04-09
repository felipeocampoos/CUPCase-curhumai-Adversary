"""Tests for case-level refinement failure analysis."""

from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from refinement.analysis import analyze_runs, collect_run_artifacts, diagnosis_matches
from refinement.schema import DiagnosticResponse, RefinementTrace


CASE_TEXT = (
    "A 53-year-old man with cough, weight loss, dry mouth, sluggish pupils, and "
    "proximal weakness that improves after repeated muscle use."
)
TRUE_DIAGNOSIS = "Lambert-Eaton syndrome"


def _write_run(tmp_path: Path, variant: str, timestamp: str, trace: RefinementTrace, bertscore: float) -> Path:
    run_dir = tmp_path / variant
    run_dir.mkdir(parents=True, exist_ok=True)

    trace_path = run_dir / f"refinement_traces_{timestamp}.jsonl"
    trace_payload = trace.to_dict()
    trace_payload["_logged_at"] = "2026-04-08T12:00:00"
    trace_payload["_index"] = 0
    trace_path.write_text(json.dumps(trace_payload) + "\n", encoding="utf-8")

    csv_path = run_dir / f"gpt4_free_text_refined_{timestamp}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Case presentation",
                "True diagnosis",
                "Generated diagnosis",
                "BERTScore F1",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Case presentation": trace.case_text,
                "True diagnosis": trace.true_diagnosis,
                "Generated diagnosis": trace.extracted_final_diagnosis,
                "BERTScore F1": bertscore,
            }
        )
    return run_dir


def _make_trace(
    *,
    variant: str,
    variant_initial_diagnosis: str,
    final_diagnosis: str,
    variant_stage_metadata: dict,
    variant_metadata: dict | None = None,
    editor_recovered_case: bool = False,
    first_failure_iteration: int | None = None,
    hard_fail_any_iteration: bool = False,
    diagnosis_trajectory: list[str] | None = None,
) -> RefinementTrace:
    final_response = DiagnosticResponse(final_diagnosis=final_diagnosis, next_steps=["step"])
    initial_response = DiagnosticResponse(final_diagnosis=variant_initial_diagnosis, next_steps=["step"])
    return RefinementTrace(
        case_id="case-1",
        case_text=CASE_TEXT,
        true_diagnosis=TRUE_DIAGNOSIS,
        final_response=final_response,
        extracted_final_diagnosis=final_diagnosis,
        iterations_to_compliance=3 if editor_recovered_case else 1,
        is_compliant=True,
        iterations=[],
        variant_initial_response=initial_response,
        variant_initial_diagnosis=variant_initial_diagnosis,
        clinical_quality_score=5,
        hard_fail=hard_fail_any_iteration,
        hard_fail_any_iteration=hard_fail_any_iteration,
        first_failure_iteration=first_failure_iteration,
        editor_recovered_case=editor_recovered_case,
        variant_name=variant,
        variant_metadata=variant_metadata or variant_stage_metadata,
        variant_stage_metadata=variant_stage_metadata,
        diagnosis_trajectory=diagnosis_trajectory or [variant_initial_diagnosis, final_diagnosis],
    )


def test_diagnosis_matches_handles_lems_synonyms():
    assert diagnosis_matches(
        "Lambert-Eaton Myasthenic Syndrome (LEMS) likely secondary to small cell lung cancer",
        "Lambert-Eaton syndrome",
    )


def test_analyze_runs_emits_stage_aware_labels(tmp_path: Path):
    baseline_dir = _write_run(
        tmp_path,
        "baseline",
        "20260408_120001",
        _make_trace(
            variant="baseline",
            variant_initial_diagnosis="Lambert-Eaton Myasthenic Syndrome",
            final_diagnosis="Lambert-Eaton Myasthenic Syndrome",
            variant_stage_metadata={},
            diagnosis_trajectory=["Lambert-Eaton Myasthenic Syndrome"],
        ),
        bertscore=0.82,
    )
    progressive_dir = _write_run(
        tmp_path,
        "progressive",
        "20260408_120002",
        _make_trace(
            variant="progressive_disclosure",
            variant_initial_diagnosis="Myasthenia Gravis",
            final_diagnosis="Lambert-Eaton Myasthenic Syndrome",
            variant_stage_metadata={"final_selection_source": "full_case_revision"},
            editor_recovered_case=True,
            first_failure_iteration=1,
            hard_fail_any_iteration=True,
            diagnosis_trajectory=["Myasthenia Gravis", "Lambert-Eaton Myasthenic Syndrome"],
        ),
        bertscore=0.81,
    )
    discriminative_dir = _write_run(
        tmp_path,
        "discriminative",
        "20260408_120003",
        _make_trace(
            variant="discriminative_question",
            variant_initial_diagnosis="Lambert-Eaton Myasthenic Syndrome",
            final_diagnosis="Lambert-Eaton Myasthenic Syndrome",
            variant_stage_metadata={
                "discriminative_question": "Did edrophonium improve the weakness?",
                "question_target_variable": "Response to edrophonium",
                "extracted_answer": "unclear",
                "final_selection_source": "discriminative_integration",
            },
            diagnosis_trajectory=["Lambert-Eaton Myasthenic Syndrome"],
        ),
        bertscore=0.81,
    )
    semantic_dir = _write_run(
        tmp_path,
        "semantic",
        "20260408_120004",
        _make_trace(
            variant="semantic_similarity_gated",
            variant_initial_diagnosis="Lambert-Eaton Myasthenic Syndrome",
            final_diagnosis="Lambert-Eaton Myasthenic Syndrome",
            variant_stage_metadata={
                "gate_triggered": True,
                "final_selection_source": "discriminator_pass",
            },
            diagnosis_trajectory=["Lambert-Eaton Myasthenic Syndrome"],
        ),
        bertscore=0.81,
    )
    domain_dir = _write_run(
        tmp_path,
        "domain",
        "20260408_120005",
        _make_trace(
            variant="domain_routed",
            variant_initial_diagnosis=(
                "Lambert-Eaton Myasthenic Syndrome (LEMS) likely secondary to malignancy"
            ),
            final_diagnosis=(
                "Lambert-Eaton Myasthenic Syndrome (LEMS) likely secondary to malignancy"
            ),
            variant_stage_metadata={
                "predicted_domain": "neurology",
                "final_selection_source": "domain_route",
            },
            diagnosis_trajectory=[
                "Lambert-Eaton Myasthenic Syndrome (LEMS) likely secondary to malignancy"
            ],
        ),
        bertscore=0.65,
    )

    artifacts = collect_run_artifacts(
        [
            str(baseline_dir),
            str(progressive_dir),
            str(discriminative_dir),
            str(semantic_dir),
            str(domain_dir),
        ]
    )
    report = analyze_runs(artifacts)

    records = {record["variant"]: record for record in report["case_records"]}

    assert "progressive_disclosure_anchor_error" in records["progressive_disclosure"]["labels"]
    assert "critic_detected_and_recovered" in records["progressive_disclosure"]["labels"]
    assert records["progressive_disclosure"]["editor_recovered_case"] is True

    assert "question_generation_ungrounded" in records["discriminative_question"]["labels"]
    assert "variant_added_no_new_information" in records["discriminative_question"]["labels"]

    assert "similarity_gate_unnecessary" in records["semantic_similarity_gated"]["labels"]
    assert "variant_correct_but_metric_loss" in records["domain_routed"]["labels"]

    assert report["aggregate"]["editor_recoveries"] == 1
    assert report["by_variant"]["progressive_disclosure"]["variant_stage_successes"] == 0


def test_collect_run_artifacts_preserves_duplicate_case_scores(tmp_path: Path):
    run_dir = tmp_path / "baseline_dupes"
    run_dir.mkdir(parents=True, exist_ok=True)

    trace_a = _make_trace(
        variant="baseline",
        variant_initial_diagnosis="Lambert-Eaton Myasthenic Syndrome",
        final_diagnosis="Lambert-Eaton Myasthenic Syndrome",
        variant_stage_metadata={},
        diagnosis_trajectory=["Lambert-Eaton Myasthenic Syndrome"],
    )
    trace_b = _make_trace(
        variant="baseline",
        variant_initial_diagnosis="Myasthenia Gravis",
        final_diagnosis="Myasthenia Gravis",
        variant_stage_metadata={},
        diagnosis_trajectory=["Myasthenia Gravis"],
    )

    trace_path = run_dir / "refinement_traces_20260408_120010.jsonl"
    payloads = []
    for index, trace in enumerate([trace_a, trace_b]):
        payload = trace.to_dict()
        payload["_logged_at"] = "2026-04-08T12:00:00"
        payload["_index"] = index
        payloads.append(json.dumps(payload))
    trace_path.write_text("\n".join(payloads) + "\n", encoding="utf-8")

    csv_path = run_dir / "gpt4_free_text_refined_20260408_120010.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Case presentation",
                "True diagnosis",
                "Generated diagnosis",
                "BERTScore F1",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Case presentation": CASE_TEXT,
                "True diagnosis": TRUE_DIAGNOSIS,
                "Generated diagnosis": "Lambert-Eaton Myasthenic Syndrome",
                "BERTScore F1": 0.9,
            }
        )
        writer.writerow(
            {
                "Case presentation": CASE_TEXT,
                "True diagnosis": TRUE_DIAGNOSIS,
                "Generated diagnosis": "Myasthenia Gravis",
                "BERTScore F1": 0.1,
            }
        )

    artifacts = collect_run_artifacts([str(run_dir)])

    assert len(artifacts) == 1
    score_map = artifacts[0].bertscore_by_case_key
    assert list(score_map.values()) == [0.9, 0.1]

    report = analyze_runs(artifacts)
    records = sorted(report["case_records"], key=lambda record: record["case_occurrence_index"])
    assert records[0]["bertscore_f1"] == 0.9
    assert records[1]["bertscore_f1"] == 0.1
