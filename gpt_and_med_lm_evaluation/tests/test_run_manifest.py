"""Tests for run manifest comparability checks."""

from __future__ import annotations

from refinement.run_manifest import compare_run_manifests


def _manifest(*, commit_sha: str = "abc123", overrides: dict | None = None) -> dict:
    manifest = {
        "provider": "openai",
        "model": "gpt-4o",
        "dataset": {
            "input_path": "datasets/CUPCASE_RTEST_eval_20.csv",
            "dataset_preset": "custom",
        },
        "config": {
            "max_iterations": 3,
            "clinical_threshold": 3,
            "similarity_threshold": 0.65,
            "disclosure_fraction": 0.2,
            "early_confidence_threshold": 0.8,
            "revision_instability_threshold": 0.5,
            "curiosity_threshold": 0,
            "humility_threshold": 0,
            "random_seed": 42,
            "sampling_mode": "unique",
            "n_batches": 1,
            "batch_size": 20,
        },
        "git": {
            "commit_sha": commit_sha,
        },
    }
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(manifest.get(key), dict):
                manifest[key] = {**manifest[key], **value}
            else:
                manifest[key] = value
    return manifest


def test_compare_run_manifests_rejects_different_git_commits():
    baseline = _manifest(commit_sha="abc123")
    refined = _manifest(commit_sha="def456")

    result = compare_run_manifests(baseline, refined)

    assert result["is_comparable"] is False
    assert any(mismatch["field"] == "git.commit_sha" for mismatch in result["mismatches"])


def test_compare_run_manifests_rejects_refinement_knob_mismatches():
    baseline = _manifest()
    refined = _manifest(
        overrides={
            "config": {
                "max_iterations": 5,
                "clinical_threshold": 4,
                "similarity_threshold": 0.7,
            }
        }
    )

    result = compare_run_manifests(baseline, refined)

    assert result["is_comparable"] is False
    mismatch_fields = {mismatch["field"] for mismatch in result["mismatches"]}
    assert "config.max_iterations" in mismatch_fields
    assert "config.clinical_threshold" in mismatch_fields
    assert "config.similarity_threshold" in mismatch_fields
