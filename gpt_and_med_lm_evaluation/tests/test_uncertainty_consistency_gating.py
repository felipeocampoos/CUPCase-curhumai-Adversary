"""Unit tests for the uncertainty-consistency gated variant."""

from pathlib import Path
import sys


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from refinement.refiner import RefinerConfig
from refinement.schema import (
    ChecklistItemResult,
    ClinicalQuality,
    CriticResult,
    DiagnosticResponse,
    HardFail,
    parse_diagnostic_response,
)
from refinement.similarity_gating import (
    Candidate,
    CandidateSet,
    DiscriminatorResult,
    FreeTextDiscriminatorOutput,
    parse_candidate_set,
    parse_free_text_discriminator_output,
)
from refinement.variants.uncertainty_consistency_gated import (
    UncertaintyConsistencyGatedRefiner,
)


class _FakeEmbeddingServiceHighSimilarity:
    def encode_texts(self, texts):
        import numpy as np

        return np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.85, 0.15],
            ]
        )


class _FakeEmbeddingServiceLowSimilarity:
    def encode_texts(self, texts):
        import numpy as np

        return np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )


def _critic_result(*, quality: int = 5, hard_fail: bool = False, failed_item: str | None = None, edit_plan=None):
    checklist = []
    for i in range(1, 9):
        item_id = f"C{i}"
        passed = item_id != failed_item
        checklist.append(
            ChecklistItemResult(
                item_id=item_id,
                passed=passed,
                rationale="ok" if passed else "needs work",
                suggested_fix=None if passed else "fix it",
            )
        )
    return CriticResult(
        checklist=checklist,
        clinical_quality=ClinicalQuality(score=quality, rationale="quality"),
        hard_fail=HardFail(failed=hard_fail, reason="unsafe" if hard_fail else None),
        edit_plan=edit_plan or ([] if failed_item is None else ["[GENERAL] Improve failed item"]),
    )


def test_variant_list_contains_uncertainty_consistency(create_refiner_variant=None):
    from refinement.variant_factory import list_refiner_variants

    assert "uncertainty_consistency_gated" in list_refiner_variants()


def test_uncertainty_consistency_variant_stable_path(monkeypatch):
    refiner = UncertaintyConsistencyGatedRefiner(
        client=object(),
        config=RefinerConfig(),
        embedding_service=_FakeEmbeddingServiceLowSimilarity(),
    )

    candidates = CandidateSet(
        candidates=[
            Candidate(label="Dx A", confidence=0.8),
            Candidate(label="Dx B", confidence=0.1),
            Candidate(label="Dx C", confidence=0.1),
        ],
        raw_response="{}",
    )

    def fake_call_api(model, prompt, parse_fn):
        if parse_fn is parse_candidate_set:
            return candidates, "{}"
        if parse_fn is parse_diagnostic_response:
            return DiagnosticResponse(
                final_diagnosis="Stable Dx",
                uncertainty="High confidence based on classic findings.",
                next_steps=["step 1"],
            ), "{}"
        raise AssertionError("Unexpected parse function")

    monkeypatch.setattr(refiner, "_call_api", fake_call_api)
    monkeypatch.setattr(refiner, "critique", lambda case_text, response: (_critic_result(), "{}"))

    response, _ = refiner.generate("Case text")

    assert response.final_diagnosis == "Stable Dx"
    metadata = refiner._get_case_variant_metadata()
    assert metadata["gate_triggered"] is False
    assert metadata["escalation_tier"] == "stable"
    assert metadata["final_selection_source"] == "baseline_generator"


def test_uncertainty_consistency_variant_guided_retry(monkeypatch):
    refiner = UncertaintyConsistencyGatedRefiner(
        client=object(),
        config=RefinerConfig(),
        embedding_service=_FakeEmbeddingServiceLowSimilarity(),
    )

    candidates = CandidateSet(
        candidates=[
            Candidate(label="Dx A", confidence=0.8),
            Candidate(label="Dx B", confidence=0.1),
            Candidate(label="Dx C", confidence=0.1),
        ],
        raw_response="{}",
    )

    def fake_call_api(model, prompt, parse_fn):
        if parse_fn is parse_candidate_set:
            return candidates, "{}"
        if parse_fn is parse_diagnostic_response:
            return DiagnosticResponse(
                final_diagnosis="Draft Dx",
                uncertainty="Moderate confidence; further testing may help.",
                next_steps=["step 1"],
            ), "{}"
        raise AssertionError("Unexpected parse function")

    monkeypatch.setattr(refiner, "_call_api", fake_call_api)
    monkeypatch.setattr(refiner, "critique", lambda case_text, response: (_critic_result(), "{}"))
    monkeypatch.setattr(
        refiner,
        "edit",
        lambda case_text, response, edit_plan: (
            DiagnosticResponse(final_diagnosis="Guided Retry Dx", next_steps=["step 2"]),
            "{}",
        ),
    )

    response, _ = refiner.generate("Case text")

    assert response.final_diagnosis == "Guided Retry Dx"
    metadata = refiner._get_case_variant_metadata()
    assert metadata["gate_triggered"] is True
    assert metadata["escalation_tier"] == "guided_retry"
    assert metadata["final_selection_source"] == "guided_retry"


def test_uncertainty_consistency_variant_discriminator_path(monkeypatch):
    refiner = UncertaintyConsistencyGatedRefiner(
        client=object(),
        config=RefinerConfig(),
        embedding_service=_FakeEmbeddingServiceHighSimilarity(),
    )

    candidates = CandidateSet(
        candidates=[
            Candidate(label="Dx A", confidence=0.45),
            Candidate(label="Dx B", confidence=0.35),
            Candidate(label="Dx C", confidence=0.20),
        ],
        raw_response="{}",
    )
    discriminator_output = FreeTextDiscriminatorOutput(
        response=DiagnosticResponse(
            final_diagnosis="Dx B",
            differential=["Dx A", "Dx C"],
            conditional_reasoning="Dx B best explains discriminator findings",
            next_steps=["step 1"],
        ),
        discriminator=DiscriminatorResult(
            final_choice="Dx B",
            differentiators=["marker B", "response pattern"],
            rationale="Dx B aligns best with the verifier concerns",
            raw_response="{}",
        ),
    )

    def fake_call_api(model, prompt, parse_fn):
        if parse_fn is parse_candidate_set:
            return candidates, "{}"
        if parse_fn is parse_diagnostic_response:
            return DiagnosticResponse(
                final_diagnosis="Draft Dx",
                uncertainty="Moderate confidence; need more data.",
                next_steps=["step 0"],
            ), "{}"
        if parse_fn is parse_free_text_discriminator_output:
            return discriminator_output, "{}"
        raise AssertionError("Unexpected parse function")

    monkeypatch.setattr(refiner, "_call_api", fake_call_api)
    monkeypatch.setattr(
        refiner,
        "critique",
        lambda case_text, response: (_critic_result(quality=2, failed_item="C4"), "{}"),
    )

    response, _ = refiner.generate("Case text")

    assert response.final_diagnosis == "Dx B"
    metadata = refiner._get_case_variant_metadata()
    assert metadata["gate_triggered"] is True
    assert metadata["escalation_tier"] == "discriminator"
    assert metadata["final_selection_source"] == "discriminator_pass"
    assert metadata["discriminator_invoked"] is True


def test_uncertainty_consistency_variant_discriminator_falls_back_to_guided_retry(monkeypatch):
    refiner = UncertaintyConsistencyGatedRefiner(
        client=object(),
        config=RefinerConfig(),
        embedding_service=_FakeEmbeddingServiceHighSimilarity(),
    )

    candidates = CandidateSet(
        candidates=[
            Candidate(label="Dx A", confidence=0.45),
            Candidate(label="Dx B", confidence=0.35),
            Candidate(label="Dx C", confidence=0.20),
        ],
        raw_response="{}",
    )

    def fake_call_api(model, prompt, parse_fn):
        if parse_fn is parse_candidate_set:
            return candidates, "{}"
        if parse_fn is parse_diagnostic_response:
            return DiagnosticResponse(
                final_diagnosis="Draft Dx",
                uncertainty="Moderate confidence; need more data.",
                next_steps=["step 0"],
            ), "{}"
        if parse_fn is parse_free_text_discriminator_output:
            raise ValueError("discriminator failed")
        raise AssertionError("Unexpected parse function")

    monkeypatch.setattr(refiner, "_call_api", fake_call_api)
    monkeypatch.setattr(
        refiner,
        "critique",
        lambda case_text, response: (_critic_result(quality=2, failed_item="C4"), "{}"),
    )
    monkeypatch.setattr(
        refiner,
        "edit",
        lambda case_text, response, edit_plan: (
            DiagnosticResponse(final_diagnosis="Retry Fallback Dx", next_steps=["step 2"]),
            "{}",
        ),
    )

    response, _ = refiner.generate("Case text")

    assert response.final_diagnosis == "Retry Fallback Dx"
    metadata = refiner._get_case_variant_metadata()
    assert metadata["final_selection_source"] == "guided_retry_fallback_after_discriminator_error"


def test_uncertainty_consistency_variant_treats_verifier_failure_as_risk(monkeypatch):
    refiner = UncertaintyConsistencyGatedRefiner(
        client=object(),
        config=RefinerConfig(),
        embedding_service=_FakeEmbeddingServiceLowSimilarity(),
    )

    candidates = CandidateSet(
        candidates=[
            Candidate(label="Dx A", confidence=0.8),
            Candidate(label="Dx B", confidence=0.1),
            Candidate(label="Dx C", confidence=0.1),
        ],
        raw_response="{}",
    )

    def fake_call_api(model, prompt, parse_fn):
        if parse_fn is parse_candidate_set:
            return candidates, "{}"
        if parse_fn is parse_diagnostic_response:
            return DiagnosticResponse(
                final_diagnosis="Draft Dx",
                uncertainty="High confidence based on classic findings.",
                next_steps=["step 1"],
            ), "{}"
        raise AssertionError("Unexpected parse function")

    monkeypatch.setattr(refiner, "_call_api", fake_call_api)
    monkeypatch.setattr(refiner, "critique", lambda case_text, response: (_ for _ in ()).throw(ValueError("verifier down")))
    monkeypatch.setattr(
        refiner,
        "edit",
        lambda case_text, response, edit_plan: (
            DiagnosticResponse(final_diagnosis="Verifier Fallback Retry Dx", next_steps=["step 2"]),
            "{}",
        ),
    )

    response, _ = refiner.generate("Case text")

    assert response.final_diagnosis == "Verifier Fallback Retry Dx"
    metadata = refiner._get_case_variant_metadata()
    assert metadata["verifier_triggered"] is True
    assert metadata["signal_count"] == 1
    assert metadata["escalation_tier"] == "guided_retry"
