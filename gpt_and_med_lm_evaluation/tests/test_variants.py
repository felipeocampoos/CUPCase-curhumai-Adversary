"""Unit tests for refinement variant framework."""

from typing import Any, Dict

from refinement.refiner import RefinerConfig
from refinement.schema import (
    ChecklistItemResult,
    ClinicalQuality,
    CriticResult,
    HardFail,
    DiagnosticResponse,
)
from refinement.variant_factory import (
    create_refiner_variant,
    list_refiner_variants,
)
from refinement.variants.domain_routed import (
    DomainRoutedRefiner,
    HeuristicDomainRouter,
)
from refinement.variants.semantic_similarity_gated import SemanticSimilarityGatedRefiner
from refinement.similarity_gating import (
    Candidate,
    CandidateSet,
    FreeTextDiscriminatorOutput,
    DiscriminatorResult,
    parse_candidate_set,
    parse_free_text_discriminator_output,
)
from refinement.schema import parse_diagnostic_response


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


def _compliant_critic_result() -> CriticResult:
    checklist = [
        ChecklistItemResult(
            item_id=f"C{i}",
            passed=True,
            rationale="ok",
            suggested_fix=None,
        )
        for i in range(1, 9)
    ]
    return CriticResult(
        checklist=checklist,
        clinical_quality=ClinicalQuality(score=5, rationale="high quality"),
        hard_fail=HardFail(failed=False, reason=None),
        edit_plan=[],
    )


def test_list_refiner_variants_contains_expected_entries():
    variants = list_refiner_variants()
    assert "baseline" in variants
    assert "domain_routed" in variants
    assert "semantic_similarity_gated" in variants


def test_create_refiner_variant_domain_routed():
    refiner = create_refiner_variant(
        variant="domain_routed",
        api_key="test-key",
        config=RefinerConfig(),
    )
    assert isinstance(refiner, DomainRoutedRefiner)


def test_create_refiner_variant_semantic_similarity_gated():
    refiner = create_refiner_variant(
        variant="semantic_similarity_gated",
        api_key="test-key",
        config=RefinerConfig(),
    )
    assert isinstance(refiner, SemanticSimilarityGatedRefiner)


def test_heuristic_router_routes_oncology_case():
    router = HeuristicDomainRouter()
    decision = router.route(
        "Patient with metastatic carcinoma on chemotherapy presents with neutropenic fever."
    )

    assert decision.domain == "oncology"
    assert decision.scores["oncology"] > 0


def test_domain_routed_generate_uses_domain_template(monkeypatch):
    refiner = DomainRoutedRefiner(client=object(), config=RefinerConfig())

    capture: Dict[str, Any] = {}

    def fake_call_api(model, prompt, parse_fn):
        capture["prompt"] = prompt
        return DiagnosticResponse(final_diagnosis="test dx", next_steps=["test step"]), "{}"

    monkeypatch.setattr(refiner, "_call_api", fake_call_api)

    response, _ = refiner.generate(
        "Fever, chills, blood culture positivity, and antibiotic exposure."
    )

    assert response.final_diagnosis == "test dx"
    assert "Specialty Focus: Infectious Disease" in capture["prompt"]


def test_domain_routed_trace_includes_variant_metadata(monkeypatch):
    config = RefinerConfig(max_iterations=1)
    refiner = DomainRoutedRefiner(client=object(), config=config)

    def fake_call_api(model, prompt, parse_fn):
        return DiagnosticResponse(
            final_diagnosis="Acute ischemic stroke",
            differential=["TIA"],
            next_steps=["Urgent MRI brain"],
        ), "{}"

    monkeypatch.setattr(refiner, "_call_api", fake_call_api)
    monkeypatch.setattr(
        refiner,
        "critique",
        lambda case_text, response: (_compliant_critic_result(), "{}"),
    )

    trace = refiner.refine(
        case_text="Sudden aphasia and right-sided weakness with concern for stroke.",
        case_id="case-1",
        true_diagnosis="Ischemic stroke",
    )

    assert trace.variant_name == "domain_routed"
    assert trace.variant_metadata["predicted_domain"] == "neurology"
    assert "domain_scores" in trace.variant_metadata


def test_semantic_similarity_variant_uses_baseline_when_not_gated(monkeypatch):
    refiner = SemanticSimilarityGatedRefiner(
        client=object(),
        config=RefinerConfig(),
        embedding_service=_FakeEmbeddingServiceLowSimilarity(),
    )

    candidates = CandidateSet(
        candidates=[
            Candidate(label="Dx A", confidence=0.6),
            Candidate(label="Dx B", confidence=0.3),
            Candidate(label="Dx C", confidence=0.1),
        ],
        raw_response="{}",
    )

    def fake_call_api(model, prompt, parse_fn):
        if parse_fn is parse_candidate_set:
            return candidates, "{}"
        if parse_fn is parse_diagnostic_response:
            return DiagnosticResponse(
                final_diagnosis="Baseline Dx",
                next_steps=["step 1"],
            ), "{}"
        raise AssertionError("Unexpected parse function in no-gate path")

    monkeypatch.setattr(refiner, "_call_api", fake_call_api)

    response, _ = refiner.generate("Case text")

    assert response.final_diagnosis == "Baseline Dx"
    metadata = refiner._get_case_variant_metadata()
    assert metadata["gate_triggered"] is False
    assert metadata["final_selection_source"] == "baseline_generator"


def test_semantic_similarity_variant_invokes_discriminator_when_gated(monkeypatch):
    refiner = SemanticSimilarityGatedRefiner(
        client=object(),
        config=RefinerConfig(),
        embedding_service=_FakeEmbeddingServiceHighSimilarity(),
    )

    candidates = CandidateSet(
        candidates=[
            Candidate(label="Dx A", confidence=0.45),
            Candidate(label="Dx B", confidence=0.30),
            Candidate(label="Dx C", confidence=0.25),
        ],
        raw_response="{}",
    )

    discriminator_output = FreeTextDiscriminatorOutput(
        response=DiagnosticResponse(
            final_diagnosis="Dx A",
            differential=["Dx B", "Dx C"],
            conditional_reasoning="A has discriminator features",
            next_steps=["step 1"],
        ),
        discriminator=DiscriminatorResult(
            final_choice="Dx A",
            differentiators=["marker M", "lab pattern L"],
            rationale="Dx A best matches differentiators",
            raw_response="{}",
        ),
    )

    def fake_call_api(model, prompt, parse_fn):
        if parse_fn is parse_candidate_set:
            return candidates, "{}"
        if parse_fn is parse_free_text_discriminator_output:
            return discriminator_output, "{}"
        raise AssertionError("Unexpected parse function in gated path")

    monkeypatch.setattr(refiner, "_call_api", fake_call_api)

    response, _ = refiner.generate("Case text")

    assert response.final_diagnosis == "Dx A"
    assert "Key differentiators" in (response.conditional_reasoning or "")
    metadata = refiner._get_case_variant_metadata()
    assert metadata["gate_triggered"] is True
    assert metadata["discriminator_invoked"] is True
    assert metadata["final_selection_source"] == "discriminator_pass"
