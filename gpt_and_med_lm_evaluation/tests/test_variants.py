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
from refinement.variants.discriminative_question import DiscriminativeQuestionRefiner
from refinement.variants.differential_audit import DifferentialAuditRefiner
from refinement.variants.progressive_disclosure import ProgressiveDisclosureRefiner
from refinement.similarity_gating import (
    Candidate,
    CandidateSet,
    FreeTextDiscriminatorOutput,
    DiscriminatorResult,
    parse_candidate_set,
    parse_free_text_discriminator_output,
)
from refinement.schema import parse_diagnostic_response
from refinement.discriminative_questioning import (
    FreeTextIntegratedOutput,
    IntegratedDecision,
    QuestionAnswerEvidence,
    DiscriminativeQuestion,
    RankedCandidateSet,
    RankedCandidate,
    parse_answer_extraction,
    parse_discriminative_question,
    parse_integrated_decision_free_text,
    parse_ranked_candidates,
)
from refinement.differential_audit import (
    ComparativeDecision,
    ComparativeFreeTextOutput,
    CounterHypothesisSet,
    merge_differential_pool,
    parse_comparative_evaluation_free_text,
    parse_counter_hypotheses,
)
from refinement.progressive_disclosure import (
    EarlyDifferential,
    EarlyCandidate,
    RevisionDecisionFreeText,
    BeliefRevisionScores,
    parse_early_differential_free_text,
    parse_revision_decision_free_text,
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
    assert "discriminative_question" in variants
    assert "differential_audit" in variants
    assert "progressive_disclosure" in variants


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


def test_create_refiner_variant_discriminative_question():
    refiner = create_refiner_variant(
        variant="discriminative_question",
        api_key="test-key",
        config=RefinerConfig(),
    )
    assert isinstance(refiner, DiscriminativeQuestionRefiner)


def test_create_refiner_variant_differential_audit():
    refiner = create_refiner_variant(
        variant="differential_audit",
        api_key="test-key",
        config=RefinerConfig(),
    )
    assert isinstance(refiner, DifferentialAuditRefiner)


def test_create_refiner_variant_progressive_disclosure():
    refiner = create_refiner_variant(
        variant="progressive_disclosure",
        api_key="test-key",
        config=RefinerConfig(),
    )
    assert isinstance(refiner, ProgressiveDisclosureRefiner)


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


def test_discriminative_question_variant_happy_path(monkeypatch):
    refiner = DiscriminativeQuestionRefiner(client=object(), config=RefinerConfig())

    ranked = RankedCandidateSet(
        candidates=[
            RankedCandidate(label="Cardiac cause", confidence=0.55),
            RankedCandidate(label="Pulmonary cause", confidence=0.35),
            RankedCandidate(label="Other", confidence=0.10),
        ],
        raw_response="{}",
    )
    question = DiscriminativeQuestion(
        question="Is there orthopnea?",
        target_variable="orthopnea",
        rationale="Distinguishes cardiac vs pulmonary pattern",
        raw_response="{}",
    )
    answer = QuestionAnswerEvidence(
        answer="yes",
        confidence=0.82,
        evidence_spans=["patient reports orthopnea at night"],
        rationale="Direct mention in case",
        raw_response="{}",
    )
    integrated = FreeTextIntegratedOutput(
        response=DiagnosticResponse(
            final_diagnosis="Heart failure exacerbation",
            conditional_reasoning="Orthopnea supports cardiac etiology",
            next_steps=["BNP", "Echo"],
        ),
        decision=IntegratedDecision(
            final_choice="Heart failure exacerbation",
            integration_summary="Orthopnea shifted confidence to cardiac cause",
            rationale="Answer integration favors cardiac differential",
            raw_response="{}",
        ),
    )

    def fake_call_api(model, prompt, parse_fn):
        if parse_fn is parse_ranked_candidates:
            return ranked, "{}"
        if parse_fn is parse_discriminative_question:
            return question, "{}"
        if parse_fn is parse_answer_extraction:
            return answer, "{}"
        if parse_fn is parse_integrated_decision_free_text:
            return integrated, "{}"
        raise AssertionError("Unexpected parse function")

    monkeypatch.setattr(refiner, "_call_api", fake_call_api)
    response, _ = refiner.generate("Case text")

    assert response.final_diagnosis == "Heart failure exacerbation"
    metadata = refiner._get_case_variant_metadata()
    assert metadata["discriminative_question"] == "Is there orthopnea?"
    assert metadata["extracted_answer"] == "yes"
    assert metadata["final_selection_source"] == "discriminative_integration"


def test_differential_audit_variant_happy_path_counter_hypothesis_wins(monkeypatch):
    refiner = DifferentialAuditRefiner(client=object(), config=RefinerConfig())

    ranked = RankedCandidateSet(
        candidates=[
            RankedCandidate(label="Polycythemia vera", confidence=0.61),
            RankedCandidate(label="Chronic myeloid leukemia", confidence=0.25),
            RankedCandidate(label="Hairy cell leukemia", confidence=0.14),
        ],
        raw_response="{}",
    )
    counter_sets = {
        "Polycythemia vera": CounterHypothesisSet(
            hypotheses=["Primary myelofibrosis", "Essential thrombocythemia"],
            raw_response="{}",
        ),
        "Chronic myeloid leukemia": CounterHypothesisSet(
            hypotheses=["Primary myelofibrosis", "Leukemoid reaction"],
            raw_response="{}",
        ),
        "Hairy cell leukemia": CounterHypothesisSet(
            hypotheses=["Splenic marginal zone lymphoma", "Primary myelofibrosis"],
            raw_response="{}",
        ),
    }
    comparative = ComparativeFreeTextOutput(
        response=DiagnosticResponse(
            final_diagnosis="Primary myelofibrosis",
            differential=["Polycythemia vera", "Chronic myeloid leukemia"],
            conditional_reasoning="Teardrop cells and dry tap favor myelofibrosis",
            next_steps=["Bone marrow biopsy review", "Molecular correlation"],
        ),
        decision=ComparativeDecision(
            final_choice="Primary myelofibrosis",
            rationale="Dry tap, teardrop cells, and splenomegaly outweigh alternatives",
            evidence_for={"Primary myelofibrosis": ["Teardrop cells", "Dry tap"]},
            evidence_against={"Polycythemia vera": ["Anemia rather than erythrocytosis"]},
            missing_information={"Chronic myeloid leukemia": ["BCR-ABL positivity"]},
            raw_response="{}",
        ),
    )

    def fake_call_api(model, prompt, parse_fn):
        if parse_fn is parse_ranked_candidates:
            return ranked, "{}"
        if parse_fn is parse_counter_hypotheses:
            if "Seed candidate to challenge\nPolycythemia vera" in prompt:
                return counter_sets["Polycythemia vera"], "{}"
            if "Seed candidate to challenge\nChronic myeloid leukemia" in prompt:
                return counter_sets["Chronic myeloid leukemia"], "{}"
            if "Seed candidate to challenge\nHairy cell leukemia" in prompt:
                return counter_sets["Hairy cell leukemia"], "{}"
            raise AssertionError(f"Unexpected seed prompt: {prompt}")
        if parse_fn is parse_comparative_evaluation_free_text:
            return comparative, "{}"
        raise AssertionError("Unexpected parse function")

    monkeypatch.setattr(refiner, "_call_api", fake_call_api)
    response, _ = refiner.generate("Case 558 text")

    assert response.final_diagnosis == "Primary myelofibrosis"
    metadata = refiner._get_case_variant_metadata()
    assert metadata["seed_candidates"][0] == "Polycythemia vera"
    assert "Primary myelofibrosis" in metadata["pooled_differential"]
    assert metadata["final_selection_source"] == "comparative_counter_hypothesis"


def test_differential_audit_variant_falls_back_on_counter_hypothesis_error(monkeypatch):
    refiner = DifferentialAuditRefiner(client=object(), config=RefinerConfig())

    ranked = RankedCandidateSet(
        candidates=[
            RankedCandidate(label="Dx A", confidence=0.7),
            RankedCandidate(label="Dx B", confidence=0.2),
            RankedCandidate(label="Dx C", confidence=0.1),
        ],
        raw_response="{}",
    )

    def fake_call_api(model, prompt, parse_fn):
        if parse_fn is parse_ranked_candidates:
            return ranked, "{}"
        if parse_fn is parse_counter_hypotheses:
            raise ValueError("counter-hypothesis parse failed")
        if parse_fn is parse_diagnostic_response:
            return DiagnosticResponse(
                final_diagnosis="Fallback diagnosis",
                next_steps=["step"],
            ), "{}"
        raise AssertionError("Unexpected parse function")

    monkeypatch.setattr(refiner, "_call_api", fake_call_api)
    response, _ = refiner.generate("Case text")

    assert response.final_diagnosis == "Fallback diagnosis"
    metadata = refiner._get_case_variant_metadata()
    assert metadata["final_selection_source"] == "baseline_fallback_counter_hypothesis_error"
    assert metadata["counter_hypothesis_error"]["seed"] == "Dx A"


def test_merge_differential_pool_deduplicates_and_caps():
    pooled = merge_differential_pool(
        seed_candidates=["Dx A", "Dx B", "Dx C"],
        counter_hypotheses_by_seed={
            "Dx A": ["Dx D", "Dx A", "Dx E"],
            "Dx B": ["Dx F", "Dx D", "Dx G"],
            "Dx C": ["Dx H", "Dx I", "Dx J"],
        },
        max_total=6,
    )

    assert pooled == ["Dx A", "Dx B", "Dx C", "Dx D", "Dx E", "Dx F"]


def test_discriminative_question_variant_fallback_on_question_error(monkeypatch):
    refiner = DiscriminativeQuestionRefiner(client=object(), config=RefinerConfig())

    ranked = RankedCandidateSet(
        candidates=[
            RankedCandidate(label="Dx A", confidence=0.7),
            RankedCandidate(label="Dx B", confidence=0.2),
            RankedCandidate(label="Dx C", confidence=0.1),
        ],
        raw_response="{}",
    )

    def fake_call_api(model, prompt, parse_fn):
        if parse_fn is parse_ranked_candidates:
            return ranked, "{}"
        if parse_fn is parse_discriminative_question:
            raise ValueError("question parse failed")
        if parse_fn is parse_diagnostic_response:
            return DiagnosticResponse(
                final_diagnosis="Fallback diagnosis",
                next_steps=["step"],
            ), "{}"
        raise AssertionError("Unexpected parse function")

    monkeypatch.setattr(refiner, "_call_api", fake_call_api)
    response, _ = refiner.generate("Case text")

    assert response.final_diagnosis == "Fallback diagnosis"
    metadata = refiner._get_case_variant_metadata()
    assert "question_error" in metadata
    assert metadata["final_selection_source"] == "baseline_fallback_question_error"


def test_progressive_disclosure_variant_happy_path(monkeypatch):
    refiner = ProgressiveDisclosureRefiner(client=object(), config=RefinerConfig())

    early = EarlyDifferential(
        candidates=[
            EarlyCandidate(label="Dx A", confidence=0.85, rationale="early support"),
            EarlyCandidate(label="Dx B", confidence=0.10, rationale="less likely"),
            EarlyCandidate(label="Dx C", confidence=0.05, rationale="least likely"),
        ],
        raw_response="{}",
    )
    revision = RevisionDecisionFreeText(
        response=DiagnosticResponse(
            final_diagnosis="Dx B",
            conditional_reasoning="full case revised diagnosis",
            next_steps=["step"],
        ),
        final_choice="Dx B",
        final_confidence=0.61,
        revision_summary="Dropped Dx A after contradictory lab",
        kept_hypotheses=[],
        dropped_hypotheses=["Dx A"],
        added_hypotheses=["Dx B"],
        contradiction_found=True,
        rationale="contradictory evidence in full case",
        raw_response="{}",
    )

    def fake_call_api(model, prompt, parse_fn):
        if parse_fn is parse_early_differential_free_text:
            return early, "{}"
        if parse_fn is parse_revision_decision_free_text:
            return revision, "{}"
        raise AssertionError("Unexpected parse function")

    monkeypatch.setattr(refiner, "_call_api", fake_call_api)
    response, _ = refiner.generate("Case text with enough tokens to split")

    assert response.final_diagnosis == "Dx B"
    metadata = refiner._get_case_variant_metadata()
    assert metadata["final_selection_source"] == "full_case_revision"
    assert "belief_penalty_score" in metadata


def test_progressive_disclosure_variant_applies_thresholds_to_penalty(monkeypatch):
    refiner = ProgressiveDisclosureRefiner(
        client=object(),
        config=RefinerConfig(
            early_confidence_threshold=0.8,
            revision_instability_threshold=0.5,
        ),
    )

    early = EarlyDifferential(
        candidates=[
            EarlyCandidate(label="Dx A", confidence=0.6, rationale="early support"),
            EarlyCandidate(label="Dx B", confidence=0.3, rationale="less likely"),
        ],
        raw_response="{}",
    )
    revision = RevisionDecisionFreeText(
        response=DiagnosticResponse(
            final_diagnosis="Dx B",
            conditional_reasoning="full case revised diagnosis",
            next_steps=["step"],
        ),
        final_choice="Dx B",
        final_confidence=0.61,
        revision_summary="changed",
        kept_hypotheses=[],
        dropped_hypotheses=["Dx A"],
        added_hypotheses=["Dx B"],
        contradiction_found=False,
        rationale="contradictory evidence in full case",
        raw_response="{}",
    )

    def fake_call_api(model, prompt, parse_fn):
        if parse_fn is parse_early_differential_free_text:
            return early, "{}"
        if parse_fn is parse_revision_decision_free_text:
            return revision, "{}"
        raise AssertionError("Unexpected parse function")

    monkeypatch.setattr(refiner, "_call_api", fake_call_api)
    monkeypatch.setattr(
        "refinement.variants.progressive_disclosure.compute_belief_revision_scores",
        lambda **kwargs: BeliefRevisionScores(
            anchoring_flag=False,
            confidence_instability_score=0.9,
            revision_delta=1.0,
            penalty_score=0.99,
        ),
    )

    refiner.generate("Case text with enough tokens to split")
    metadata = refiner._get_case_variant_metadata()

    assert metadata["confidence_instability_score"] == 0.0
    assert metadata["belief_penalty_score"] == 0.2


def test_progressive_disclosure_variant_fallback_on_revision_error(monkeypatch):
    refiner = ProgressiveDisclosureRefiner(client=object(), config=RefinerConfig())

    early = EarlyDifferential(
        candidates=[
            EarlyCandidate(label="Dx A", confidence=0.70, rationale="early support"),
            EarlyCandidate(label="Dx B", confidence=0.20, rationale="alt"),
        ],
        raw_response="{}",
    )

    def fake_call_api(model, prompt, parse_fn):
        if parse_fn is parse_early_differential_free_text:
            return early, "{}"
        if parse_fn is parse_revision_decision_free_text:
            raise ValueError("revision parse failed")
        if parse_fn is parse_diagnostic_response:
            return DiagnosticResponse(final_diagnosis="Fallback diagnosis", next_steps=["step"]), "{}"
        raise AssertionError("Unexpected parse function")

    monkeypatch.setattr(refiner, "_call_api", fake_call_api)
    response, _ = refiner.generate("Case text")

    assert response.final_diagnosis == "Fallback diagnosis"
    metadata = refiner._get_case_variant_metadata()
    assert "revision_stage_error" in metadata
    assert metadata["final_selection_source"] == "baseline_fallback_revision_error"
