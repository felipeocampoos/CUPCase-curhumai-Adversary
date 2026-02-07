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


def test_create_refiner_variant_domain_routed():
    refiner = create_refiner_variant(
        variant="domain_routed",
        api_key="test-key",
        config=RefinerConfig(),
    )
    assert isinstance(refiner, DomainRoutedRefiner)


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
