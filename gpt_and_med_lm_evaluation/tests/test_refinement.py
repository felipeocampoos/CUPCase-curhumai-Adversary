"""
Unit tests for the refinement module.

Tests:
- JSON parsing/validation
- Stop condition logic
- CCR subset computation from YAML
- Minimality metrics correctness
"""

import json
import pytest
from pathlib import Path
from typing import Dict, Any

# Import modules to test
from refinement.schema import (
    DiagnosticResponse,
    CriticResult,
    RefinementTrace,
    ChecklistItemResult,
    ClinicalQuality,
    HardFail,
    ChecklistConfig,
    extract_json_from_response,
    parse_diagnostic_response,
    parse_critic_result,
    validate_diagnostic_response,
)
from refinement.metrics import (
    compute_edit_distance,
    compute_word_changes,
    compute_minimality_metrics,
    compute_ccr_for_case,
    compute_ccr_metrics,
)
from refinement.stats import (
    paired_bootstrap_ci,
    paired_permutation_test,
)


class TestJSONParsing:
    """Tests for JSON parsing and validation."""
    
    def test_extract_json_from_plain_text(self):
        """Test extracting JSON from plain text."""
        text = '{"final_diagnosis": "Acute appendicitis", "next_steps": ["Surgery"]}'
        result = extract_json_from_response(text)
        
        assert result["final_diagnosis"] == "Acute appendicitis"
        assert result["next_steps"] == ["Surgery"]
    
    def test_extract_json_from_markdown_code_block(self):
        """Test extracting JSON from markdown code block."""
        text = """
Here is my analysis:

```json
{
    "final_diagnosis": "Type 2 diabetes mellitus",
    "differential": ["Type 1 diabetes", "LADA"],
    "next_steps": ["Check HbA1c"]
}
```

That's my response.
"""
        result = extract_json_from_response(text)
        
        assert result["final_diagnosis"] == "Type 2 diabetes mellitus"
        assert len(result["differential"]) == 2
    
    def test_extract_json_from_code_block_without_lang(self):
        """Test extracting JSON from code block without language marker."""
        text = """
```
{"final_diagnosis": "Hypertension"}
```
"""
        result = extract_json_from_response(text)
        
        assert result["final_diagnosis"] == "Hypertension"
    
    def test_extract_json_fails_on_invalid(self):
        """Test that extraction fails on invalid JSON."""
        text = "This is not valid JSON at all"
        
        with pytest.raises(ValueError):
            extract_json_from_response(text)
    
    def test_parse_diagnostic_response(self):
        """Test parsing a full diagnostic response."""
        text = """
```json
{
    "final_diagnosis": "Acute myocardial infarction",
    "differential": ["Unstable angina", "Pulmonary embolism"],
    "clarifying_questions": ["Duration of chest pain?"],
    "red_flags": ["Chest pain", "Dyspnea"],
    "uncertainty": "Moderate confidence pending troponin",
    "next_steps": ["ECG", "Troponin", "Aspirin"]
}
```
"""
        response = parse_diagnostic_response(text)
        
        assert isinstance(response, DiagnosticResponse)
        assert response.final_diagnosis == "Acute myocardial infarction"
        assert len(response.differential) == 2
        assert len(response.next_steps) == 3

    def test_parse_diagnostic_response_falls_back_from_leaked_template_heading(self):
        text = """
Construct Final Response:
Acute appendicitis
"""
        response = parse_diagnostic_response(text)

        assert response.final_diagnosis == "Acute appendicitis"

    def test_parse_diagnostic_response_sanitizes_leaked_heading_inside_json_field(self):
        text = '{"final_diagnosis": "Draft Output:\\nAcute pancreatitis", "next_steps": []}'
        response = parse_diagnostic_response(text)

        assert response.final_diagnosis == "Acute pancreatitis"
    
    def test_parse_critic_result(self):
        """Test parsing a critic result."""
        text = """
{
    "checklist": [
        {"item_id": "C1", "pass": true, "rationale": "Clear diagnosis", "suggested_fix": null},
        {"item_id": "C2", "pass": false, "rationale": "No differential", "suggested_fix": "Add differential"}
    ],
    "clinical_quality": {"score": 4, "rationale": "Good overall"},
    "hard_fail": {"failed": false, "reason": null},
    "edit_plan": ["Add differential diagnoses"]
}
"""
        result = parse_critic_result(text)
        
        assert isinstance(result, CriticResult)
        assert len(result.checklist) == 2
        assert result.checklist[0].passed == True
        assert result.checklist[1].passed == False
        assert result.clinical_quality.score == 4
        assert result.hard_fail.failed == False


class TestValidation:
    """Tests for response validation."""
    
    def test_valid_response_passes(self):
        """Test that a valid response has no errors."""
        response = DiagnosticResponse(
            final_diagnosis="Acute appendicitis",
            differential=["Cholecystitis"],
            next_steps=["CT scan", "Surgery consult"],
        )
        
        errors = validate_diagnostic_response(response)
        assert len(errors) == 0
    
    def test_missing_diagnosis_fails(self):
        """Test that missing diagnosis is caught."""
        response = DiagnosticResponse(
            final_diagnosis="",
            next_steps=["Test"],
        )
        
        errors = validate_diagnostic_response(response)
        assert any("final_diagnosis" in e for e in errors)
    
    def test_short_diagnosis_warns(self):
        """Test that very short diagnosis is caught."""
        response = DiagnosticResponse(
            final_diagnosis="A",
            next_steps=["Test"],
        )
        
        errors = validate_diagnostic_response(response)
        assert any("too short" in e for e in errors)


class TestStopCondition:
    """Tests for compliance stop condition logic."""
    
    def create_critic_result(
        self,
        all_pass: bool = True,
        quality_score: int = 4,
        hard_fail: bool = False,
    ) -> CriticResult:
        """Helper to create test CriticResult."""
        checklist = [
            ChecklistItemResult(
                item_id=f"C{i}",
                passed=all_pass,
                rationale="Test",
            )
            for i in range(1, 9)
        ]
        
        return CriticResult(
            checklist=checklist,
            clinical_quality=ClinicalQuality(score=quality_score, rationale="Test"),
            hard_fail=HardFail(failed=hard_fail, reason="Test" if hard_fail else None),
            edit_plan=[] if all_pass else ["Fix something"],
        )
    
    def test_all_pass_is_compliant(self):
        """Test that all passing is compliant."""
        result = self.create_critic_result(all_pass=True, quality_score=4, hard_fail=False)
        
        assert result.is_compliant(quality_threshold=3) == True
    
    def test_failed_item_not_compliant(self):
        """Test that any failed item means not compliant."""
        result = self.create_critic_result(all_pass=False, quality_score=4, hard_fail=False)
        
        assert result.is_compliant(quality_threshold=3) == False
    
    def test_low_quality_not_compliant(self):
        """Test that low quality score means not compliant."""
        result = self.create_critic_result(all_pass=True, quality_score=2, hard_fail=False)
        
        assert result.is_compliant(quality_threshold=3) == False
    
    def test_hard_fail_not_compliant(self):
        """Test that hard fail means not compliant."""
        result = self.create_critic_result(all_pass=True, quality_score=4, hard_fail=True)
        
        assert result.is_compliant(quality_threshold=3) == False
    
    def test_threshold_boundary(self):
        """Test quality threshold boundary."""
        result = self.create_critic_result(all_pass=True, quality_score=3, hard_fail=False)
        
        # Exactly at threshold should pass
        assert result.is_compliant(quality_threshold=3) == True
        # Below threshold should fail
        assert result.is_compliant(quality_threshold=4) == False


class TestCCRComputation:
    """Tests for CCR metric computation."""
    
    @pytest.fixture
    def mock_config(self) -> ChecklistConfig:
        """Create a mock checklist config for testing."""
        from refinement.schema import ChecklistItem, CCRGroup
        
        items = [
            ChecklistItem(
                id=f"C{i}",
                name=f"Item {i}",
                description=f"Description {i}",
                when_required="always",
                json_field=None,
                examples_good=[],
                examples_bad=[],
            )
            for i in range(1, 9)
        ]
        
        ccr_groups = {
            "CCR_all": CCRGroup(
                name="CCR_all",
                description="All items",
                items=["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
            ),
            "CCR_Q": CCRGroup(
                name="CCR_Q",
                description="Quality subset",
                items=["C3", "C4", "C5"],
            ),
            "CCR_H": CCRGroup(
                name="CCR_H",
                description="Health subset",
                items=["C6", "C7", "C8"],
            ),
        }
        
        return ChecklistConfig(
            version="1.0",
            items=items,
            ccr_groups=ccr_groups,
            clinical_quality_threshold=3,
            hard_fail_conditions=[],
        )
    
    def test_all_pass_ccr_all(self, mock_config):
        """Test CCR_all when all items pass."""
        pass_map = {f"C{i}": True for i in range(1, 9)}
        
        result = compute_ccr_for_case(pass_map, mock_config)
        
        assert result["CCR_all"] == True
        assert result["CCR_Q"] == True
        assert result["CCR_H"] == True
    
    def test_partial_fail_ccr_q(self, mock_config):
        """Test CCR_Q when only Q items fail."""
        pass_map = {f"C{i}": True for i in range(1, 9)}
        pass_map["C3"] = False  # C3 is in CCR_Q
        
        result = compute_ccr_for_case(pass_map, mock_config)
        
        assert result["CCR_all"] == False
        assert result["CCR_Q"] == False
        assert result["CCR_H"] == True
    
    def test_partial_fail_ccr_h(self, mock_config):
        """Test CCR_H when only H items fail."""
        pass_map = {f"C{i}": True for i in range(1, 9)}
        pass_map["C6"] = False  # C6 is in CCR_H
        
        result = compute_ccr_for_case(pass_map, mock_config)
        
        assert result["CCR_all"] == False
        assert result["CCR_Q"] == True
        assert result["CCR_H"] == False


class TestMinimalityMetrics:
    """Tests for minimality/edit distance metrics."""
    
    def test_edit_distance_identical(self):
        """Test edit distance of identical strings."""
        dist, ratio = compute_edit_distance("hello world", "hello world")
        
        assert dist == 0
        assert ratio == 0.0
    
    def test_edit_distance_different(self):
        """Test edit distance of different strings."""
        dist, ratio = compute_edit_distance("hello", "hallo")
        
        assert dist > 0
        assert 0 < ratio < 1
    
    def test_edit_distance_empty(self):
        """Test edit distance with empty strings."""
        dist, ratio = compute_edit_distance("", "")
        
        assert dist == 0
        assert ratio == 0.0
    
    def test_word_changes_identical(self):
        """Test word changes for identical text."""
        changes = compute_word_changes("hello world", "hello world")
        
        assert changes == 0
    
    def test_word_changes_different(self):
        """Test word changes for different text."""
        changes = compute_word_changes("hello world", "hello there world")
        
        assert changes >= 1
    
    def test_minimality_metrics_single_response(self):
        """Test minimality metrics with single response."""
        responses = [
            DiagnosticResponse(final_diagnosis="Test diagnosis", next_steps=["Test"]),
        ]
        
        metrics = compute_minimality_metrics(responses)
        
        assert metrics.edit_distance_total == 0
        assert metrics.edit_ratio_total == 0.0
    
    def test_minimality_metrics_multiple_responses(self):
        """Test minimality metrics with multiple responses."""
        responses = [
            DiagnosticResponse(final_diagnosis="Initial diagnosis", next_steps=["Test"]),
            DiagnosticResponse(final_diagnosis="Revised diagnosis", next_steps=["Test", "More tests"]),
            DiagnosticResponse(final_diagnosis="Final diagnosis", next_steps=["Test", "More tests", "Treatment"]),
        ]
        
        metrics = compute_minimality_metrics(responses)
        
        assert metrics.edit_distance_total > 0
        assert metrics.edit_distance_last > 0
        assert 0 < metrics.edit_ratio_total < 1


class TestStatistics:
    """Tests for statistical comparison functions."""
    
    def test_bootstrap_identical_data(self):
        """Test bootstrap with identical data."""
        baseline = [0.5, 0.6, 0.7, 0.8, 0.9]
        refined = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        result = paired_bootstrap_ci(baseline, refined)
        
        assert abs(result.mean_difference) < 0.01
        assert result.ci_lower <= 0 <= result.ci_upper
    
    def test_bootstrap_different_data(self):
        """Test bootstrap with clearly different data."""
        baseline = [0.5, 0.5, 0.5, 0.5, 0.5]
        refined = [0.9, 0.9, 0.9, 0.9, 0.9]
        
        result = paired_bootstrap_ci(baseline, refined)
        
        assert result.mean_difference > 0.3
        assert result.ci_lower > 0  # Significant improvement
    
    def test_bootstrap_empty_raises(self):
        """Test bootstrap raises on empty data."""
        with pytest.raises(ValueError):
            paired_bootstrap_ci([], [])
    
    def test_bootstrap_unequal_length_raises(self):
        """Test bootstrap raises on unequal length."""
        with pytest.raises(ValueError):
            paired_bootstrap_ci([0.5, 0.6], [0.5])
    
    def test_permutation_identical_data(self):
        """Test permutation with identical data."""
        baseline = [0.5, 0.6, 0.7, 0.8, 0.9]
        refined = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        result = paired_permutation_test(baseline, refined)
        
        assert result.p_value > 0.5  # Not significant
    
    def test_permutation_different_data(self):
        """Test permutation with clearly different data."""
        baseline = [0.5] * 20
        refined = [0.9] * 20
        
        result = paired_permutation_test(baseline, refined)
        
        assert result.p_value < 0.05  # Significant


class TestChecklistConfig:
    """Tests for checklist configuration loading."""
    
    def test_load_default_config(self):
        """Test loading the default checklist config."""
        config = ChecklistConfig.load()
        
        assert config.version == "1.1"
        assert len(config.items) == 8
        assert "CCR_all" in config.ccr_groups
        assert "CCR_Q" in config.ccr_groups
        assert "CCR_H" in config.ccr_groups
        assert "CCR_CurHum" in config.ccr_groups
    
    def test_ccr_groups_have_items(self):
        """Test that CCR groups have the expected items."""
        config = ChecklistConfig.load()
        
        assert len(config.get_ccr_group_items("CCR_all")) == 8
        assert len(config.get_ccr_group_items("CCR_Q")) > 0
        assert len(config.get_ccr_group_items("CCR_H")) > 0
    
    def test_get_item_by_id(self):
        """Test getting item by ID."""
        config = ChecklistConfig.load()
        
        item = config.get_item_by_id("C1")
        
        assert item is not None
        assert item.id == "C1"
        assert item.name == "Primary Diagnosis"


class TestCuriosityHumilityScores:
    """Tests for curiosity and humility score support."""

    def test_critic_result_backward_compat_missing_scores(self):
        """CriticResult.from_dict handles missing curiosity/humility scores."""
        data = {
            "checklist": [{"item_id": "C1", "pass": True, "rationale": "ok"}],
            "clinical_quality": {"score": 4, "rationale": "good"},
            "hard_fail": {"failed": False},
            "edit_plan": [],
        }
        result = CriticResult.from_dict(data)
        assert result.curiosity_score is None
        assert result.humility_score is None

    def test_critic_result_parses_scores(self):
        """CriticResult.from_dict parses curiosity/humility scores."""
        data = {
            "checklist": [{"item_id": "C1", "pass": True, "rationale": "ok"}],
            "clinical_quality": {"score": 4, "rationale": "good"},
            "hard_fail": {"failed": False},
            "edit_plan": [],
            "curiosity_score": 3,
            "humility_score": 5,
        }
        result = CriticResult.from_dict(data)
        assert result.curiosity_score == 3
        assert result.humility_score == 5

    def test_critic_result_to_dict_includes_scores(self):
        """CriticResult.to_dict includes curiosity/humility scores."""
        result = CriticResult(
            checklist=[],
            clinical_quality=ClinicalQuality(score=4, rationale="good"),
            hard_fail=HardFail(failed=False),
            edit_plan=[],
            curiosity_score=2,
            humility_score=4,
        )
        d = result.to_dict()
        assert d["curiosity_score"] == 2
        assert d["humility_score"] == 4

    def test_compliance_not_affected_by_scores(self):
        """is_compliant() is NOT affected by curiosity/humility scores."""
        items = [
            ChecklistItemResult(item_id=f"C{i}", passed=True, rationale="ok")
            for i in range(1, 9)
        ]
        result = CriticResult(
            checklist=items,
            clinical_quality=ClinicalQuality(score=4, rationale="good"),
            hard_fail=HardFail(failed=False),
            edit_plan=[],
            curiosity_score=0,
            humility_score=0,
        )
        assert result.is_compliant(quality_threshold=3) is True

    def test_ccr_curhum_group(self):
        """CCR_CurHum group includes C3, C4, C6."""
        config = ChecklistConfig.load()
        curhum_items = config.get_ccr_group_items("CCR_CurHum")
        assert set(curhum_items) == {"C3", "C4", "C6"}

    def test_ccr_curhum_computation(self):
        """compute_ccr_for_case computes CCR_CurHum correctly."""
        config = ChecklistConfig.load()
        pass_map_all = {f"C{i}": True for i in range(1, 9)}
        result = compute_ccr_for_case(pass_map_all, config)
        assert result["CCR_CurHum"] is True

        pass_map_fail_c3 = {f"C{i}": True for i in range(1, 9)}
        pass_map_fail_c3["C3"] = False
        result2 = compute_ccr_for_case(pass_map_fail_c3, config)
        assert result2["CCR_CurHum"] is False

    def test_refinement_trace_round_trip(self):
        """RefinementTrace serializes/deserializes curiosity/humility scores."""
        from refinement.schema import DiagnosticResponse, RefinementTrace

        trace = RefinementTrace(
            case_id="test",
            case_text="test case",
            true_diagnosis="test dx",
            final_response=DiagnosticResponse(final_diagnosis="dx", next_steps=[]),
            extracted_final_diagnosis="dx",
            iterations_to_compliance=1,
            is_compliant=True,
            iterations=[],
            curiosity_score=3,
            humility_score=4,
        )
        d = trace.to_dict()
        assert d["curiosity_score"] == 3
        assert d["humility_score"] == 4

        restored = RefinementTrace.from_dict(d)
        assert restored.curiosity_score == 3
        assert restored.humility_score == 4

    def test_refinement_trace_backward_compat(self):
        """Old traces without curiosity/humility scores deserialize correctly."""
        from refinement.schema import RefinementTrace

        data = {
            "case_id": "old",
            "case_text": "old case",
            "true_diagnosis": "dx",
            "final_response": {"final_diagnosis": "dx"},
            "extracted_final_diagnosis": "dx",
            "is_compliant": True,
            "iterations": [],
        }
        trace = RefinementTrace.from_dict(data)
        assert trace.curiosity_score is None
        assert trace.humility_score is None


class TestCuriosityHumilityGating:
    """Tests for curiosity/humility score gating in is_jointly_compliant."""

    @staticmethod
    def _make_critic(curiosity_score=None, humility_score=None):
        items = [
            ChecklistItemResult(item_id=f"C{i}", passed=True, rationale="ok")
            for i in range(1, 9)
        ]
        return CriticResult(
            checklist=items,
            clinical_quality=ClinicalQuality(score=4, rationale="good"),
            hard_fail=HardFail(failed=False),
            edit_plan=[],
            curiosity_score=curiosity_score,
            humility_score=humility_score,
        )

    @staticmethod
    def _make_refiner(curiosity_threshold=0, humility_threshold=0):
        from refinement.refiner import RefinerConfig, IterativeRefiner
        from unittest.mock import MagicMock

        config = RefinerConfig(
            curiosity_threshold=curiosity_threshold,
            humility_threshold=humility_threshold,
        )
        refiner = object.__new__(IterativeRefiner)
        refiner.config = config
        return refiner

    def test_default_thresholds_ignore_scores(self):
        """With thresholds=0, scores don't affect compliance."""
        refiner = self._make_refiner()
        critic = self._make_critic(curiosity_score=0, humility_score=0)
        assert refiner.is_jointly_compliant(critic) is True

    def test_curiosity_below_threshold(self):
        """curiosity_score < threshold → not compliant."""
        refiner = self._make_refiner(curiosity_threshold=3)
        critic = self._make_critic(curiosity_score=2, humility_score=4)
        assert refiner.is_jointly_compliant(critic) is False

    def test_humility_below_threshold(self):
        """humility_score < threshold → not compliant."""
        refiner = self._make_refiner(humility_threshold=3)
        critic = self._make_critic(curiosity_score=4, humility_score=2)
        assert refiner.is_jointly_compliant(critic) is False

    def test_both_above_threshold(self):
        """Both scores >= threshold → compliant."""
        refiner = self._make_refiner(curiosity_threshold=3, humility_threshold=3)
        critic = self._make_critic(curiosity_score=4, humility_score=4)
        assert refiner.is_jointly_compliant(critic) is True

    def test_none_scores_with_threshold_fails(self):
        """None score with threshold > 0 → not compliant."""
        refiner = self._make_refiner(curiosity_threshold=3)
        critic = self._make_critic(curiosity_score=None, humility_score=4)
        assert refiner.is_jointly_compliant(critic) is False

    def test_none_scores_with_zero_threshold_passes(self):
        """None score with threshold=0 → compliant (check skipped)."""
        refiner = self._make_refiner(curiosity_threshold=0)
        critic = self._make_critic(curiosity_score=None, humility_score=None)
        assert refiner.is_jointly_compliant(critic) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
