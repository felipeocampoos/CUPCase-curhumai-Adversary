"""
Schema definitions and validation helpers for structured I/O.

Provides dataclasses and parsing utilities for:
- DiagnosticResponse (Generator/Editor output)
- CriticResult (Critic output)
- RefinementTrace (full iteration log)
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path

import yaml


@dataclass
class DiagnosticResponse:
    """Structured diagnostic response from Generator/Editor."""
    
    final_diagnosis: str
    differential: Optional[List[str]] = None
    conditional_reasoning: Optional[str] = None
    clarifying_questions: Optional[List[str]] = None
    red_flags: Optional[List[str]] = None
    uncertainty: Optional[str] = None
    next_steps: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values for cleaner output."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagnosticResponse":
        """Create from dictionary with validation."""
        # Handle missing or None values gracefully
        return cls(
            final_diagnosis=data.get("final_diagnosis", ""),
            differential=data.get("differential"),
            conditional_reasoning=data.get("conditional_reasoning"),
            clarifying_questions=data.get("clarifying_questions"),
            red_flags=data.get("red_flags"),
            uncertainty=data.get("uncertainty"),
            next_steps=data.get("next_steps"),
        )


@dataclass
class ChecklistItemResult:
    """Result of evaluating a single checklist item."""
    
    item_id: str
    passed: bool
    rationale: str
    suggested_fix: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChecklistItemResult":
        return cls(
            item_id=data.get("item_id", ""),
            passed=data.get("pass", False),
            rationale=data.get("rationale", ""),
            suggested_fix=data.get("suggested_fix"),
        )


@dataclass
class ClinicalQuality:
    """Clinical quality assessment."""
    
    score: int
    rationale: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClinicalQuality":
        return cls(
            score=data.get("score", 0),
            rationale=data.get("rationale", ""),
        )


@dataclass
class HardFail:
    """Hard failure assessment."""
    
    failed: bool
    reason: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HardFail":
        return cls(
            failed=data.get("failed", False),
            reason=data.get("reason"),
        )


@dataclass
class CriticResult:
    """Full critic evaluation result."""
    
    checklist: List[ChecklistItemResult]
    clinical_quality: ClinicalQuality
    hard_fail: HardFail
    edit_plan: List[str]
    
    def is_compliant(self, quality_threshold: int = 3) -> bool:
        """Check if response meets joint compliance criteria."""
        all_passed = all(item.passed for item in self.checklist)
        quality_ok = self.clinical_quality.score >= quality_threshold
        no_hard_fail = not self.hard_fail.failed
        return all_passed and quality_ok and no_hard_fail
    
    def get_failed_items(self) -> List[ChecklistItemResult]:
        """Get list of failed checklist items."""
        return [item for item in self.checklist if not item.passed]
    
    def get_passed_items(self) -> List[ChecklistItemResult]:
        """Get list of passed checklist items."""
        return [item for item in self.checklist if item.passed]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checklist": [
                {
                    "item_id": item.item_id,
                    "pass": item.passed,
                    "rationale": item.rationale,
                    "suggested_fix": item.suggested_fix,
                }
                for item in self.checklist
            ],
            "clinical_quality": {
                "score": self.clinical_quality.score,
                "rationale": self.clinical_quality.rationale,
            },
            "hard_fail": {
                "failed": self.hard_fail.failed,
                "reason": self.hard_fail.reason,
            },
            "edit_plan": self.edit_plan,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CriticResult":
        """Create from dictionary with validation."""
        checklist = [
            ChecklistItemResult.from_dict(item)
            for item in data.get("checklist", [])
        ]
        clinical_quality = ClinicalQuality.from_dict(
            data.get("clinical_quality", {})
        )
        hard_fail = HardFail.from_dict(data.get("hard_fail", {}))
        edit_plan = data.get("edit_plan", [])
        
        return cls(
            checklist=checklist,
            clinical_quality=clinical_quality,
            hard_fail=hard_fail,
            edit_plan=edit_plan,
        )


@dataclass
class IterationLog:
    """Log entry for a single iteration."""
    
    iteration: int
    response: DiagnosticResponse
    critic_result: Optional[CriticResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "response": self.response.to_dict(),
            "critic_result": self.critic_result.to_dict() if self.critic_result else None,
        }


@dataclass
class RefinementTrace:
    """Complete trace of a refinement process for a single case."""
    
    case_id: str
    case_text: str
    true_diagnosis: str
    final_response: DiagnosticResponse
    extracted_final_diagnosis: str
    iterations_to_compliance: Optional[int]
    is_compliant: bool
    iterations: List[IterationLog]
    minimality_metrics: Dict[str, float] = field(default_factory=dict)
    checklist_pass_map: Dict[str, bool] = field(default_factory=dict)
    clinical_quality_score: Optional[int] = None
    hard_fail: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "case_id": self.case_id,
            "case_text": self.case_text,
            "true_diagnosis": self.true_diagnosis,
            "final_response": self.final_response.to_dict(),
            "extracted_final_diagnosis": self.extracted_final_diagnosis,
            "iterations_to_compliance": self.iterations_to_compliance,
            "is_compliant": self.is_compliant,
            "iterations": [it.to_dict() for it in self.iterations],
            "minimality_metrics": self.minimality_metrics,
            "checklist_pass_map": self.checklist_pass_map,
            "clinical_quality_score": self.clinical_quality_score,
            "hard_fail": self.hard_fail,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RefinementTrace":
        """Create from dictionary."""
        iterations = []
        for it_data in data.get("iterations", []):
            response = DiagnosticResponse.from_dict(it_data.get("response", {}))
            critic_data = it_data.get("critic_result")
            critic_result = CriticResult.from_dict(critic_data) if critic_data else None
            iterations.append(IterationLog(
                iteration=it_data.get("iteration", 0),
                response=response,
                critic_result=critic_result,
            ))
        
        return cls(
            case_id=data.get("case_id", ""),
            case_text=data.get("case_text", ""),
            true_diagnosis=data.get("true_diagnosis", ""),
            final_response=DiagnosticResponse.from_dict(data.get("final_response", {})),
            extracted_final_diagnosis=data.get("extracted_final_diagnosis", ""),
            iterations_to_compliance=data.get("iterations_to_compliance"),
            is_compliant=data.get("is_compliant", False),
            iterations=iterations,
            minimality_metrics=data.get("minimality_metrics", {}),
            checklist_pass_map=data.get("checklist_pass_map", {}),
            clinical_quality_score=data.get("clinical_quality_score"),
            hard_fail=data.get("hard_fail", False),
        )


def extract_json_from_response(text: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM response, handling markdown code blocks.
    
    Args:
        text: Raw text response from LLM
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        ValueError: If no valid JSON found
    """
    # Try to find JSON in markdown code block
    code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    
    if matches:
        # Try each code block
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # Try to find raw JSON (object or array)
    json_pattern = r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})"
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        # Try longest match first (likely the full response)
        for match in sorted(matches, key=len, reverse=True):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Last resort: try parsing the entire text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    raise ValueError(f"Could not extract valid JSON from response: {text[:200]}...")


def parse_diagnostic_response(text: str) -> DiagnosticResponse:
    """
    Parse LLM output into DiagnosticResponse.
    
    Args:
        text: Raw text response from Generator/Editor
        
    Returns:
        Parsed DiagnosticResponse
        
    Raises:
        ValueError: If parsing fails
    """
    data = extract_json_from_response(text)
    return DiagnosticResponse.from_dict(data)


def parse_critic_result(text: str) -> CriticResult:
    """
    Parse LLM output into CriticResult.
    
    Args:
        text: Raw text response from Critic
        
    Returns:
        Parsed CriticResult
        
    Raises:
        ValueError: If parsing fails
    """
    data = extract_json_from_response(text)
    return CriticResult.from_dict(data)


@dataclass
class ChecklistItem:
    """Definition of a single checklist item from YAML config."""
    
    id: str
    name: str
    description: str
    when_required: str
    json_field: Optional[str]
    examples_good: List[str]
    examples_bad: List[str]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChecklistItem":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            when_required=data.get("when_required", "always"),
            json_field=data.get("json_field"),
            examples_good=data.get("examples_good", []),
            examples_bad=data.get("examples_bad", []),
        )


@dataclass
class CCRGroup:
    """Definition of a CCR metric group."""
    
    name: str
    description: str
    items: List[str]
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "CCRGroup":
        return cls(
            name=name,
            description=data.get("description", ""),
            items=data.get("items", []),
        )


@dataclass
class ChecklistConfig:
    """Full checklist configuration loaded from YAML."""
    
    version: str
    items: List[ChecklistItem]
    ccr_groups: Dict[str, CCRGroup]
    clinical_quality_threshold: int
    hard_fail_conditions: List[str]
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ChecklistConfig":
        """Load checklist configuration from YAML file."""
        if path is None:
            # Default path relative to this module
            path = Path(__file__).parent / "checklist.yaml"
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        items = [
            ChecklistItem.from_dict(item)
            for item in data.get("checklist_items", [])
        ]
        
        ccr_groups = {
            name: CCRGroup.from_dict(name, group_data)
            for name, group_data in data.get("ccr_groups", {}).items()
        }
        
        clinical_config = data.get("clinical_quality", {})
        threshold = clinical_config.get("threshold", 3)
        
        hard_fail_conditions = data.get("hard_fail_conditions", [])
        
        return cls(
            version=data.get("version", "1.0"),
            items=items,
            ccr_groups=ccr_groups,
            clinical_quality_threshold=threshold,
            hard_fail_conditions=hard_fail_conditions,
        )
    
    def get_item_by_id(self, item_id: str) -> Optional[ChecklistItem]:
        """Get checklist item by ID."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None
    
    def get_ccr_group_items(self, group_name: str) -> List[str]:
        """Get item IDs for a CCR group."""
        if group_name in self.ccr_groups:
            return self.ccr_groups[group_name].items
        return []


def validate_diagnostic_response(response: DiagnosticResponse) -> List[str]:
    """
    Validate a diagnostic response for structural correctness.
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if not response.final_diagnosis:
        errors.append("Missing required field: final_diagnosis")
    
    if response.final_diagnosis and len(response.final_diagnosis) < 3:
        errors.append("final_diagnosis is too short (less than 3 characters)")
    
    if response.differential is not None and not isinstance(response.differential, list):
        errors.append("differential must be a list")
    
    if response.clarifying_questions is not None and not isinstance(response.clarifying_questions, list):
        errors.append("clarifying_questions must be a list")
    
    if response.red_flags is not None and not isinstance(response.red_flags, list):
        errors.append("red_flags must be a list")
    
    if response.next_steps is not None and not isinstance(response.next_steps, list):
        errors.append("next_steps must be a list")
    
    return errors
