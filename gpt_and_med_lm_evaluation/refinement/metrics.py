"""
Metrics computation for iterative refinement evaluation.

Computes:
- Edit distance / minimality metrics between iterations
- CCR (Checklist Compliance Rate) metrics: CCR_all, CCR_Q, CCR_H
- Iterations to compliance
"""

import difflib
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .schema import (
    DiagnosticResponse,
    CriticResult,
    RefinementTrace,
    ChecklistConfig,
)


@dataclass
class MinimalityMetrics:
    """Metrics measuring minimality of edits across iterations."""
    
    edit_distance_total: int
    edit_distance_last: int
    edit_ratio_total: float
    word_changes_total: int
    word_changes_last: int
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "edit_distance_total": self.edit_distance_total,
            "edit_distance_last": self.edit_distance_last,
            "edit_ratio_total": self.edit_ratio_total,
            "word_changes_total": self.word_changes_total,
            "word_changes_last": self.word_changes_last,
        }


@dataclass 
class CCRMetrics:
    """Checklist Compliance Rate metrics."""
    
    ccr_all: float
    ccr_q: float
    ccr_h: float
    per_item_rates: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "CCR_all": self.ccr_all,
            "CCR_Q": self.ccr_q,
            "CCR_H": self.ccr_h,
            "per_item_rates": self.per_item_rates,
        }


def compute_edit_distance(text1: str, text2: str) -> Tuple[int, float]:
    """
    Compute character-level edit distance and ratio between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Tuple of (edit_distance, edit_ratio)
        edit_ratio is normalized by the length of the longer string
    """
    if not text1 and not text2:
        return 0, 0.0
    
    # Use SequenceMatcher for efficient similarity computation
    matcher = difflib.SequenceMatcher(None, text1, text2)
    
    # Calculate edit operations
    opcodes = matcher.get_opcodes()
    edit_distance = sum(
        max(j2 - j1, i2 - i1)
        for tag, i1, i2, j1, j2 in opcodes
        if tag != 'equal'
    )
    
    # Normalize by length of longer string
    max_len = max(len(text1), len(text2))
    edit_ratio = edit_distance / max_len if max_len > 0 else 0.0
    
    return edit_distance, edit_ratio


def compute_word_changes(text1: str, text2: str) -> int:
    """
    Compute number of word-level changes between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Number of word changes (additions, deletions, modifications)
    """
    words1 = text1.split()
    words2 = text2.split()
    
    matcher = difflib.SequenceMatcher(None, words1, words2)
    opcodes = matcher.get_opcodes()
    
    changes = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != 'equal':
            changes += max(i2 - i1, j2 - j1)
    
    return changes


def response_to_comparable_string(response: DiagnosticResponse) -> str:
    """Convert DiagnosticResponse to a string for comparison."""
    return json.dumps(response.to_dict(), sort_keys=True, indent=2)


def compute_minimality_metrics(
    responses: List[DiagnosticResponse],
) -> MinimalityMetrics:
    """
    Compute minimality metrics across a sequence of responses.
    
    Args:
        responses: List of DiagnosticResponse objects from each iteration
        
    Returns:
        MinimalityMetrics object
    """
    if len(responses) < 2:
        return MinimalityMetrics(
            edit_distance_total=0,
            edit_distance_last=0,
            edit_ratio_total=0.0,
            word_changes_total=0,
            word_changes_last=0,
        )
    
    total_edit_distance = 0
    total_word_changes = 0
    total_chars = 0
    last_edit_distance = 0
    last_word_changes = 0
    
    for i in range(1, len(responses)):
        prev_str = response_to_comparable_string(responses[i - 1])
        curr_str = response_to_comparable_string(responses[i])
        
        edit_dist, _ = compute_edit_distance(prev_str, curr_str)
        word_changes = compute_word_changes(prev_str, curr_str)
        
        total_edit_distance += edit_dist
        total_word_changes += word_changes
        total_chars += max(len(prev_str), len(curr_str))
        
        # Track last iteration metrics
        if i == len(responses) - 1:
            last_edit_distance = edit_dist
            last_word_changes = word_changes
    
    edit_ratio_total = total_edit_distance / total_chars if total_chars > 0 else 0.0
    
    return MinimalityMetrics(
        edit_distance_total=total_edit_distance,
        edit_distance_last=last_edit_distance,
        edit_ratio_total=edit_ratio_total,
        word_changes_total=total_word_changes,
        word_changes_last=last_word_changes,
    )


def compute_ccr_for_case(
    checklist_pass_map: Dict[str, bool],
    config: ChecklistConfig,
) -> Dict[str, bool]:
    """
    Compute CCR compliance for a single case.
    
    Args:
        checklist_pass_map: Dict mapping item_id to pass/fail
        config: Checklist configuration
        
    Returns:
        Dict with CCR_all, CCR_Q, CCR_H as booleans
    """
    # CCR_all: all items must pass
    ccr_all_items = config.get_ccr_group_items("CCR_all")
    ccr_all = all(
        checklist_pass_map.get(item_id, False)
        for item_id in ccr_all_items
    )
    
    # CCR_Q: quality subset
    ccr_q_items = config.get_ccr_group_items("CCR_Q")
    ccr_q = all(
        checklist_pass_map.get(item_id, False)
        for item_id in ccr_q_items
    ) if ccr_q_items else True
    
    # CCR_H: safety/health subset
    ccr_h_items = config.get_ccr_group_items("CCR_H")
    ccr_h = all(
        checklist_pass_map.get(item_id, False)
        for item_id in ccr_h_items
    ) if ccr_h_items else True
    
    return {
        "CCR_all": ccr_all,
        "CCR_Q": ccr_q,
        "CCR_H": ccr_h,
    }


def compute_ccr_metrics(
    traces: List[RefinementTrace],
    config: Optional[ChecklistConfig] = None,
) -> CCRMetrics:
    """
    Compute CCR metrics across all cases.
    
    Args:
        traces: List of RefinementTrace objects
        config: Optional checklist configuration (loads default if None)
        
    Returns:
        CCRMetrics object with rates across all cases
    """
    if config is None:
        config = ChecklistConfig.load()
    
    if not traces:
        return CCRMetrics(
            ccr_all=0.0,
            ccr_q=0.0,
            ccr_h=0.0,
            per_item_rates={},
        )
    
    # Count compliance per CCR group
    ccr_all_count = 0
    ccr_q_count = 0
    ccr_h_count = 0
    
    # Track per-item pass rates
    item_pass_counts: Dict[str, int] = {}
    
    for trace in traces:
        # Compute CCR for this case
        case_ccr = compute_ccr_for_case(trace.checklist_pass_map, config)
        
        if case_ccr["CCR_all"]:
            ccr_all_count += 1
        if case_ccr["CCR_Q"]:
            ccr_q_count += 1
        if case_ccr["CCR_H"]:
            ccr_h_count += 1
        
        # Track per-item rates
        for item_id, passed in trace.checklist_pass_map.items():
            if item_id not in item_pass_counts:
                item_pass_counts[item_id] = 0
            if passed:
                item_pass_counts[item_id] += 1
    
    n_cases = len(traces)
    
    per_item_rates = {
        item_id: count / n_cases
        for item_id, count in item_pass_counts.items()
    }
    
    return CCRMetrics(
        ccr_all=ccr_all_count / n_cases,
        ccr_q=ccr_q_count / n_cases,
        ccr_h=ccr_h_count / n_cases,
        per_item_rates=per_item_rates,
    )


def compute_iterations_to_compliance(trace: RefinementTrace) -> Optional[int]:
    """
    Compute iterations to compliance for a single case.
    
    Args:
        trace: RefinementTrace for the case
        
    Returns:
        Number of iterations if compliant, None if not compliant
    """
    return trace.iterations_to_compliance


def aggregate_minimality_metrics(
    traces: List[RefinementTrace],
) -> Dict[str, float]:
    """
    Aggregate minimality metrics across all cases.
    
    Args:
        traces: List of RefinementTrace objects
        
    Returns:
        Dict with aggregated metrics (mean values)
    """
    if not traces:
        return {
            "mean_edit_distance_total": 0.0,
            "mean_edit_ratio_total": 0.0,
            "mean_word_changes_total": 0.0,
            "mean_iterations": 0.0,
        }
    
    total_edit_distance = 0.0
    total_edit_ratio = 0.0
    total_word_changes = 0.0
    total_iterations = 0.0
    
    for trace in traces:
        metrics = trace.minimality_metrics
        total_edit_distance += metrics.get("edit_distance_total", 0)
        total_edit_ratio += metrics.get("edit_ratio_total", 0)
        total_word_changes += metrics.get("word_changes_total", 0)
        
        if trace.iterations_to_compliance is not None:
            total_iterations += trace.iterations_to_compliance
        else:
            # Use max iterations + 1 for non-compliant cases
            total_iterations += len(trace.iterations)
    
    n = len(traces)
    
    return {
        "mean_edit_distance_total": total_edit_distance / n,
        "mean_edit_ratio_total": total_edit_ratio / n,
        "mean_word_changes_total": total_word_changes / n,
        "mean_iterations": total_iterations / n,
    }


def compute_clinical_quality_stats(
    traces: List[RefinementTrace],
) -> Dict[str, float]:
    """
    Compute statistics on clinical quality scores.
    
    Args:
        traces: List of RefinementTrace objects
        
    Returns:
        Dict with mean, min, max, and distribution of scores
    """
    scores = [
        trace.clinical_quality_score
        for trace in traces
        if trace.clinical_quality_score is not None
    ]
    
    if not scores:
        return {
            "mean_clinical_quality": 0.0,
            "min_clinical_quality": 0.0,
            "max_clinical_quality": 0.0,
        }
    
    return {
        "mean_clinical_quality": sum(scores) / len(scores),
        "min_clinical_quality": min(scores),
        "max_clinical_quality": max(scores),
    }


def compute_hard_fail_rate(traces: List[RefinementTrace]) -> float:
    """Compute rate of hard failures across cases."""
    if not traces:
        return 0.0
    
    hard_fail_count = sum(1 for trace in traces if trace.hard_fail)
    return hard_fail_count / len(traces)


def compute_compliance_rate(traces: List[RefinementTrace]) -> float:
    """Compute overall compliance rate across cases."""
    if not traces:
        return 0.0
    
    compliant_count = sum(1 for trace in traces if trace.is_compliant)
    return compliant_count / len(traces)
