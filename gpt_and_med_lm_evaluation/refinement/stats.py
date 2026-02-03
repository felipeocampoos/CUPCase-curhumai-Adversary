"""
Statistical comparison utilities for paired analysis.

Provides:
- Paired bootstrap confidence intervals
- Paired permutation tests for p-values
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    """Result of paired bootstrap analysis."""
    
    mean_difference: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_difference": self.mean_difference,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence_level": self.confidence_level,
            "n_bootstrap": self.n_bootstrap,
            "significant": self.is_significant(),
        }
    
    def is_significant(self) -> bool:
        """Check if the difference is statistically significant (CI excludes 0)."""
        return (self.ci_lower > 0) or (self.ci_upper < 0)


@dataclass
class PermutationResult:
    """Result of paired permutation test."""
    
    observed_difference: float
    p_value: float
    n_permutations: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "observed_difference": self.observed_difference,
            "p_value": self.p_value,
            "n_permutations": self.n_permutations,
            "significant_05": self.p_value < 0.05,
            "significant_01": self.p_value < 0.01,
        }


def paired_bootstrap_ci(
    baseline: List[float],
    refined: List[float],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = 42,
) -> BootstrapResult:
    """
    Compute paired bootstrap confidence interval for the difference.
    
    Args:
        baseline: Baseline metric values (one per case)
        refined: Refined metric values (one per case)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
        
    Returns:
        BootstrapResult with mean difference and confidence interval
        
    Raises:
        ValueError: If inputs have different lengths or are empty
    """
    if len(baseline) != len(refined):
        raise ValueError(
            f"Baseline and refined must have same length: {len(baseline)} vs {len(refined)}"
        )
    
    if len(baseline) == 0:
        raise ValueError("Cannot compute bootstrap on empty data")
    
    baseline_arr = np.array(baseline)
    refined_arr = np.array(refined)
    
    # Compute paired differences
    differences = refined_arr - baseline_arr
    observed_mean_diff = np.mean(differences)
    
    # Set random seed
    rng = np.random.default_rng(random_seed)
    
    # Bootstrap sampling
    n_samples = len(differences)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = differences[indices]
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    
    return BootstrapResult(
        mean_difference=float(observed_mean_diff),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )


def paired_permutation_test(
    baseline: List[float],
    refined: List[float],
    n_permutations: int = 10000,
    random_seed: Optional[int] = 42,
) -> PermutationResult:
    """
    Compute paired permutation test for the difference.
    
    Tests null hypothesis that the distribution of differences
    is symmetric around zero.
    
    Args:
        baseline: Baseline metric values (one per case)
        refined: Refined metric values (one per case)
        n_permutations: Number of permutations
        random_seed: Random seed for reproducibility
        
    Returns:
        PermutationResult with observed difference and p-value
        
    Raises:
        ValueError: If inputs have different lengths or are empty
    """
    if len(baseline) != len(refined):
        raise ValueError(
            f"Baseline and refined must have same length: {len(baseline)} vs {len(refined)}"
        )
    
    if len(baseline) == 0:
        raise ValueError("Cannot compute permutation test on empty data")
    
    baseline_arr = np.array(baseline)
    refined_arr = np.array(refined)
    
    # Compute paired differences
    differences = refined_arr - baseline_arr
    observed_mean_diff = np.mean(differences)
    
    # Set random seed
    rng = np.random.default_rng(random_seed)
    
    # Permutation test: randomly flip signs of differences
    n_samples = len(differences)
    more_extreme_count = 0
    
    for _ in range(n_permutations):
        # Randomly flip signs
        signs = rng.choice([-1, 1], size=n_samples)
        permuted_diffs = differences * signs
        permuted_mean = np.mean(permuted_diffs)
        
        # Two-tailed test
        if abs(permuted_mean) >= abs(observed_mean_diff):
            more_extreme_count += 1
    
    # Compute p-value
    p_value = (more_extreme_count + 1) / (n_permutations + 1)
    
    return PermutationResult(
        observed_difference=float(observed_mean_diff),
        p_value=float(p_value),
        n_permutations=n_permutations,
    )


def compare_metrics_paired(
    baseline_values: Dict[str, List[float]],
    refined_values: Dict[str, List[float]],
    n_bootstrap: int = 10000,
    n_permutations: int = 10000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple metrics between baseline and refined using paired tests.
    
    Args:
        baseline_values: Dict mapping metric names to lists of baseline values
        refined_values: Dict mapping metric names to lists of refined values
        n_bootstrap: Number of bootstrap samples
        n_permutations: Number of permutations for p-value
        confidence_level: Confidence level for bootstrap CI
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict mapping metric names to comparison results
    """
    results = {}
    
    for metric_name in baseline_values.keys():
        if metric_name not in refined_values:
            continue
        
        baseline = baseline_values[metric_name]
        refined = refined_values[metric_name]
        
        # Compute bootstrap CI
        bootstrap_result = paired_bootstrap_ci(
            baseline=baseline,
            refined=refined,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )
        
        # Compute permutation test
        permutation_result = paired_permutation_test(
            baseline=baseline,
            refined=refined,
            n_permutations=n_permutations,
            random_seed=random_seed,
        )
        
        results[metric_name] = {
            "baseline_mean": float(np.mean(baseline)),
            "refined_mean": float(np.mean(refined)),
            "bootstrap": bootstrap_result.to_dict(),
            "permutation": permutation_result.to_dict(),
        }
    
    return results


def format_comparison_report(
    comparison_results: Dict[str, Dict[str, Any]],
) -> str:
    """
    Format comparison results as a human-readable report.
    
    Args:
        comparison_results: Output from compare_metrics_paired
        
    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("PAIRED COMPARISON REPORT: Baseline vs Refined")
    lines.append("=" * 70)
    lines.append("")
    
    for metric_name, result in comparison_results.items():
        lines.append(f"Metric: {metric_name}")
        lines.append("-" * 50)
        
        baseline_mean = result["baseline_mean"]
        refined_mean = result["refined_mean"]
        
        lines.append(f"  Baseline Mean: {baseline_mean:.4f}")
        lines.append(f"  Refined Mean:  {refined_mean:.4f}")
        
        bootstrap = result["bootstrap"]
        lines.append(f"  Difference:    {bootstrap['mean_difference']:+.4f}")
        lines.append(
            f"  95% CI:        [{bootstrap['ci_lower']:+.4f}, {bootstrap['ci_upper']:+.4f}]"
        )
        
        permutation = result["permutation"]
        lines.append(f"  P-value:       {permutation['p_value']:.4f}")
        
        # Significance stars
        if permutation["p_value"] < 0.001:
            sig = "***"
        elif permutation["p_value"] < 0.01:
            sig = "**"
        elif permutation["p_value"] < 0.05:
            sig = "*"
        else:
            sig = "ns"
        
        lines.append(f"  Significance:  {sig}")
        lines.append("")
    
    lines.append("=" * 70)
    lines.append("Significance: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant")
    lines.append("=" * 70)
    
    return "\n".join(lines)
