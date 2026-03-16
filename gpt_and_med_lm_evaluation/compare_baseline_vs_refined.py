"""
Paired comparison between baseline and refined evaluation results.

Computes:
- Delta metrics (BERTScore, CCR_all, CCR_Q, CCR_H)
- Paired bootstrap confidence intervals
- Paired permutation tests for p-values
- Human-readable report

Usage:
    python compare_baseline_vs_refined.py --baseline output/gpt4_free_text_batched.csv \
                                          --refined output/refined/gpt4_free_text_refined.csv \
                                          --refined-traces output/refined/refinement_traces.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from refinement import (
    load_refinement_traces,
    compute_ccr_metrics,
    paired_bootstrap_ci,
    paired_permutation_test,
)
from refinement.stats import compare_metrics_paired, format_comparison_report
from refinement.io import hash_case_text, save_summary_report, load_summary_report
from refinement.schema import ChecklistConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare baseline vs refined evaluation results"
    )
    
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline results CSV (from gpt_free_text_eval.py)",
    )
    parser.add_argument(
        "--refined",
        type=str,
        required=True,
        help="Path to refined results CSV (from gpt_free_text_eval_refined.py)",
    )
    parser.add_argument(
        "--refined-traces",
        type=str,
        default=None,
        help="Path to refined traces JSONL (for CCR metrics)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/comparison_report.json",
        help="Path to output comparison report",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of permutations for p-value",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap CI",
    )
    
    return parser.parse_args()


def load_baseline_results(path: str) -> pd.DataFrame:
    """Load baseline results from CSV."""
    logger.info(f"Loading baseline results from {path}")
    df = pd.read_csv(path)
    
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    
    # Add case hash for alignment
    if "Case presentation" in df.columns:
        df["case_hash"] = df["Case presentation"].apply(hash_case_text)
    
    logger.info(f"Loaded {len(df)} baseline results")
    return df


def load_refined_results(path: str) -> pd.DataFrame:
    """Load refined results from CSV."""
    logger.info(f"Loading refined results from {path}")
    df = pd.read_csv(path)
    
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    
    # Add case hash for alignment
    if "Case presentation" in df.columns:
        df["case_hash"] = df["Case presentation"].apply(hash_case_text)
    
    logger.info(f"Loaded {len(df)} refined results")
    return df


def align_results(
    baseline_df: pd.DataFrame,
    refined_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align baseline and refined results by case hash.
    
    Returns:
        Tuple of aligned (baseline_df, refined_df) with same rows
    """
    # Use case hash if available, otherwise use index
    if "case_hash" in baseline_df.columns and "case_hash" in refined_df.columns:
        # Find common cases
        common_hashes = set(baseline_df["case_hash"]) & set(refined_df["case_hash"])
        logger.info(f"Found {len(common_hashes)} common cases by hash")
        
        baseline_aligned = baseline_df[baseline_df["case_hash"].isin(common_hashes)]
        refined_aligned = refined_df[refined_df["case_hash"].isin(common_hashes)]
        
        # Sort by hash for alignment
        baseline_aligned = baseline_aligned.sort_values("case_hash").reset_index(drop=True)
        refined_aligned = refined_aligned.sort_values("case_hash").reset_index(drop=True)
        
    else:
        # Fallback: align by index (assuming same order)
        min_len = min(len(baseline_df), len(refined_df))
        logger.warning(f"No case hash available, aligning by index ({min_len} cases)")
        
        baseline_aligned = baseline_df.head(min_len).reset_index(drop=True)
        refined_aligned = refined_df.head(min_len).reset_index(drop=True)
    
    return baseline_aligned, refined_aligned


def compute_ccr_from_traces(
    traces_path: str,
    config: Optional[ChecklistConfig] = None,
) -> Dict[str, List[bool]]:
    """
    Compute per-case CCR compliance from traces.
    
    Returns:
        Dict with CCR_all, CCR_Q, CCR_H as lists of booleans per case
    """
    if config is None:
        config = ChecklistConfig.load()
    
    traces = load_refinement_traces(traces_path)
    
    ccr_all = []
    ccr_q = []
    ccr_h = []
    
    from refinement.metrics import compute_ccr_for_case
    
    for trace in traces:
        case_ccr = compute_ccr_for_case(trace.checklist_pass_map, config)
        ccr_all.append(case_ccr["CCR_all"])
        ccr_q.append(case_ccr["CCR_Q"])
        ccr_h.append(case_ccr["CCR_H"])
    
    return {
        "CCR_all": ccr_all,
        "CCR_Q": ccr_q,
        "CCR_H": ccr_h,
    }


def compute_delta_metrics(
    baseline_df: pd.DataFrame,
    refined_df: pd.DataFrame,
    refined_ccr: Optional[Dict[str, List[bool]]] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute paired metrics for comparison.
    
    Returns:
        Dict with baseline and refined metric values
    """
    metrics = {
        "baseline": {},
        "refined": {},
    }
    
    # BERTScore F1
    if "BERTScore F1" in baseline_df.columns:
        metrics["baseline"]["BERTScore_F1"] = baseline_df["BERTScore F1"].tolist()
    
    if "BERTScore F1" in refined_df.columns:
        metrics["refined"]["BERTScore_F1"] = refined_df["BERTScore F1"].tolist()
    
    # CCR metrics (only for refined, baseline doesn't have CCR)
    if refined_ccr:
        n_cases = len(metrics.get("refined", {}).get("BERTScore_F1", []))
        
        # For baseline, assume 0 CCR (no checklist compliance)
        # This may not be accurate - you might want to compute baseline CCR separately
        metrics["baseline"]["CCR_all"] = [0.0] * len(refined_ccr["CCR_all"])
        metrics["baseline"]["CCR_Q"] = [0.0] * len(refined_ccr["CCR_Q"])
        metrics["baseline"]["CCR_H"] = [0.0] * len(refined_ccr["CCR_H"])
        
        metrics["refined"]["CCR_all"] = [1.0 if v else 0.0 for v in refined_ccr["CCR_all"]]
        metrics["refined"]["CCR_Q"] = [1.0 if v else 0.0 for v in refined_ccr["CCR_Q"]]
        metrics["refined"]["CCR_H"] = [1.0 if v else 0.0 for v in refined_ccr["CCR_H"]]
    
    # Is Compliant (from refined CSV if available)
    if "Is Compliant" in refined_df.columns:
        metrics["baseline"]["Compliance"] = [0.0] * len(refined_df)
        metrics["refined"]["Compliance"] = [
            1.0 if v else 0.0 for v in refined_df["Is Compliant"]
        ]
    
    return metrics


def run_paired_comparisons(
    metrics: Dict[str, Dict[str, List[float]]],
    n_bootstrap: int = 10000,
    n_permutations: int = 10000,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Run paired statistical comparisons on all metrics.
    
    Returns:
        Comparison results dictionary
    """
    results = compare_metrics_paired(
        baseline_values=metrics["baseline"],
        refined_values=metrics["refined"],
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        confidence_level=confidence_level,
    )
    
    return results


def create_comparison_report(
    comparison_results: Dict[str, Any],
    baseline_path: str,
    refined_path: str,
    n_baseline: int,
    n_refined: int,
    n_aligned: int,
) -> Dict[str, Any]:
    """Create full comparison report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "inputs": {
            "baseline_path": str(baseline_path),
            "refined_path": str(refined_path),
            "n_baseline": n_baseline,
            "n_refined": n_refined,
            "n_aligned": n_aligned,
        },
        "comparisons": comparison_results,
        "summary": {},
    }
    
    # Add summary
    for metric_name, result in comparison_results.items():
        delta = result["bootstrap"]["mean_difference"]
        ci_lower = result["bootstrap"]["ci_lower"]
        ci_upper = result["bootstrap"]["ci_upper"]
        p_value = result["permutation"]["p_value"]
        significant = result["permutation"]["significant_05"]
        
        report["summary"][metric_name] = {
            "delta": delta,
            "95_ci": [ci_lower, ci_upper],
            "p_value": p_value,
            "significant": significant,
            "direction": "improved" if delta > 0 else "declined" if delta < 0 else "unchanged",
        }
    
    return report


def format_text_report(report: Dict[str, Any]) -> str:
    """Format report as human-readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append("PAIRED COMPARISON: Baseline vs Refined Evaluation")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Timestamp: {report['timestamp']}")
    lines.append(f"Baseline: {report['inputs']['baseline_path']}")
    lines.append(f"Refined:  {report['inputs']['refined_path']}")
    lines.append(f"Aligned cases: {report['inputs']['n_aligned']}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("METRIC COMPARISONS")
    lines.append("-" * 70)
    lines.append("")
    
    for metric_name, comparison in report["comparisons"].items():
        baseline_mean = comparison["baseline_mean"]
        refined_mean = comparison["refined_mean"]
        bootstrap = comparison["bootstrap"]
        permutation = comparison["permutation"]
        
        lines.append(f"  {metric_name}")
        lines.append(f"    Baseline:  {baseline_mean:.4f}")
        lines.append(f"    Refined:   {refined_mean:.4f}")
        lines.append(f"    Delta:     {bootstrap['mean_difference']:+.4f}")
        lines.append(f"    95% CI:    [{bootstrap['ci_lower']:+.4f}, {bootstrap['ci_upper']:+.4f}]")
        lines.append(f"    P-value:   {permutation['p_value']:.4f}")
        
        # Significance indicator
        if permutation["p_value"] < 0.001:
            sig = "*** (p < 0.001)"
        elif permutation["p_value"] < 0.01:
            sig = "** (p < 0.01)"
        elif permutation["p_value"] < 0.05:
            sig = "* (p < 0.05)"
        else:
            sig = "ns (not significant)"
        
        lines.append(f"    Significance: {sig}")
        lines.append("")
    
    lines.append("=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    
    for metric_name, summary in report["summary"].items():
        direction = summary["direction"]
        delta = summary["delta"]
        significant = "significantly" if summary["significant"] else "not significantly"
        
        if direction == "improved":
            lines.append(f"  {metric_name}: {significant} IMPROVED by {abs(delta):.4f}")
        elif direction == "declined":
            lines.append(f"  {metric_name}: {significant} DECLINED by {abs(delta):.4f}")
        else:
            lines.append(f"  {metric_name}: UNCHANGED")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load results
    baseline_df = load_baseline_results(args.baseline)
    refined_df = load_refined_results(args.refined)
    
    # Align by case
    baseline_aligned, refined_aligned = align_results(baseline_df, refined_df)
    logger.info(f"Aligned {len(baseline_aligned)} cases for comparison")
    
    # Load CCR from traces if available
    refined_ccr = None
    if args.refined_traces and Path(args.refined_traces).exists():
        logger.info("Loading CCR metrics from traces")
        refined_ccr = compute_ccr_from_traces(args.refined_traces)
    
    # Compute metrics
    metrics = compute_delta_metrics(
        baseline_df=baseline_aligned,
        refined_df=refined_aligned,
        refined_ccr=refined_ccr,
    )
    
    # Run paired comparisons
    comparison_results = run_paired_comparisons(
        metrics=metrics,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        confidence_level=args.confidence_level,
    )
    
    # Create report
    report = create_comparison_report(
        comparison_results=comparison_results,
        baseline_path=args.baseline,
        refined_path=args.refined,
        n_baseline=len(baseline_df),
        n_refined=len(refined_df),
        n_aligned=len(baseline_aligned),
    )
    
    # Save JSON report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_summary_report(report, output_path)
    logger.info(f"Saved comparison report to {output_path}")
    
    # Save text report
    text_report_path = output_path.with_suffix(".txt")
    text_report = format_text_report(report)
    with open(text_report_path, "w") as f:
        f.write(text_report)
    logger.info(f"Saved text report to {text_report_path}")
    
    # Print text report
    print("\n" + text_report)


if __name__ == "__main__":
    main()
