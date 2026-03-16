"""
Open-ended evaluation with Iterative Adversarial Refinement.

This script extends gpt_free_text_eval.py to use the iterative refinement
pipeline with checklist enforcement.

Outputs:
- CSV with BERTScore on final_diagnosis (compatible with baseline)
- JSONL with full refinement traces, CCR metrics, and iterations/minimality
"""

import pandas as pd
import bert_score
import os
import time
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Import refinement module
from refinement import (
    IterativeRefiner,
    RefinementTrace,
    compute_ccr_metrics,
    aggregate_minimality_metrics,
    compute_compliance_rate,
    compute_clinical_quality_stats,
    compute_hard_fail_rate,
    create_refiner_variant,
    list_refiner_variants,
)
from refinement.refiner import RefinerConfig
from refinement.refiner import JudgeProvider
from refinement.io import (
    JSONLLogger,
    CSVExporter,
    hash_case_text,
    load_refinement_traces,
    save_summary_report,
)
from refinement.schema import ChecklistConfig
from eval_batching import build_eval_batches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Open-ended evaluation with iterative adversarial refinement"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/refined",
        help="Directory for output files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for generation/critique/editing",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=JudgeProvider.OPENAI.value,
        choices=[provider.value for provider in JudgeProvider],
        help="Judge backend provider",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum refinement iterations",
    )
    parser.add_argument(
        "--clinical-threshold",
        type=int,
        default=3,
        help="Minimum clinical quality score (0-5)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.65,
        help="Cosine similarity threshold for semantic_similarity_gated variant",
    )
    parser.add_argument(
        "--disclosure-fraction",
        type=float,
        default=0.2,
        help="Early-case fraction for progressive_disclosure variant",
    )
    parser.add_argument(
        "--early-confidence-threshold",
        type=float,
        default=0.8,
        help="Early confidence threshold used for revision-penalty telemetry",
    )
    parser.add_argument(
        "--revision-instability-threshold",
        type=float,
        default=0.5,
        help="Confidence shift threshold used for revision-penalty telemetry",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=4,
        help="Number of batches to sample",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250,
        help="Size of each batch",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to JSONL to resume from (for partial runs)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="baseline",
        choices=list_refiner_variants(),
        help="Refinement variant to run",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="easy",
        choices=["easy", "hard", "custom"],
        help="Dataset preset: easy (DiagnosisMedQA 20) or hard (CUPCASE_RTEST); custom requires --input",
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="unique",
        choices=["unique", "bootstrap"],
        help="Batch sampling mode: unique evaluates each case at most once; bootstrap resamples each batch independently",
    )

    args = parser.parse_args()
    if args.model is None:
        args.model = JudgeProvider(args.provider).default_model
    return args


def load_dataset(path: str) -> pd.DataFrame:
    """Load the dataset from CSV."""
    logger.info(f"Loading dataset from {path}")
    ds = pd.read_csv(path)
    logger.info(f"Loaded {len(ds)} cases")
    return ds


def sample_batches(
    ds: pd.DataFrame,
    n_batches: int,
    batch_size: int,
    random_seed: int,
    sampling_mode: str,
) -> List[pd.DataFrame]:
    """Build evaluation batches from the dataset."""
    batches = build_eval_batches(
        ds=ds,
        n_batches=n_batches,
        batch_size=batch_size,
        random_seed=random_seed,
        sampling_mode=sampling_mode,
    )
    for batch_num, batch in enumerate(batches, start=1):
        logger.info(
            "Prepared batch %s/%s with %s cases (%s mode)",
            batch_num,
            len(batches),
            len(batch),
            sampling_mode,
        )
    return batches


def process_single_case(
    refiner: IterativeRefiner,
    case_text: str,
    true_diagnosis: str,
    case_id: str,
) -> RefinementTrace:
    """Process a single case through the refinement pipeline."""
    trace = refiner.refine(
        case_text=case_text,
        case_id=case_id,
        true_diagnosis=true_diagnosis,
    )
    return trace


def compute_bertscores(
    traces: List[RefinementTrace],
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> List[float]:
    """
    Compute BERTScore F1 for all traces.
    
    Args:
        traces: List of RefinementTrace objects
        model_type: BERTScore model to use
        
    Returns:
        List of BERTScore F1 values
    """
    predictions = [trace.extracted_final_diagnosis for trace in traces]
    references = [trace.true_diagnosis for trace in traces]
    
    logger.info(f"Computing BERTScore for {len(predictions)} predictions...")
    
    P, R, F1 = bert_score.score(
        predictions,
        references,
        lang="en",
        model_type=model_type,
    )
    
    return F1.tolist()


def create_summary_report(
    traces: List[RefinementTrace],
    bertscore_f1: List[float],
    config: RefinerConfig,
    checklist_config: ChecklistConfig,
    runtime_seconds: float,
    variant: str,
) -> Dict[str, Any]:
    """Create a summary report of the evaluation."""
    import numpy as np
    
    # Compute CCR metrics
    ccr_metrics = compute_ccr_metrics(traces, checklist_config)
    
    # Compute minimality metrics
    minimality_stats = aggregate_minimality_metrics(traces)
    
    # Compute clinical quality stats
    quality_stats = compute_clinical_quality_stats(traces)
    
    # Compliance and hard fail rates
    compliance_rate = compute_compliance_rate(traces)
    hard_fail_rate = compute_hard_fail_rate(traces)
    
    # BERTScore stats
    bertscore_mean = float(np.mean(bertscore_f1))
    bertscore_std = float(np.std(bertscore_f1))
    
    # Iterations stats
    iterations_to_compliance = [
        t.iterations_to_compliance
        for t in traces
        if t.iterations_to_compliance is not None
    ]
    mean_iterations = (
        float(np.mean(iterations_to_compliance))
        if iterations_to_compliance else None
    )
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_cases": len(traces),
        "variant": variant,
        "config": config.to_dict(),
        "metrics": {
            "bertscore": {
                "mean": bertscore_mean,
                "std": bertscore_std,
            },
            "ccr": ccr_metrics.to_dict(),
            "compliance_rate": compliance_rate,
            "hard_fail_rate": hard_fail_rate,
            "clinical_quality": quality_stats,
            "minimality": minimality_stats,
            "iterations": {
                "mean_to_compliance": mean_iterations,
                "n_compliant": len(iterations_to_compliance),
                "n_non_compliant": len(traces) - len(iterations_to_compliance),
            },
        },
        "runtime_seconds": runtime_seconds,
    }
    
    return report


def main():
    """Main entry point."""
    args = parse_args()

    if args.dataset == "easy" and args.input is None:
        input_path = "datasets/DiagnosisMedQA_eval_20.csv"
    elif args.dataset == "hard" and args.input is None:
        input_path = "datasets/CUPCASE_RTEST_eval.csv"
    elif args.input is None:
        raise ValueError("--input is required when --dataset=custom")
    else:
        input_path = args.input

    if args.variant != "baseline" and args.output_dir == "output/refined":
        args.output_dir = f"output/refined_{args.variant}"
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load dataset
    ds = load_dataset(input_path)
    
    # Configure refiner
    provider = JudgeProvider(args.provider)
    config = RefinerConfig(
        generator_model=args.model,
        critic_model=args.model,
        editor_model=args.model,
        max_iterations=args.max_iterations,
        clinical_quality_threshold=args.clinical_threshold,
        similarity_threshold=args.similarity_threshold,
        disclosure_fraction=args.disclosure_fraction,
        early_confidence_threshold=args.early_confidence_threshold,
        revision_instability_threshold=args.revision_instability_threshold,
        provider=provider,
    )
    
    # Create refiner variant
    refiner = create_refiner_variant(variant=args.variant, config=config)
    
    # Load checklist config for metrics
    checklist_config = ChecklistConfig.load()
    
    # Sample batches
    batches = sample_batches(
        ds,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
        sampling_mode=args.sampling_mode,
    )

    # Process all batches
    all_traces: List[RefinementTrace] = []
    start_time = time.time()

    completed_case_hashes = set()
    if args.resume_from:
        existing_traces = load_refinement_traces(args.resume_from)
        all_traces.extend(existing_traces)
        completed_case_hashes = {trace.case_id for trace in existing_traces if trace.case_id}
        logger.info(
            "Loaded %s existing traces from %s; will skip matching case hashes",
            len(existing_traces),
            args.resume_from,
        )
    
    # JSONL logger for traces
    jsonl_path = output_dir / f"refinement_traces_{timestamp}.jsonl"
    
    with JSONLLogger(jsonl_path) as logger_jsonl:
        for batch_num, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_num + 1}/{len(batches)}")
            
            for idx, row in batch.iterrows():
                case_text = row.get("clean text", row.get("case presentation", ""))
                true_diagnosis = row.get("final diagnosis", "")
                case_id = hash_case_text(case_text)

                if case_id in completed_case_hashes:
                    logger.info("Skipping already completed case %s", case_id)
                    continue
                
                logger.info(f"Processing case {case_id}")
                
                try:
                    trace = process_single_case(
                        refiner=refiner,
                        case_text=case_text,
                        true_diagnosis=true_diagnosis,
                        case_id=case_id,
                    )

                    all_traces.append(trace)
                    completed_case_hashes.add(case_id)
                    logger_jsonl.log(trace)
                    
                except Exception as e:
                    logger.error(f"Failed to process case {case_id}: {e}")
                    continue
            
            logger.info(f"Completed batch {batch_num + 1}/{len(batches)}")
            
            # Delay between batches
            if batch_num < len(batches) - 1:
                time.sleep(10)
    
    runtime = time.time() - start_time
    logger.info(f"Processed {len(all_traces)} cases in {runtime:.1f} seconds")
    
    # Compute BERTScores
    bertscore_f1 = compute_bertscores(all_traces)
    
    # Export CSV for BERTScore compatibility
    csv_path = output_dir / f"gpt4_free_text_refined_{timestamp}.csv"
    CSVExporter.export_with_metrics(
        traces=all_traces,
        bertscore_f1=bertscore_f1,
        output_path=csv_path,
    )
    logger.info(f"Saved CSV to {csv_path}")
    
    # Create and save summary report
    report = create_summary_report(
        traces=all_traces,
        bertscore_f1=bertscore_f1,
        config=config,
        checklist_config=checklist_config,
        runtime_seconds=runtime,
        variant=args.variant,
    )
    
    report_path = output_dir / f"summary_report_{timestamp}.json"
    save_summary_report(report, report_path)
    logger.info(f"Saved summary report to {report_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("REFINEMENT EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Variant:              {args.variant}")
    print(f"Cases processed:      {len(all_traces)}")
    print(f"BERTScore F1 (mean):  {report['metrics']['bertscore']['mean']:.4f}")
    print(f"BERTScore F1 (std):   {report['metrics']['bertscore']['std']:.4f}")
    print(f"CCR_all:              {report['metrics']['ccr']['CCR_all']:.2%}")
    print(f"CCR_Q:                {report['metrics']['ccr']['CCR_Q']:.2%}")
    print(f"CCR_H:                {report['metrics']['ccr']['CCR_H']:.2%}")
    print(f"Compliance rate:      {report['metrics']['compliance_rate']:.2%}")
    print(f"Hard fail rate:       {report['metrics']['hard_fail_rate']:.2%}")
    if report['metrics']['iterations']['mean_to_compliance']:
        print(f"Mean iterations:      {report['metrics']['iterations']['mean_to_compliance']:.2f}")
    print(f"Runtime:              {runtime:.1f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
