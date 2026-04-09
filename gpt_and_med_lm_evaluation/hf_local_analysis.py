"""Run a Hugging Face local baseline/refined comparison bundle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Hugging Face local baseline/refined comparison bundle"
    )
    parser.add_argument("--baseline-variant", type=str, default="baseline")
    parser.add_argument("--refined-variant", type=str, default="domain_routed")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("HUGGINGFACE_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
    )
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/CUPCASE_RTEST_eval_20.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/hf_local_analysis",
    )
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--n-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="unique",
        choices=["unique", "bootstrap"],
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use",
    )
    return parser.parse_args()


def run_command(cmd: list[str], *, env: Dict[str, str], cwd: Optional[Path] = None) -> None:
    print("Running:")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, env=env, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def locate_single(pattern: str, folder: Path) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {pattern} in {folder}, found {len(matches)}")
    return matches[0]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def format_delta(summary: Dict[str, Any]) -> str:
    delta = summary["delta"]
    direction = summary["direction"]
    p_value = summary["p_value"]
    return f"{direction} ({delta:+.4f}, p={p_value:.4f})"


def write_markdown_summary(
    *,
    output_path: Path,
    baseline_manifest: Dict[str, Any],
    refined_manifest: Dict[str, Any],
    comparison_report: Dict[str, Any],
) -> None:
    summary = comparison_report["summary"]
    manifest_check = comparison_report.get("manifest_comparison") or {}

    lines = [
        "# HF Local Analysis Summary",
        "",
        "## What changed",
        "",
        f"- Baseline variant: `{baseline_manifest.get('variant')}`",
        f"- Refined variant: `{refined_manifest.get('variant')}`",
    ]

    for metric_name, metric_summary in summary.items():
        lines.append(f"- {metric_name}: {format_delta(metric_summary)}")

    lines.extend(
        [
            "",
            "## What stayed constant",
            "",
            f"- Provider: `{baseline_manifest.get('provider')}`",
            f"- Model: `{baseline_manifest.get('model')}`",
            f"- Input: `{baseline_manifest.get('dataset', {}).get('input_path')}`",
            f"- Seed: `{baseline_manifest.get('config', {}).get('random_seed')}`",
            f"- Sampling mode: `{baseline_manifest.get('config', {}).get('sampling_mode')}`",
            f"- Batch shape: `{baseline_manifest.get('config', {}).get('n_batches')} x {baseline_manifest.get('config', {}).get('batch_size')}`",
            "",
            "## Evidence still missing",
            "",
            "- Felipe and Max logs are not included here yet, so this bundle does not claim a root cause for external failures.",
            "- If those logs come from runs with different provider/model/input/seed settings, they must be normalized before comparison.",
        ]
    )

    if manifest_check.get("mismatches"):
        lines.append(f"- Manifest mismatches detected: `{len(manifest_check['mismatches'])}`")
        for mismatch in manifest_check["mismatches"]:
            lines.append(
                f"- {mismatch['label']}: baseline=`{mismatch['baseline']}` refined=`{mismatch['refined']}`"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = Path(args.output_dir) / timestamp
    runs_dir = bundle_dir / "runs"
    baseline_dir = runs_dir / "baseline"
    refined_dir = runs_dir / "refined"
    compare_dir = bundle_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HUGGINGFACE_LOCAL_MODEL"] = args.model

    if not args.skip_smoke:
        smoke_cmd = [
            args.python,
            str(script_dir / "huggingface_local_smoke.py"),
            "--task",
            "free_text",
            "--variant",
            args.baseline_variant,
        ]
        run_command(smoke_cmd, env=env, cwd=script_dir)

    common_run_args = [
        args.python,
        str(script_dir / "gpt_free_text_eval_refined.py"),
        "--provider",
        "huggingface_local",
        "--model",
        args.model,
        "--input",
        args.input,
        "--n-batches",
        str(args.n_batches),
        "--batch-size",
        str(args.batch_size),
        "--random-seed",
        str(args.random_seed),
        "--sampling-mode",
        args.sampling_mode,
        "--dataset",
        "custom",
    ]

    baseline_cmd = common_run_args + [
        "--variant",
        args.baseline_variant,
        "--output-dir",
        str(baseline_dir),
    ]
    refined_cmd = common_run_args + [
        "--variant",
        args.refined_variant,
        "--output-dir",
        str(refined_dir),
    ]

    run_command(baseline_cmd, env=env, cwd=script_dir)
    run_command(refined_cmd, env=env, cwd=script_dir)

    baseline_csv = locate_single("gpt4_free_text_refined_*.csv", baseline_dir)
    baseline_manifest = locate_single("run_manifest_*.json", baseline_dir)
    refined_csv = locate_single("gpt4_free_text_refined_*.csv", refined_dir)
    refined_manifest = locate_single("run_manifest_*.json", refined_dir)
    refined_traces = locate_single("refinement_traces_*.jsonl", refined_dir)

    comparison_report = compare_dir / "comparison_report.json"
    compare_cmd = [
        args.python,
        str(script_dir / "compare_baseline_vs_refined.py"),
        "--baseline",
        str(baseline_csv),
        "--refined",
        str(refined_csv),
        "--refined-traces",
        str(refined_traces),
        "--baseline-manifest",
        str(baseline_manifest),
        "--refined-manifest",
        str(refined_manifest),
        "--output",
        str(comparison_report),
    ]
    run_command(compare_cmd, env=env, cwd=script_dir)

    baseline_manifest_data = load_json(baseline_manifest)
    refined_manifest_data = load_json(refined_manifest)
    comparison_report_data = load_json(comparison_report)
    markdown_summary = bundle_dir / "analysis_summary.md"
    write_markdown_summary(
        output_path=markdown_summary,
        baseline_manifest=baseline_manifest_data,
        refined_manifest=refined_manifest_data,
        comparison_report=comparison_report_data,
    )

    print(f"Analysis bundle: {bundle_dir}")
    print(f"Summary: {markdown_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
