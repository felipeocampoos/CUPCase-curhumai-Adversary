"""Generate case-level failure analysis reports from refinement traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from refinement.analysis import analyze_runs, collect_run_artifacts, render_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze refinement traces and emit case-level failure reports"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more refinement trace JSONL files or directories containing them",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/failure_analysis",
        help="Directory for JSON and Markdown analysis reports",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = collect_run_artifacts(args.inputs)
    if not artifacts:
        raise SystemExit("No refinement trace files found in the provided inputs")

    report = analyze_runs(artifacts)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "failure_case_analysis.json"
    markdown_path = output_dir / "failure_case_analysis.md"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")

    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
