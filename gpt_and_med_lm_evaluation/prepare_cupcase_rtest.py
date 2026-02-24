"""
Convert CUPCASE_RTEST Hugging Face dataset to the MCQ CSV schema used by the
evaluation runners. The schema matches `prepare_hf_diagnosismedqa.py` output:

    id, clean text, final diagnosis, distractor1, distractor2, distractor3

Usage examples:

```
python prepare_cupcase_rtest.py \
  --hf-token $HF_TOKEN \
  --output datasets/CUPCASE_RTEST_eval.csv

python prepare_cupcase_rtest.py \
  --hf-token $HF_TOKEN \
  --sample-size 20 --seed 42 \
  --output datasets/CUPCASE_RTEST_eval_20.csv
```
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "oriel9p/CUPCASE_RTEST"
DEFAULT_FILENAME = "ext_test.jsonl"
OUTPUT_FIELDS = [
    "id",
    "clean text",
    "final diagnosis",
    "distractor1",
    "distractor2",
    "distractor3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CUPCASE_RTEST HF dataset to CUPCase eval CSV")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token (or set HF_TOKEN)")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="HF dataset repo id")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="Dataset file within the repo")
    parser.add_argument("--output", type=str, default="datasets/CUPCASE_RTEST_eval.csv", help="Output CSV path")
    parser.add_argument("--sample-size", type=int, default=0, help="Optional sample size; 0 keeps all rows")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    return parser.parse_args()


def download_jsonl(repo_id: str, filename: str, hf_token: str | None, cache_dir: Path | None) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            token=hf_token,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def maybe_sample(rows: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def pick_str(record: Dict[str, Any], key: str, fallback: str = "") -> str:
    value = record.get(key, fallback)
    if value is None:
        return fallback
    return str(value)


def convert_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    converted: List[Dict[str, str]] = []
    for record in rows:
        distractors = [
            pick_str(record, "distractor1"),
            pick_str(record, "distractor2"),
            pick_str(record, "distractor3"),
            pick_str(record, "distractor4"),
            pick_str(record, "distractor5"),
        ]
        converted.append(
            {
                "id": pick_str(record, "case_presentation_index"),
                "clean text": pick_str(record, "clean text", pick_str(record, "case_presentation")),
                "final diagnosis": pick_str(record, "final_answer", pick_str(record, "final diagnosis")),
                "distractor1": distractors[0],
                "distractor2": distractors[1],
                "distractor3": distractors[2],
            }
        )
    return converted


def save_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    hf_token = args.hf_token or None
    cache_dir = Path("/tmp/hf_home")

    jsonl_path = download_jsonl(args.repo_id, args.filename, hf_token, cache_dir)
    raw_rows = load_rows(jsonl_path)
    sampled = maybe_sample(raw_rows, args.sample_size, args.seed)
    converted = convert_rows(sampled)
    save_csv(converted, Path(args.output))

    print(f"Loaded rows: {len(raw_rows)}")
    print(f"Saved rows:  {len(converted)}")
    print(f"Output:      {args.output}")


if __name__ == "__main__":
    main()
