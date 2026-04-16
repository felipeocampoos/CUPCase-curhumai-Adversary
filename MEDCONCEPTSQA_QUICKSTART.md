# MedConceptsQA Quickstart

This repo supports `ofir408/MedConceptsQA` on two surfaces:

- `lm_eval_evaluation/`: native `lm_eval` task runner
- `gpt_and_med_lm_evaluation/`: CUPCase-style MCQ evaluator path

## Fresh Setup

From repo root:

```bash
cd gpt_and_med_lm_evaluation
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements-py312.txt
pip install pytest
pip install -e ../lm_eval_evaluation
```

Notes:

- The commands below assume this environment is active.
- `prepare_hf_medconceptsqa.py` is the dataset download/materialization script.
- It pulls the dataset from Hugging Face and writes a CUPCase-compatible CSV locally.

## Subset Syntax

Use `--subset` in one of these forms:

- `all`
- vocab-only: `icd9cm`, `icd10cm`, `icd9proc`, `icd10proc`, `atc`
- vocab + difficulty: `<vocab>_easy`, `<vocab>_medium`, `<vocab>_hard`

Examples:

- `icd10cm_easy`
- `icd10cm_medium`
- `icd10cm_hard`
- `atc_hard`
- `icd9proc_medium`

## Sample Count

- `lm_eval_evaluation/scripts/run_medconceptsqa.py`
  - `--sample-size` is forwarded to `lm_eval` as the harness `--limit`
  - use it for smoke runs and debugging
- `gpt_and_med_lm_evaluation/run_medconceptsqa_mcq.py`
  - `--sample-size` controls how many dataset rows are materialized into the generated CSV

## 1) lm_eval Harness

From repo root:

```bash
cd lm_eval_evaluation
```

Full benchmark:

```bash
python scripts/run_medconceptsqa.py \
  --model hf \
  --model-args pretrained=BioMistral/BioMistral-7B-DARE \
  --subset all \
  --device cuda:0 \
  --batch-size auto
```

Difficulty-specific sample:

```bash
python scripts/run_medconceptsqa.py \
  --model hf \
  --model-args pretrained=BioMistral/BioMistral-7B-DARE \
  --subset atc_hard \
  --sample-size 50 \
  --device cuda:0 \
  --batch-size auto \
  --log-samples
```

Smoke test:

```bash
python scripts/medconceptsqa_smoke.py --keep-output
```

Outputs land under:

```text
lm_eval_evaluation/output/medconceptsqa/<subset>/<model_slug>/<timestamp>/
```

## 2) CUPCase MCQ Evaluator

From repo root:

```bash
cd gpt_and_med_lm_evaluation
```

Download and materialize an evaluator-compatible CSV:

```bash
python prepare_hf_medconceptsqa.py \
  --subset icd10cm_easy \
  --split test \
  --sample-size 25
```

Generated CSVs land under:

```text
gpt_and_med_lm_evaluation/datasets/generated/medconceptsqa/
```

Difficulty-specific download/materialization examples:

```bash
python prepare_hf_medconceptsqa.py \
  --subset icd10cm_hard \
  --split test \
  --sample-size 50

python prepare_hf_medconceptsqa.py \
  --subset atc_medium \
  --split test \
  --sample-size 100
```

Run the MCQ evaluator:

```bash
python run_medconceptsqa_mcq.py \
  --provider huggingface_local \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --subset icd10cm_easy \
  --sample-size 25 \
  --variant baseline
```

Difficulty-specific sample:

```bash
python run_medconceptsqa_mcq.py \
  --provider huggingface_local \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --subset icd9proc_medium \
  --sample-size 10 \
  --variant baseline
```

Smoke test:

```bash
python medconceptsqa_smoke.py --keep-output
```

Outputs land under:

```text
gpt_and_med_lm_evaluation/output/experiments/medconceptsqa/<subset>/<provider>/mcq/<variant>/<model_slug>/<sample_label>/
```

## Verified Smoke Output Paths

Current verified examples:

- `gpt_and_med_lm_evaluation/datasets/generated/medconceptsqa/icd10cm_easy_test_n1_seed42.csv`
- `gpt_and_med_lm_evaluation/output/experiments/medconceptsqa_smoke/icd10cm_easy/huggingface_local/mcq/baseline/sshleifer_tiny-gpt2/n1_seed42/results.csv`
- `gpt_and_med_lm_evaluation/output/experiments/medconceptsqa_smoke/icd10cm_easy/huggingface_local/mcq/baseline/sshleifer_tiny-gpt2/n1_seed42/run_manifest_20260416_144615.json`
- `lm_eval_evaluation/output/medconceptsqa_smoke/icd10cm_easy/dummy/20260416_144606/results_2026-04-16T14-46-14.594549.json`
