## CUPCase: End-to-End Experiment Runbook

This repository contains CUPCase evaluation and preprocessing code. This runbook is the canonical way to run the paper-core experiment matrix end-to-end and compare methods later.

### What this runbook covers

- API-based MCQ and free-text evaluation in `gpt_and_med_lm_evaluation/`
- Refined variants (baseline + all implemented methods)
- Full vs partial dataset modes
- One-command Slurm submission for the full matrix
- Comparison report generation for baseline vs refined

### What this runbook does **not** cover

- Full preprocessing/regeneration pipeline in `preprocess/` and `utils/`
- On-prem model benchmarking in `lm_eval_evaluation/`

Those workflows are intentionally out of this orchestration to keep the paper-core comparison matrix reproducible.

---

## 1) Environment setup

From repository root:

```bash
cd gpt_and_med_lm_evaluation
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements-py312.txt
python check_env.py
```

Expected check output: `Environment check passed.`

Notes:
- Repo root `.python-version` is pinned to `3.12.5`.
- Use `requirements-py312.txt` (not `requierments.txt`) for reproducible runs.

---

## 2) Required environment variables

Create `gpt_and_med_lm_evaluation/.env`:

```bash
OPENAI_API_KEY="<your_openai_key>"
DEEPSEEK_API_KEY="<your_deepseek_key>"
```

Key usage:
- OpenAI runs require `OPENAI_API_KEY`
- DeepSeek runs require `DEEPSEEK_API_KEY`

---

## 3) Dataset modes used by this runbook

This runbook uses pre-existing DiagnosisMedQA-formatted CSVs:

- `full`: `gpt_and_med_lm_evaluation/datasets/DiagnosisMedQA_eval_20.csv`
- `partial`: `gpt_and_med_lm_evaluation/datasets/DiagnosisMedQA_eval_20_first10.csv`

If needed, you can regenerate compatible files with:

```bash
cd gpt_and_med_lm_evaluation
source .venv312/bin/activate
python prepare_hf_diagnosismedqa.py --output datasets/DiagnosisMedQA_eval_custom.csv
python prepare_hf_diagnosismedqa.py --sample-size 10 --output datasets/DiagnosisMedQA_eval_custom_first10.csv
```

### 3.1 Full Hugging Face integration instructions

Dataset source:
- `oriel9p/DiagnosisMedQA`
- Split: `train` (829 rows)
- Expected columns: `id`, `clean_case_presentation`, `correct_diagnosis`, `distractor1`, `distractor2`, `distractor3`

Prepare evaluator-compatible CSV from Hugging Face:

```bash
cd gpt_and_med_lm_evaluation
source .venv312/bin/activate
pip install datasets
python prepare_hf_diagnosismedqa.py \
  --dataset oriel9p/DiagnosisMedQA \
  --split train \
  --output datasets/DiagnosisMedQA_eval.csv
```

Optional quick subset for fast iteration:

```bash
python prepare_hf_diagnosismedqa.py \
  --output datasets/DiagnosisMedQA_eval_100.csv \
  --sample-size 100 \
  --seed 42
```

Run the domain-routed free-text refinement on the HF-prepared CSV:

```bash
python gpt_free_text_eval_refined.py \
  --variant domain_routed \
  --input datasets/DiagnosisMedQA_eval.csv \
  --output-dir output/refined_domain_routed_diagnosismedqa \
  --n-batches 1 \
  --batch-size 250
```

Run MCQ refined evaluation on the same HF-prepared CSV:

```bash
python gpt_qa_eval_refined.py \
  --variant discriminative_question \
  --input datasets/DiagnosisMedQA_eval.csv \
  --output output/gpt4_multiple_choice_refined_hf.csv \
  --provider openai \
  --n-batches 1 \
  --batch-size 250
```

---

## 4) Methods included in “all methods”

### MCQ refined runner
Script: `gpt_and_med_lm_evaluation/gpt_qa_eval_refined.py`

Providers:
- `openai`
- `deepseek`

Variants:
- `baseline`
- `semantic_similarity_gated`
- `discriminative_question`
- `progressive_disclosure`

### Free-text refined runner
Script: `gpt_and_med_lm_evaluation/gpt_free_text_eval_refined.py`

Variants:
- `baseline`
- `domain_routed`
- `semantic_similarity_gated`
- `discriminative_question`
- `progressive_disclosure`

### DeepSeek free-text baseline
Script: `gpt_and_med_lm_evaluation/deepseek_free_text_eval.py`

### Comparison reports
Script: `gpt_and_med_lm_evaluation/compare_baseline_vs_refined.py`

Compares free-text baseline-equivalent output vs each refined variant output.

---

## 5) Output structure (standardized)

All runs in this runbook write under:

`gpt_and_med_lm_evaluation/output/experiments/<dataset_mode>/<provider>/<task>/<variant>/`

Examples:
- `.../full/openai/mcq/baseline/results.csv`
- `.../partial/openai/free_text/discriminative_question/summary_report_*.json`
- `.../full/deepseek/free_text/baseline/results.csv`
- `.../full/openai/comparisons/domain_routed/comparison_report.json`

---

## 6) Local one-off commands (without Slurm)

From `gpt_and_med_lm_evaluation/` with `.venv312` active.

### 6.1 MCQ refined (OpenAI example)

```bash
python gpt_qa_eval_refined.py \
  --input datasets/DiagnosisMedQA_eval_20.csv \
  --output output/experiments/full/openai/mcq/discriminative_question/results.csv \
  --provider openai \
  --variant discriminative_question \
  --n-batches 1 \
  --batch-size 20
```

### 6.2 MCQ refined (DeepSeek example)

```bash
python gpt_qa_eval_refined.py \
  --input datasets/DiagnosisMedQA_eval_20.csv \
  --output output/experiments/full/deepseek/mcq/semantic_similarity_gated/results.csv \
  --provider deepseek \
  --variant semantic_similarity_gated \
  --n-batches 1 \
  --batch-size 20
```

### 6.3 Free-text refined (OpenAI example)

```bash
python gpt_free_text_eval_refined.py \
  --input datasets/DiagnosisMedQA_eval_20.csv \
  --output-dir output/experiments/full/openai/free_text/domain_routed \
  --variant domain_routed \
  --model gpt-4o \
  --n-batches 1 \
  --batch-size 20
```

### 6.4 DeepSeek free-text baseline

```bash
python deepseek_free_text_eval.py \
  --input datasets/DiagnosisMedQA_eval_20.csv \
  --output output/experiments/full/deepseek/free_text/baseline/results.csv \
  --n-batches 1 \
  --batch-size 20
```

### 6.5 Compare baseline vs refined

```bash
python compare_baseline_vs_refined.py \
  --baseline output/experiments/full/openai/free_text/baseline/gpt4_free_text_refined_YYYYMMDD_HHMMSS.csv \
  --refined output/experiments/full/openai/free_text/domain_routed/gpt4_free_text_refined_YYYYMMDD_HHMMSS.csv \
  --refined-traces output/experiments/full/openai/free_text/domain_routed/refinement_traces_YYYYMMDD_HHMMSS.jsonl \
  --output output/experiments/full/openai/comparisons/domain_routed/comparison_report.json
```

---

## 7) Full experiment with one Slurm file

Slurm file added by this runbook:

`slurm/run_full_experiment_array.slurm`

### 7.1 Submit full matrix

From repo root:

```bash
sbatch slurm/run_full_experiment_array.slurm
```

### 7.2 Submit only a subset of array indices

```bash
sbatch --array=0-7 slurm/run_full_experiment_array.slurm
```

### 7.3 Override compute resources at submission time

```bash
sbatch \
  --account=my_account \
  --partition=gpu \
  --time=12:00:00 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=32G \
  slurm/run_full_experiment_array.slurm
```

### 7.4 Override dataset files (optional)

```bash
sbatch \
  --export=ALL,FULL_DATASET=datasets/DiagnosisMedQA_eval_20.csv,PARTIAL_DATASET=datasets/DiagnosisMedQA_eval_20_first10.csv \
  slurm/run_full_experiment_array.slurm
```

---

## 8) Slurm matrix details

The array includes:

- MCQ refined:
  - 2 dataset modes (`full`, `partial`)
  - 2 providers (`openai`, `deepseek`)
  - 4 variants (`baseline`, `semantic_similarity_gated`, `discriminative_question`, `progressive_disclosure`)
- Free-text refined (OpenAI):
  - 2 dataset modes
  - 5 variants (`baseline`, `domain_routed`, `semantic_similarity_gated`, `discriminative_question`, `progressive_disclosure`)
- DeepSeek free-text baseline:
  - 2 dataset modes
- Comparison jobs (OpenAI free-text baseline vs each non-baseline refined variant):
  - 2 dataset modes
  - 4 refined variants

Total array tasks: **36**

---

## 9) Troubleshooting

- Missing API key:
  - OpenAI jobs fail if `OPENAI_API_KEY` is unset.
  - DeepSeek jobs fail if `DEEPSEEK_API_KEY` is unset.
- Missing dataset file:
  - Ensure `FULL_DATASET` / `PARTIAL_DATASET` paths exist relative to `gpt_and_med_lm_evaluation/`.
- Comparison tasks:
  - Run after baseline + refined outputs exist (or resubmit only compare indices later).
- Legacy scripts:
  - `gpt_qa_eval.py` and `gpt_free_text_eval.py` are not used by this orchestrator because they are hardcoded to non-portable input paths.

---

## Paper and citation

Dataset on Hugging Face: https://huggingface.co/datasets/ofir408/CupCase

Paper: https://ojs.aaai.org/index.php/AAAI/article/view/35050

```bibtex
@inproceedings{perets2025cupcase,
  title={CUPCase: Clinically Uncommon Patient Cases and Diagnoses Dataset},
  author={Perets, Oriel and Shoham, Ofir Ben and Grinberg, Nir and Rappoport, Nadav},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={27},
  pages={28293--28301},
  year={2025}
}
```
