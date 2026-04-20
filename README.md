## CUPCase: End-to-End Experiment Runbook

This repository contains CUPCase evaluation and preprocessing code. This runbook is the canonical way to run the paper-core experiment matrix end-to-end and compare methods later.

Focused MedConceptsQA commands are collected in:

`MEDCONCEPTSQA_QUICKSTART.md`

Focused MedCalc-Bench commands are collected in:

`MEDCALC_BENCH_QUICKSTART.md`

### What this runbook covers

- API-based MCQ and free-text evaluation in `gpt_and_med_lm_evaluation/`
- Refined variants (baseline + all implemented methods)
- Full vs partial dataset modes
- One-command Slurm submission for the full matrix
- Comparison report generation for baseline vs refined

### What this runbook does **not** cover

- Full preprocessing/regeneration pipeline in `preprocess/` and `utils/`
- General on-prem model benchmarking in `lm_eval_evaluation/`, except for the
  repo-owned `MedConceptsQA` runner and smoke path documented below.

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
OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
OPENAI_COMPATIBLE_API_KEY="dummy"
OPENAI_COMPATIBLE_MODEL="Qwen/Qwen3.5-0.8B"
HUGGINGFACE_LOCAL_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
HF_TOKEN=""
```

Key usage:
- OpenAI runs require `OPENAI_API_KEY`
- DeepSeek runs require `DEEPSEEK_API_KEY`
- OpenAI-compatible local or cluster Qwen runs use `OPENAI_COMPATIBLE_BASE_URL` and `OPENAI_COMPATIBLE_MODEL`
- Native local Hugging Face runs use `HUGGINGFACE_LOCAL_MODEL` and optionally `HF_TOKEN`

Preflight smoke checks:

```bash
cd gpt_and_med_lm_evaluation
python openai_compatible_smoke.py --task mcq
python openai_compatible_smoke.py --task free_text --variant baseline
python huggingface_local_smoke.py --task mcq
python huggingface_local_smoke.py --task free_text --variant baseline
```

HF local comparison bundle:

```bash
python hf_local_analysis.py \
  --baseline-variant baseline \
  --refined-variant domain_routed \
  --input datasets/CUPCASE_RTEST_eval_20.csv \
  --output-dir output/hf_local_analysis
```

This writes one analysis bundle containing:
- baseline/refined run directories,
- `run_manifest_*.json` provenance for each run,
- `compare/comparison_report.json` and `.txt`,
- `analysis_summary.md` with what changed, what stayed constant, and what evidence is still missing.

Additional MedConceptsQA smoke checks:

```bash
cd lm_eval_evaluation
python scripts/medconceptsqa_smoke.py --keep-output

cd ../gpt_and_med_lm_evaluation
python medconceptsqa_smoke.py --keep-output
```

Additional MedCalc-Bench smoke checks:

```bash
cd lm_eval_evaluation
python scripts/medcalc_bench_smoke.py --keep-output

cd ../gpt_and_med_lm_evaluation
python medcalc_bench_smoke.py --keep-output
```

---

## 3) Dataset modes used by this runbook

This runbook uses two dataset presets:

- **easy** (default): DiagnosisMedQA sample
  - `full`: `gpt_and_med_lm_evaluation/datasets/DiagnosisMedQA_eval_20.csv`
  - `partial`: `gpt_and_med_lm_evaluation/datasets/DiagnosisMedQA_eval_20_first10.csv`
- **hard**: CUPCASE_RTEST sample
  - `full`: `gpt_and_med_lm_evaluation/datasets/CUPCASE_RTEST_eval.csv`
  - `sample`: `gpt_and_med_lm_evaluation/datasets/CUPCASE_RTEST_eval_20.csv`

If needed, you can regenerate compatible files with:

```bash
cd gpt_and_med_lm_evaluation
source .venv312/bin/activate
python prepare_hf_diagnosismedqa.py --output datasets/DiagnosisMedQA_eval_custom.csv
python prepare_hf_diagnosismedqa.py --sample-size 10 --output datasets/DiagnosisMedQA_eval_custom_first10.csv

### 3.2 CUPCASE_RTEST (hard preset)

Upstream: `oriel9p/CUPCASE_RTEST` (HF dataset). Requires `HF_TOKEN`.

Regenerate full and sample CSVs:

```
python prepare_cupcase_rtest.py --hf-token $HF_TOKEN \
  --output datasets/CUPCASE_RTEST_eval.csv

python prepare_cupcase_rtest.py --hf-token $HF_TOKEN \
  --sample-size 20 --seed 42 \
  --output datasets/CUPCASE_RTEST_eval_20.csv
```
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

### 3.3 MedConceptsQA

Upstream: `ofir408/MedConceptsQA`

Supported `--subset` values:

- `all`: the full benchmark group
- vocab-only: `icd9cm`, `icd10cm`, `icd9proc`, `icd10proc`, `atc`
- vocab + difficulty: `<vocab>_easy`, `<vocab>_medium`, `<vocab>_hard`

Examples:

- `icd10cm_easy`: easy ICD-10-CM questions only
- `atc_hard`: hard ATC drug-code questions only
- `icd9proc_medium`: medium ICD-9 procedure questions only

Sample-count control:

- `lm_eval` runner: `--sample-size` is forwarded to the harness as a fixed-document limit for testing/debugging.
- CUPCase MCQ runner: `--sample-size` controls how many HF rows are materialized into the generated evaluator CSV.

Repo-owned `lm_eval` runner:

```bash
cd lm_eval_evaluation
python scripts/run_medconceptsqa.py \
  --model hf \
  --model-args pretrained=BioMistral/BioMistral-7B-DARE \
  --subset all \
  --sample-size 100 \
  --device cuda:0 \
  --log-samples
```

Difficulty-specific `lm_eval` example:

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

Repo-owned CUPCase MCQ runner:

```bash
cd ../gpt_and_med_lm_evaluation
python run_medconceptsqa_mcq.py \
  --provider huggingface_local \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --subset icd10cm_easy \
  --sample-size 25 \
  --variant baseline
```

Difficulty-specific CUPCase MCQ example:

```bash
python run_medconceptsqa_mcq.py \
  --provider huggingface_local \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --subset icd9proc_medium \
  --sample-size 10 \
  --variant baseline
```

Generated evaluator-compatible CSVs are written under:

`gpt_and_med_lm_evaluation/datasets/generated/medconceptsqa/`

`lm_eval` outputs are written under:

`lm_eval_evaluation/output/medconceptsqa/<subset>/<model_slug>/<timestamp>/`

CUPCase MCQ outputs are written under:

`gpt_and_med_lm_evaluation/output/experiments/medconceptsqa/<subset>/<provider>/mcq/<variant>/<model_slug>/<sample_label>/`

### 3.4 MedCalc-Bench-Verified

Upstream: `nsk7153/MedCalc-Bench-Verified`

This benchmark is integrated on both repo-owned experiment surfaces:

- `lm_eval_evaluation/` as a generative task with structured scoring
- `gpt_and_med_lm_evaluation/` as a free-text evaluator with numeric tolerance scoring

Prepare a normalized local CSV:

```bash
cd gpt_and_med_lm_evaluation
python prepare_hf_medcalc_bench.py \
  --split test \
  --sample-size 25
```

Generated CSVs are written under:

`gpt_and_med_lm_evaluation/datasets/generated/medcalc_bench/`

Run the API-based evaluator:

```bash
python run_medcalc_bench_free_text.py \
  --provider huggingface_local \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --method medcalc_semantic_gate \
  --split test \
  --sample-size 25
```

Available API-side MedCalc methods:

- `direct`
- `zero_shot_cot`
- `one_shot_cot`
- `medcalc_semantic_gate`

API-based evaluator artifacts are written under:

`gpt_and_med_lm_evaluation/output/experiments/medcalc_bench/<split>/<provider>/free_text/<method>/<model_slug>/<sample_label>/`

That directory contains:

- `results.csv` for per-example predictions and scoring
- `summary_report_<timestamp>.json` for aggregate metrics
- `run_manifest_<timestamp>.json` for run provenance

`one_shot_cot` supports deterministic example selection through:

- `--icl-example-id <id>`
- `--icl-seed <seed>`

Run the `lm_eval` wrapper:

```bash
cd ../lm_eval_evaluation
python scripts/run_medcalc_bench.py \
  --model hf \
  --model-args pretrained=BioMistral/BioMistral-7B-DARE \
  --method zero_shot_cot \
  --split test \
  --sample-size 50 \
  --device cuda:0 \
  --log-samples
```

`lm_eval` outputs are written under:

`lm_eval_evaluation/output/medcalc_bench/<split>/<method>/<model_slug>/<timestamp>/`

For `direct`, `zero_shot_cot`, and `one_shot_cot`, the actual aggregate metrics file is written under the nested harness model directory inside that timestamped folder, for example:

`lm_eval_evaluation/output/medcalc_bench/test/direct/hf_pretrained_sshleifer_tiny-gpt2/<timestamp>/sshleifer__tiny-gpt2/results_<timestamp>.json`

For `medcalc_semantic_gate`, the wrapper delegates to the repo-owned MedCalc local-model runner, so artifacts are nested under the wrapper run root, for example:

`lm_eval_evaluation/output/medcalc_bench/test/medcalc_semantic_gate/hf_pretrained_sshleifer_tiny-gpt2/<timestamp>/test/huggingface_local/free_text/medcalc_semantic_gate/sshleifer_tiny-gpt2/n1_seed42/results.csv`

That `lm_eval` run also writes:

- `results_<timestamp>.json` for aggregate metrics
- `samples_med_calc_bench_<timestamp>.jsonl` when `--log-samples` is enabled

Tested smoke commands:

```bash
cd gpt_and_med_lm_evaluation
.venv312/bin/python medcalc_bench_smoke.py --method medcalc_semantic_gate --keep-output

cd ../lm_eval_evaluation
../gpt_and_med_lm_evaluation/.venv312/bin/python scripts/medcalc_bench_smoke.py --method one_shot_cot --keep-output
```

For portability, the `lm_eval` smoke path is configured to use `sshleifer/tiny-gpt2` on `cpu`.

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
- `openai_compatible`
- `huggingface_local`

Variants:
- `baseline`
- `semantic_similarity_gated`
- `discriminative_question`
- `progressive_disclosure`

### Free-text refined runner
Script: `gpt_and_med_lm_evaluation/gpt_free_text_eval_refined.py`

Providers:
- `openai`
- `deepseek`
- `openai_compatible`
- `huggingface_local`

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
- `.../hf_local_analysis/<timestamp>/runs/baseline/run_manifest_*.json`

---

## 6) Local one-off commands (without Slurm)

From `gpt_and_med_lm_evaluation/` with `.venv312` active.

Dataset presets:
- `--dataset easy` (default) → `datasets/DiagnosisMedQA_eval_20.csv`
- `--dataset hard` → `datasets/CUPCASE_RTEST_eval.csv`
You can still pass `--input` for custom files; `--dataset custom` requires `--input`.

### 6.1 MCQ refined (OpenAI example)

```bash
python gpt_qa_eval_refined.py \
  --dataset easy \
  --output output/experiments/full/openai/mcq/discriminative_question/results.csv \
  --provider openai \
  --variant discriminative_question \
  --n-batches 1 \
  --batch-size 20
```

### 6.2 MCQ refined (DeepSeek example)

```bash
python gpt_qa_eval_refined.py \
  --dataset easy \
  --output output/experiments/full/deepseek/mcq/semantic_similarity_gated/results.csv \
  --provider deepseek \
  --variant semantic_similarity_gated \
  --n-batches 1 \
  --batch-size 20
```

### 6.3 MCQ refined (OpenAI-compatible Qwen example)

```bash
cd gpt_and_med_lm_evaluation
python openai_compatible_smoke.py --task mcq

python gpt_qa_eval_refined.py \
  --dataset hard \
  --output output/experiments/full/openai_compatible/mcq/baseline/results.csv \
  --provider openai_compatible \
  --model Qwen/Qwen3.5-0.8B \
  --variant baseline \
  --n-batches 1 \
  --batch-size 20
```

### 6.4 MCQ refined (local Hugging Face Qwen example)

```bash
cd gpt_and_med_lm_evaluation
python huggingface_local_smoke.py --task mcq

python gpt_qa_eval_refined.py \
  --dataset easy \
  --output output/experiments/full/huggingface_local/mcq/baseline/results.csv \
  --provider huggingface_local \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --variant baseline \
  --n-batches 1 \
  --batch-size 1
```

### 6.5 Free-text refined (OpenAI example)

```bash
python gpt_free_text_eval_refined.py \
  --dataset easy \
  --output-dir output/experiments/full/openai/free_text/domain_routed \
  --provider openai \
  --variant domain_routed \
  --model gpt-4o \
  --n-batches 1 \
  --batch-size 20
```

### 6.6 Free-text refined (OpenAI-compatible Qwen example)

```bash
cd gpt_and_med_lm_evaluation
python openai_compatible_smoke.py --task free_text --variant baseline

python gpt_free_text_eval_refined.py \
  --dataset hard \
  --output-dir output/experiments/full/openai_compatible/free_text/baseline \
  --provider openai_compatible \
  --variant baseline \
  --model Qwen/Qwen3.5-0.8B \
  --n-batches 1 \
  --batch-size 20
```

### 6.7 Free-text refined (local Hugging Face Qwen example)

```bash
cd gpt_and_med_lm_evaluation
python huggingface_local_smoke.py --task free_text --variant baseline

python gpt_free_text_eval_refined.py \
  --dataset hard \
  --output-dir output/experiments/full/huggingface_local/free_text/baseline \
  --provider huggingface_local \
  --variant baseline \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --n-batches 1 \
  --batch-size 1
```

### 6.8 DeepSeek free-text baseline

```bash
python deepseek_free_text_eval.py \
  --input datasets/DiagnosisMedQA_eval_20.csv \
  --output output/experiments/full/deepseek/free_text/baseline/results.csv \
  --n-batches 1 \
  --batch-size 20
```

### 6.9 Compare baseline vs refined

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
