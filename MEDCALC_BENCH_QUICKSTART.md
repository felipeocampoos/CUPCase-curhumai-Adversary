# MedCalc-Bench-Verified Quickstart

This repo supports `nsk7153/MedCalc-Bench-Verified` on two surfaces:

- `lm_eval_evaluation/`: repo-owned `lm_eval` task runner
- `gpt_and_med_lm_evaluation/`: API-based free-text evaluator path

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

## 1) Prepare a Local CSV

Materialize the Hugging Face dataset into the normalized local CSV used by the API-based runner:

```bash
python prepare_hf_medcalc_bench.py \
  --split test \
  --sample-size 25
```

Generated CSVs land under:

```text
gpt_and_med_lm_evaluation/datasets/generated/medcalc_bench/
```

## 2) API-Based Evaluator

Run the repo-owned MedCalc free-text evaluator:

```bash
python run_medcalc_bench_free_text.py \
  --provider huggingface_local \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --method medcalc_uncertainty_consistency_gate \
  --split test \
  --sample-size 25
```

Supported API-side methods:

- `direct`
- `zero_shot_cot`
- `one_shot_cot`
- `medcalc_semantic_gate`
- `medcalc_uncertainty_consistency_gate`

Artifacts land under:

```text
gpt_and_med_lm_evaluation/output/experiments/medcalc_bench/<split>/<provider>/free_text/<method>/<model_slug>/<sample_label>/
```

Expected files in that directory:

- `results.csv`: per-example outputs and scoring columns
- `summary_report_<timestamp>.json`: aggregate metrics such as accuracy and parse rate
- `run_manifest_<timestamp>.json`: provenance and config for the run

`one_shot_cot` supports:

- `--icl-example-id <id>` to pin a demonstration
- `--icl-seed <seed>` for deterministic selection from the dataset `one_shot` split

`medcalc_uncertainty_consistency_gate` also supports:

- `--num-candidates <n>` to control the number of sampled candidate answers
- `--candidate-temperature <float>` to control candidate diversity
- `--verifier-risk-threshold <float>` to tune when verifier output counts as risk

Smoke test:

```bash
python medcalc_bench_smoke.py --method medcalc_uncertainty_consistency_gate --keep-output
```

## 3) `lm_eval` Harness

From repo root:

```bash
cd lm_eval_evaluation
```

Run the repo-owned task:

```bash
python scripts/run_medcalc_bench.py \
  --model hf \
  --model-args pretrained=BioMistral/BioMistral-7B-DARE \
  --method zero_shot_cot \
  --split test \
  --sample-size 50 \
  --device cuda:0 \
  --log-samples
```

Outputs land under:

```text
lm_eval_evaluation/output/medcalc_bench/<split>/<method>/<model_slug>/<timestamp>/
```

For `direct`, `zero_shot_cot`, and `one_shot_cot`, the harness writes the aggregate metrics JSON one level below that timestamped directory, under the internal model slug directory, for example:

```text
lm_eval_evaluation/output/medcalc_bench/test/direct/hf_pretrained_sshleifer_tiny-gpt2/<timestamp>/sshleifer__tiny-gpt2/results_<timestamp>.json
```

Expected files from `lm_eval`:

- `results_<timestamp>.json`: aggregate benchmark metrics
- `samples_med_calc_bench_<timestamp>.jsonl`: per-sample generations when `--log-samples` is enabled

For `medcalc_semantic_gate` and `medcalc_uncertainty_consistency_gate`, the wrapper uses the repo-owned MedCalc local-model runner because these methods need multi-candidate generation and escalation logic beyond a single-pass `lm_eval` YAML task. Those artifacts are written under the wrapper run root, for example:

```text
lm_eval_evaluation/output/medcalc_bench/test/medcalc_semantic_gate/hf_pretrained_sshleifer_tiny-gpt2/<timestamp>/test/huggingface_local/free_text/medcalc_semantic_gate/sshleifer_tiny-gpt2/n1_seed42/results.csv
```

or

```text
lm_eval_evaluation/output/medcalc_bench/test/medcalc_uncertainty_consistency_gate/hf_pretrained_sshleifer_tiny-gpt2/<timestamp>/test/huggingface_local/free_text/medcalc_uncertainty_consistency_gate/sshleifer_tiny-gpt2/n1_seed42/results.csv
```

Smoke test:

```bash
python scripts/medcalc_bench_smoke.py --method one_shot_cot --keep-output
```

Notes:

- The repo-tested smoke command uses `sshleifer/tiny-gpt2` on `cpu` for portability.
- The `dummy` model is not suitable for this task because `med_calc_bench` uses `generate_until`.
- The gated methods currently require `--model hf --model-args pretrained=<hf-model>` on this local-model surface.
