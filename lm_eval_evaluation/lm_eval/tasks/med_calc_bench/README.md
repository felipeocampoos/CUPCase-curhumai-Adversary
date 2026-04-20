# MedCalc-Bench-Verified

### Dataset

Hugging Face: `nsk7153/MedCalc-Bench-Verified`

MedCalc-Bench-Verified is a medical calculation and structured QA benchmark built from patient notes. Each example includes a patient note, a calculator-style question, a ground-truth answer, optional lower and upper tolerance bounds, and an explanation.

This repo-owned task evaluates models in generative mode and scores correctness with structured answer parsing:

- numeric answers are scored against the provided tolerance interval when available
- otherwise numeric answers fall back to canonical exact-match
- date answers use normalized date equality
- other answers use normalized text equality

### Task

- `med_calc_bench` - the full benchmark on the dataset `test` split
- `med_calc_bench_train` - the same prompt/scoring on the dataset `train` split
- `med_calc_bench_one_shot` - the same prompt/scoring on the dataset `one_shot` split
- `med_calc_bench_zero_shot_cot` - zero-shot chain-of-thought prompting on the dataset `test` split
- `med_calc_bench_one_shot_cot` - one worked-example chain-of-thought prompting on the dataset `test` split

The repo-owned wrapper `scripts/run_medcalc_bench.py` exposes four method names:

- `direct`
- `zero_shot_cot`
- `one_shot_cot`
- `medcalc_semantic_gate`

`medcalc_semantic_gate` is implemented as a custom runner branch rather than a single-pass `lm_eval` YAML task because it requires multi-candidate generation and adjudication.
