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
