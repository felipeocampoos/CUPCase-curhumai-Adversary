How To Run?

Install the required dependencies:

```
git clone https://github.com/nadavlab/CUPCase
cd lm_eval_evaluation
pip install -e .
export DATA_SEED=42
```

Run the benchmark evaluation:

`lm_eval --model hf --model_args pretrained=MODEL_ID --tasks cupcase_qa,cupcase_generation --device cuda:0  --batch_size auto`

Replace `MODEL_ID` with the model name (HuggingFace) or local path to the pretrained model you want to evaluate.

For example: 

`lm_eval --model hf --model_args pretrained=BioMistral/BioMistral-7B-DARE --tasks cupcase_qa,cupcase_generation --device cuda:0  --batch_size auto`

## MedConceptsQA

This repo also includes a repo-owned runner for the native `med_concepts_qa`
task family backed by `ofir408/MedConceptsQA`.

Supported `--subset` values:

- `all`
- vocab-only: `icd9cm`, `icd10cm`, `icd9proc`, `icd10proc`, `atc`
- vocab + difficulty: `<vocab>_easy`, `<vocab>_medium`, `<vocab>_hard`

Examples:

- `icd10cm_easy`
- `icd10cm_medium`
- `icd10cm_hard`
- `atc_hard`

`--sample-size` is passed through to `lm_eval` as the harness document limit.
Use it for smoke runs and debugging, not for final reported metrics.

Run the full benchmark group:

```bash
python scripts/run_medconceptsqa.py \
  --model hf \
  --model-args pretrained=BioMistral/BioMistral-7B-DARE \
  --subset all \
  --device cuda:0 \
  --batch-size auto
```

Limit to a fixed number of evaluation samples:

```bash
python scripts/run_medconceptsqa.py \
  --model hf \
  --model-args pretrained=BioMistral/BioMistral-7B-DARE \
  --subset icd10cm_easy \
  --sample-size 100 \
  --device cuda:0 \
  --batch-size auto \
  --log-samples
```

Difficulty-specific harness examples:

```bash
python scripts/run_medconceptsqa.py \
  --model hf \
  --model-args pretrained=BioMistral/BioMistral-7B-DARE \
  --subset icd10cm_medium \
  --sample-size 100 \
  --device cuda:0 \
  --batch-size auto \
  --log-samples

python scripts/run_medconceptsqa.py \
  --model hf \
  --model-args pretrained=BioMistral/BioMistral-7B-DARE \
  --subset atc_hard \
  --sample-size 50 \
  --device cuda:0 \
  --batch-size auto \
  --log-samples
```

Smoke test the integration:

```bash
python scripts/medconceptsqa_smoke.py --keep-output
```

Outputs are written under:

`output/medconceptsqa/<subset>/<model_slug>/<timestamp>/`
