# Eval Directory

Eval directory for the CUPCase dataset. 
This directory contains all the tools and scripts necessary to recreate the evaluation in the CUPCase paper.

## Installation

To get started, you'll need to install the required dependencies. Run the following command:

```bash
pip install -r requirements.txt
```

## Dataset

Please ensure that the CUPCase dataset is placed in the /datasets folder within this directory.

## API keys

To use gpt-4o (or any OpenAI model) for evaluation, please create and place your API_KEY in a .env file.
```bash
OPENAI_API_KEY="YOUR_API_KEY_HERE"
```

To use DeepSeek Chat as a judge model, add your DeepSeek API key:
```bash
DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY_HERE"
```
## Usage

To perform evaluation, run the specific evaluation script you wish to use. For example:

```bash
python gpt_free_text_eval.py
```

## Scripts

The scripts available in this directory are:

## GPT-4o
### Multiple-choice evaluation 

Used for evaluation GPT-4o with the multiple-choice QA in CUPCase.
Run 
```bash
gpt_qa_eval.py
```

### Multiple-choice evaluation (refined variants)

Used for running baseline, semantic-similarity-gated, discriminative-question, or progressive-disclosure MCQ evaluation with telemetry.
Run:
```bash
python gpt_qa_eval_refined.py --variant discriminative_question
```

### Open-ended evaluation

Used for evaluation GPT-4o with the open-ended QA in CUPCase.\
Run
```bash
gpt_free_text_eval.py
```
### Boostrap sampling calculate mean, std

Used for calculating the mean and standard deviation of the 4 bootstrap sampling iterations of 250 samples each. \
Run
```bash
bootstrap_sampling_mean_std.py
```

## MedLM-Large

To evaluate CUPCase using MedLM-Large, follow the instructions in the .ipynb.
```bash
medlm_inference.ipynb
```
MedLM-Large is a closed source model, requires specific access from Google to use.
And is most easily accessible through Google Colab.

---

## DeepSeek Chat

DeepSeek Chat can be used as an alternative judge model for evaluation. It uses an OpenAI-compatible API.

### Setup

Set your DeepSeek API key in the `.env` file:
```bash
DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY_HERE"
```

### Multiple-choice evaluation

Used for evaluating DeepSeek Chat with the multiple-choice QA in CUPCase.
```bash
python deepseek_qa_eval.py --input ablation_study_tokens.csv --output output/deepseek_multiple_choice.csv
```

### Open-ended evaluation

Used for evaluating DeepSeek Chat with the open-ended QA in CUPCase.
```bash
python deepseek_free_text_eval.py --input datasets/Case_report_w_images_dis_VF.csv --output output/deepseek_free_text.csv
```

### Command Line Options

Both scripts support the following options:
- `--input`: Path to input dataset CSV
- `--output`: Path to output CSV file
- `--model`: DeepSeek model to use (default: `deepseek-chat`)
- `--n-batches`: Number of batches to sample (default: 4)
- `--batch-size`: Size of each batch (default: 250)

### Using DeepSeek with Iterative Refinement

The refinement module also supports DeepSeek as a provider:

```python
from refinement import create_refiner, JudgeProvider
from refinement.refiner import RefinerConfig

config = RefinerConfig(
    generator_model="deepseek-chat",
    critic_model="deepseek-chat", 
    editor_model="deepseek-chat",
    provider=JudgeProvider.DEEPSEEK,
)

refiner = create_refiner(config=config)
```

---

## Iterative Adversarial Refinement with Checklist Enforcement

This module implements an iterative refinement pipeline that:
1. Generates an initial diagnostic response (Generator)
2. Evaluates it against an 8-item checklist + clinical quality (Critic)
3. Applies targeted, minimal edits for failed items (Editor)
4. Iterates until compliance or max iterations
5. Logs metrics: CCR_all, CCR_Q, CCR_H, iterations to compliance, minimality of edits

### Quick Start

Run the refined open-ended evaluation:

```bash
python gpt_free_text_eval_refined.py
```

This will:
- Process cases using the iterative refinement pipeline
- Save a CSV with BERTScore on `final_diagnosis` (compatible with baseline)
- Save a JSONL with full refinement traces and metrics

### Variant Framework

The refined evaluator now supports pluggable variants for idea-by-idea experiments.

Available variants:
- `baseline`: Original Generator -> Critic -> Editor loop
- `domain_routed`: Domain Routed Prompt Specialization (implemented idea #5)
- `semantic_similarity_gated`: Semantic Similarity Gated Differential Reasoning (implemented idea #2)
- `discriminative_question`: Self Generated Discriminative Questions with Answer Integration (implemented idea #4)
- `progressive_disclosure`: Progressive Disclosure with Explicit Belief Revision (implemented idea #3)

Run a specific variant:

```bash
python gpt_free_text_eval_refined.py --variant domain_routed
```

Or use the dedicated variant script:

```bash
python gpt_free_text_eval_refined_domain_routed.py
```

Semantic-similarity variant:

```bash
python gpt_free_text_eval_refined.py --variant semantic_similarity_gated
```

Or use the dedicated wrapper:

```bash
python gpt_free_text_eval_refined_semantic_similarity.py
```

Discriminative-question variant:

```bash
python gpt_free_text_eval_refined.py --variant discriminative_question
```

Or use the dedicated wrapper:

```bash
python gpt_free_text_eval_refined_discriminative_question.py
```

Progressive-disclosure variant:

```bash
python gpt_free_text_eval_refined.py --variant progressive_disclosure
```

Or use the dedicated wrapper:

```bash
python gpt_free_text_eval_refined_progressive_disclosure.py
```

### Command Line Options

```bash
python gpt_free_text_eval_refined.py \
    --input datasets/Case_report_w_images_dis_VF.csv \
    --output-dir output/refined \
    --variant baseline \
    --model gpt-4o \
    --max-iterations 3 \
    --clinical-threshold 3 \
    --similarity-threshold 0.65 \
    --disclosure-fraction 0.2 \
    --early-confidence-threshold 0.8 \
    --revision-instability-threshold 0.5 \
    --n-batches 4 \
    --batch-size 250
```

When `--variant` is not `baseline` and `--output-dir` is left as default,
the script automatically writes to `output/refined_<variant>`.

### Implemented Variant: Domain Routed Prompt Specialization

`domain_routed` adds a deterministic first-pass specialty classifier and routes
generation to domain-specific prompt templates.

Current specialty templates:
- `general_medicine`
- `oncology`
- `infectious_disease`
- `neurology`
- `cardiology`

The selected domain and routing scores are logged per case in trace metadata.

### Implemented Variant: Semantic Similarity Gated Differential Reasoning

`semantic_similarity_gated` runs an additional discriminator pass only when the
model's top-3 candidates are semantically clustered.

Flow:
- Candidate pass generates model-ranked top-3 diagnoses
- Pairwise cosine similarity is computed with JINA embeddings (`jinaai/jina-embeddings-v2-base-en`)
- If mean pairwise cosine is `>= 0.65`, discriminator reasoning is invoked
- Final response includes explicit differentiators in `conditional_reasoning`

Per-case telemetry (in `variant_metadata`) includes:
- candidate top-3 and confidences
- pairwise cosine matrix and mean cosine
- gate trigger flag
- discriminator rationale and differentiators

### Implemented Variant: Self Generated Discriminative Questions with Answer Integration

`discriminative_question` forces the model to actively resolve top-2 ambiguity:
- Generate ranked candidates
- Generate exactly one discriminative question targeting a concrete variable
- Re-scan the case and extract grounded answer + evidence snippets
- Integrate that answer explicitly into final diagnostic reasoning and choice

Per-case telemetry (in `variant_metadata`) includes:
- discriminative question and target variable
- extracted answer and confidence
- evidence spans used for re-integration
- integration summary and final selection source

### Implemented Variant: Progressive Disclosure with Explicit Belief Revision

`progressive_disclosure` enforces a two-stage reasoning pattern:
- Stage 1 uses only the first 20% of the case to produce an early differential.
- Stage 2 uses the full case and explicitly reports what changed in the differential.
- Belief-revision penalty signals are logged as telemetry (no forced decision override).

Per-case telemetry (in `variant_metadata`) includes:
- early top-3 and confidences
- revision kept/dropped/added hypotheses
- anchoring flag
- confidence instability score
- revision delta
- belief penalty score

### Compare Baseline vs Refined

After running both baseline and refined evaluations, compare them:

```bash
python compare_baseline_vs_refined.py \
    --baseline output/gpt4_free_text_batched.csv \
    --refined output/refined_domain_routed/gpt4_free_text_refined_*.csv \
    --refined-traces output/refined_domain_routed/refinement_traces_*.jsonl \
    --output output/comparison_report.json
```

This produces:
- JSON report with paired delta metrics and 95% CIs
- Text report with significance tests

### Configuring the Checklist

The checklist is defined in `refinement/checklist.yaml`:

```yaml
checklist_items:
  - id: "C1"
    name: "Primary Diagnosis"
    description: "Response includes a clear, specific primary diagnosis"
    when_required: "always"
    # ... more fields

ccr_groups:
  CCR_all:
    items: ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
  CCR_Q:
    items: ["C3", "C4", "C5"]  # Quality subset
  CCR_H:
    items: ["C6", "C7", "C8"]  # Health/safety subset
```

Modify this file to:
- Change checklist item definitions
- Adjust CCR group memberships
- Set clinical quality thresholds

### Response Contract

Generator/Editor outputs strict JSON with these fields:
- `final_diagnosis` (string, required) - scored with BERTScore
- `differential` (list, optional)
- `conditional_reasoning` (string, optional)
- `clarifying_questions` (list, optional)
- `red_flags` (list, optional)
- `uncertainty` (string, optional)
- `next_steps` (list, required)

### Metrics Computed

**CCR Metrics:**
- `CCR_all`: % cases compliant on ALL 8 items
- `CCR_Q`: % cases compliant on quality subset (clarifying questions, uncertainty, conditional reasoning)
- `CCR_H`: % cases compliant on health/safety subset (red flags, next steps, consistency)

**Minimality Metrics:**
- `edit_distance_total`: Sum of edit distances across iterations
- `edit_ratio_total`: Normalized edit ratio
- `word_changes_total`: Total word-level changes

**Iteration Metrics:**
- `iterations_to_compliance`: Number of Critic→Editor cycles needed
- `is_compliant`: Whether joint compliance was achieved

### Running Tests

```bash
cd gpt_and_med_lm_evaluation
pytest tests/test_refinement.py tests/test_variants.py tests/test_similarity_gating.py tests/test_discriminative_questioning.py tests/test_progressive_disclosure.py tests/test_domain_routed_wrapper.py tests/test_semantic_wrapper.py tests/test_discriminative_wrapper.py tests/test_progressive_wrapper.py tests/test_qa_refined.py -v
```

### Module Structure

```
refinement/
├── __init__.py          # Public API exports
├── refiner.py           # Main IterativeRefiner class
├── variant_factory.py   # Variant registry + factory
├── discriminative_questioning.py # Shared question-answer integration core
├── progressive_disclosure.py # Shared early/full-stage belief revision core
├── similarity_gating.py # Shared candidate similarity gate core
├── schema.py            # Data classes and JSON parsing
├── metrics.py           # CCR and minimality metrics
├── stats.py             # Paired bootstrap/permutation tests
├── io.py                # JSONL logging utilities
├── checklist.yaml       # Configurable checklist
├── variants/
│   ├── __init__.py
│   ├── domain_routed.py             # Domain routed variant implementation
│   ├── semantic_similarity_gated.py # Similarity-gated variant implementation
│   ├── discriminative_question.py   # Discriminative-question variant implementation
│   └── progressive_disclosure.py    # Progressive-disclosure variant implementation
└── prompts/
    ├── generator.md     # Generator prompt template
    ├── critic.md        # Critic prompt template
    ├── editor.md        # Editor prompt template
    ├── domain_routes/   # Domain-specific generator templates
    ├── discriminative_question/
    │   ├── candidate_free_text.md
    │   ├── question_free_text.md
    │   ├── answer_extraction_free_text.md
    │   └── integrate_free_text.md
    ├── progressive_disclosure/
    │   ├── early_free_text.md
    │   └── revision_free_text.md
    └── semantic_similarity/
        ├── candidate_free_text.md
        └── discriminator_free_text.md
```

### Output Files

After running `gpt_free_text_eval_refined.py`:

```
output/refined/
├── gpt4_free_text_refined_TIMESTAMP.csv    # BERTScore-compatible CSV
├── refinement_traces_TIMESTAMP.jsonl        # Full refinement traces
└── summary_report_TIMESTAMP.json            # Aggregate metrics
```

Each trace now includes:
- `variant_name`
- `variant_metadata` (for example routed domain scores)

After running `compare_baseline_vs_refined.py`:

```
output/
├── comparison_report.json    # Detailed comparison with CIs and p-values
└── comparison_report.txt     # Human-readable summary
```

## Using DiagnosisMedQA (Hugging Face)

Dataset:
- `oriel9p/DiagnosisMedQA`
- Split: `train` (829 rows)
- Columns: `id`, `clean_case_presentation`, `correct_diagnosis`, `distractor1`, `distractor2`, `distractor3`

Convert it to the evaluator CSV schema:

```bash
cd gpt_and_med_lm_evaluation
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

Run the domain-routed refinement variant:

```bash
python gpt_free_text_eval_refined.py \
  --variant domain_routed \
  --input datasets/DiagnosisMedQA_eval.csv \
  --output-dir output/refined_domain_routed_diagnosismedqa \
  --n-batches 1 \
  --batch-size 250
```

## MCQ Refined Evaluation

Use the new MCQ refined runner for baseline, similarity-gated, discriminative-question, or progressive-disclosure evaluation:

```bash
python gpt_qa_eval_refined.py \
  --variant discriminative_question \
  --input ablation_study_tokens.csv \
  --output output/gpt4_multiple_choice_refined.csv \
  --model gpt-4o \
  --n-batches 4 \
  --batch-size 250
```

Additional MCQ output columns:
- `Variant`
- `Gate Triggered`
- `Mean Cosine`
- `Pairwise Cosine JSON`
- `Candidate Top3`
- `Candidate Rationale`
- `Discriminative Question`
- `Question Target Variable`
- `Extracted Answer`
- `Answer Confidence`
- `Evidence Spans JSON`
- `Integration Summary`
- `Discriminator Rationale`
- `Differentiators JSON`
- `Early Candidate Top3`
- `Early Candidate Rationale`
- `Early Top Confidence`
- `Revision Summary`
- `Revision Evidence`
- `Anchoring Flag`
- `Confidence Instability Score`
- `Revision Delta`
- `Belief Penalty Score`
