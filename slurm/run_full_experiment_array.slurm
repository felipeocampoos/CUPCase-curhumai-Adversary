#!/bin/bash
#SBATCH --job-name=cupcase-exp
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --array=0-35

#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_DIR="$ROOT_DIR/gpt_and_med_lm_evaluation"

mkdir -p "$EVAL_DIR/output/experiments"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set. This script must run as a Slurm array job."
  exit 1
fi

TASK_ID="$SLURM_ARRAY_TASK_ID"

if [[ ! -d "$EVAL_DIR" ]]; then
  echo "Missing eval directory: $EVAL_DIR"
  exit 1
fi

if [[ ! -d "$EVAL_DIR/.venv312" ]]; then
  echo "Missing virtualenv: $EVAL_DIR/.venv312"
  echo "Create it first: cd gpt_and_med_lm_evaluation && python3.12 -m venv .venv312"
  exit 1
fi

source "$EVAL_DIR/.venv312/bin/activate"
cd "$EVAL_DIR"

FULL_DATASET="${FULL_DATASET:-datasets/DiagnosisMedQA_eval_20.csv}"
PARTIAL_DATASET="${PARTIAL_DATASET:-datasets/DiagnosisMedQA_eval_20_first10.csv}"

if [[ ! -f "$FULL_DATASET" ]]; then
  echo "Full dataset file not found: $FULL_DATASET"
  exit 1
fi

if [[ ! -f "$PARTIAL_DATASET" ]]; then
  echo "Partial dataset file not found: $PARTIAL_DATASET"
  exit 1
fi

# Build matrix entries.
# Arrays store per-index metadata used to dispatch commands.
declare -a JOB_KIND DATASET_MODE PROVIDER TASK_FAMILY VARIANT

idx=0

# 1) MCQ refined: 2 modes x 2 providers x 4 variants = 16 jobs
for mode in full partial; do
  for provider in openai deepseek; do
    for variant in baseline semantic_similarity_gated discriminative_question progressive_disclosure; do
      JOB_KIND[$idx]="run"
      DATASET_MODE[$idx]="$mode"
      PROVIDER[$idx]="$provider"
      TASK_FAMILY[$idx]="mcq"
      VARIANT[$idx]="$variant"
      idx=$((idx + 1))
    done
  done
done

# 2) Free-text refined (OpenAI): 2 modes x 5 variants = 10 jobs
for mode in full partial; do
  for variant in baseline domain_routed semantic_similarity_gated discriminative_question progressive_disclosure; do
    JOB_KIND[$idx]="run"
    DATASET_MODE[$idx]="$mode"
    PROVIDER[$idx]="openai"
    TASK_FAMILY[$idx]="free_refined"
    VARIANT[$idx]="$variant"
    idx=$((idx + 1))
  done
done

# 3) DeepSeek free-text baseline: 2 modes = 2 jobs
for mode in full partial; do
  JOB_KIND[$idx]="run"
  DATASET_MODE[$idx]="$mode"
  PROVIDER[$idx]="deepseek"
  TASK_FAMILY[$idx]="free_baseline"
  VARIANT[$idx]="baseline"
  idx=$((idx + 1))
done

# 4) Comparison jobs (same-provider refined baseline vs non-baseline refined): 2 modes x 4 variants = 8 jobs
for mode in full partial; do
  for variant in domain_routed semantic_similarity_gated discriminative_question progressive_disclosure; do
    JOB_KIND[$idx]="compare"
    DATASET_MODE[$idx]="$mode"
    PROVIDER[$idx]="openai"
    TASK_FAMILY[$idx]="free_text"
    VARIANT[$idx]="$variant"
    idx=$((idx + 1))
  done
done

TOTAL_JOBS="$idx"

if (( TASK_ID < 0 || TASK_ID >= TOTAL_JOBS )); then
  echo "Task ID $TASK_ID out of range [0, $((TOTAL_JOBS - 1))]"
  exit 1
fi

MODE="${DATASET_MODE[$TASK_ID]}"
PROV="${PROVIDER[$TASK_ID]}"
FAMILY="${TASK_FAMILY[$TASK_ID]}"
VAR="${VARIANT[$TASK_ID]}"
KIND="${JOB_KIND[$TASK_ID]}"

if [[ "$MODE" == "full" ]]; then
  INPUT_CSV="$FULL_DATASET"
  N_BATCHES="${N_BATCHES_FULL:-1}"
  BATCH_SIZE="${BATCH_SIZE_FULL:-20}"
else
  INPUT_CSV="$PARTIAL_DATASET"
  N_BATCHES="${N_BATCHES_PARTIAL:-1}"
  BATCH_SIZE="${BATCH_SIZE_PARTIAL:-10}"
fi

RUN_ROOT="output/experiments/$MODE/$PROV"
TS="$(date +%Y%m%d_%H%M%S)"

echo "------------------------------------------------------------"
echo "Task ID:           $TASK_ID / $((TOTAL_JOBS - 1))"
echo "Kind:              $KIND"
echo "Mode:              $MODE"
echo "Provider:          $PROV"
echo "Family:            $FAMILY"
echo "Variant:           $VAR"
echo "Input CSV:         $INPUT_CSV"
echo "n-batches:         $N_BATCHES"
echo "batch-size:        $BATCH_SIZE"
echo "Timestamp:         $TS"
echo "------------------------------------------------------------"

require_openai_key() {
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is required for this task but is not set."
    exit 1
  fi
}

require_deepseek_key() {
  if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "DEEPSEEK_API_KEY is required for this task but is not set."
    exit 1
  fi
}

latest_file() {
  local pattern="$1"
  ls -1t $pattern 2>/dev/null | head -n 1 || true
}

if [[ "$KIND" == "run" ]]; then
  if [[ "$FAMILY" == "mcq" ]]; then
    if [[ "$PROV" == "openai" ]]; then
      require_openai_key
    else
      require_deepseek_key
    fi

    OUT_DIR="$RUN_ROOT/mcq/$VAR"
    mkdir -p "$OUT_DIR"
    OUT_CSV="$OUT_DIR/results.csv"

    python gpt_qa_eval_refined.py \
      --input "$INPUT_CSV" \
      --output "$OUT_CSV" \
      --provider "$PROV" \
      --variant "$VAR" \
      --n-batches "$N_BATCHES" \
      --batch-size "$BATCH_SIZE"

    echo "Saved MCQ results: $OUT_CSV"
    exit 0
  fi

  if [[ "$FAMILY" == "free_refined" ]]; then
    require_openai_key

    OUT_DIR="$RUN_ROOT/free_text/$VAR"
    mkdir -p "$OUT_DIR"

    python gpt_free_text_eval_refined.py \
      --input "$INPUT_CSV" \
      --output-dir "$OUT_DIR" \
      --variant "$VAR" \
      --model "${OPENAI_MODEL:-gpt-4o}" \
      --n-batches "$N_BATCHES" \
      --batch-size "$BATCH_SIZE"

    echo "Saved free-text refined outputs in: $OUT_DIR"
    exit 0
  fi

  if [[ "$FAMILY" == "free_baseline" ]]; then
    require_deepseek_key

    OUT_DIR="$RUN_ROOT/free_text/baseline"
    mkdir -p "$OUT_DIR"
    OUT_CSV="$OUT_DIR/results.csv"

    python deepseek_free_text_eval.py \
      --input "$INPUT_CSV" \
      --output "$OUT_CSV" \
      --n-batches "$N_BATCHES" \
      --batch-size "$BATCH_SIZE"

    echo "Saved free-text baseline results: $OUT_CSV"
    exit 0
  fi

  echo "Unknown run family: $FAMILY"
  exit 1
fi

if [[ "$KIND" == "compare" ]]; then
  BASE_DIR="$RUN_ROOT/free_text/baseline"
  REFINED_DIR="$RUN_ROOT/free_text/$VAR"
  CMP_DIR="$RUN_ROOT/comparisons/$VAR"
  mkdir -p "$CMP_DIR"

  BASELINE_CSV="$(latest_file "$BASE_DIR/gpt4_free_text_refined_*.csv")"
  REFINED_CSV="$(latest_file "$REFINED_DIR/gpt4_free_text_refined_*.csv")"
  REFINED_TRACES="$(latest_file "$REFINED_DIR/refinement_traces_*.jsonl")"

  if [[ -z "$BASELINE_CSV" || -z "$REFINED_CSV" || -z "$REFINED_TRACES" ]]; then
    echo "Comparison prerequisites missing."
    echo "Expected files:"
    echo "  baseline csv: $BASE_DIR/gpt4_free_text_refined_*.csv"
    echo "  refined csv:  $REFINED_DIR/gpt4_free_text_refined_*.csv"
    echo "  refined jsonl:$REFINED_DIR/refinement_traces_*.jsonl"
    exit 1
  fi

  OUT_JSON="$CMP_DIR/comparison_report.json"

  python compare_baseline_vs_refined.py \
    --baseline "$BASELINE_CSV" \
    --refined "$REFINED_CSV" \
    --refined-traces "$REFINED_TRACES" \
    --output "$OUT_JSON"

  echo "Saved comparison report: $OUT_JSON"
  exit 0
fi

echo "Unknown job kind: $KIND"
exit 1
