"""MCQ evaluation with optional semantic similarity gated reasoning."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from refinement.refiner import JudgeProvider, create_client
from refinement.schema import extract_json_from_response
from refinement.similarity_gating import (
    JinaEmbeddingService,
    compute_similarity_for_top3,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VALID_VARIANTS = ["baseline", "semantic_similarity_gated"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCQ evaluation with optional similarity gating")
    parser.add_argument("--input", type=str, default="ablation_study_tokens.csv")
    parser.add_argument("--output", type=str, default="output/gpt4_multiple_choice_refined.csv")
    parser.add_argument("--variant", type=str, default="baseline", choices=VALID_VARIANTS)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "deepseek"])
    parser.add_argument("--n-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--similarity-threshold", type=float, default=0.65)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=60.0)
    parser.add_argument("--api-delay", type=float, default=1.0)
    return parser.parse_args()


def load_prompt_template(name: str) -> str:
    prompts_dir = Path(__file__).parent / "prompts" / "semantic_similarity"
    path = prompts_dir / f"{name}.md"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def call_model(
    client,
    model: str,
    prompt: str,
    retry_attempts: int,
    retry_delay: float,
    temperature: float = 0.0,
) -> str:
    last_error = None
    for attempt in range(retry_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # pragma: no cover - network call behavior
            last_error = exc
            logger.warning(
                "MCQ API call attempt %s/%s failed: %s",
                attempt + 1,
                retry_attempts,
                exc,
            )
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
    raise last_error


def parse_predicted_index(text: str, num_options: int) -> int:
    stripped = text.strip()
    if not stripped:
        return -1

    for char in stripped:
        if char.isdigit():
            idx = int(char) - 1
            if 0 <= idx < num_options:
                return idx
            return -1
    return -1


def parse_ranked_indices(text: str, num_options: int) -> Tuple[List[int], str]:
    data = extract_json_from_response(text)
    ranked = data.get("ranked_indices", data.get("candidate_indices", []))
    rationale = str(data.get("rationale", "")).strip()

    if not isinstance(ranked, list):
        raise ValueError("ranked_indices must be a list")

    parsed: List[int] = []
    for item in ranked:
        idx = int(item) - 1
        if idx < 0 or idx >= num_options:
            raise ValueError(f"ranked index out of range: {item}")
        if idx not in parsed:
            parsed.append(idx)

    if len(parsed) < 3:
        raise ValueError("Need at least 3 unique ranked indices")

    return parsed[:3], rationale


def parse_discriminator_choice(text: str, num_options: int) -> Tuple[int, List[str], str]:
    data = extract_json_from_response(text)

    final_choice = int(data.get("final_choice_index", -1)) - 1
    if final_choice < 0 or final_choice >= num_options:
        raise ValueError("Invalid discriminator final_choice_index")

    differentiators = data.get("differentiators", [])
    if not isinstance(differentiators, list):
        differentiators = [str(differentiators)]

    rationale = str(data.get("rationale", "")).strip()
    differentiators = [str(item).strip() for item in differentiators if str(item).strip()]
    return final_choice, differentiators, rationale


def extract_distractors(row: pd.Series, true_diagnosis: str) -> List[str]:
    distractors: List[str] = []
    for key in ("distractor1", "distractor2", "distractor3", "distractor4"):
        value = str(row.get(key, "")).strip()
        if value and value != true_diagnosis and value not in distractors:
            distractors.append(value)
    if len(distractors) < 3:
        raise ValueError("Need at least 3 distractors for MCQ setup")
    return distractors[:3]


def evaluate_case_baseline(
    client,
    model: str,
    case_text: str,
    options: Sequence[str],
    retry_attempts: int,
    retry_delay: float,
) -> Tuple[int, Dict[str, Any]]:
    options_text = "\n".join(f"{i+1}. {option}" for i, option in enumerate(options))
    prompt = (
        "Predict the diagnosis of this case presentation of a patient. "
        "Return only the correct index from the following list, for example: 3\n"
        f"{options_text}\nCase presentation: {case_text}"
    )

    raw = call_model(client, model, prompt, retry_attempts, retry_delay)
    predicted_index = parse_predicted_index(raw, len(options))
    telemetry = {
        "variant": "baseline",
        "gate_triggered": False,
        "candidate_rationale": "",
        "discriminator_rationale": "",
        "differentiators": [],
        "candidate_top3": [],
        "pairwise_cosine": {},
        "mean_cosine": None,
    }
    return predicted_index, telemetry


def evaluate_case_semantic_similarity(
    client,
    model: str,
    case_text: str,
    options: Sequence[str],
    similarity_threshold: float,
    embedding_service: JinaEmbeddingService,
    candidate_prompt_template: str,
    discriminator_prompt_template: str,
    retry_attempts: int,
    retry_delay: float,
) -> Tuple[int, Dict[str, Any]]:
    options_text = "\n".join(f"{i+1}. {option}" for i, option in enumerate(options))

    telemetry: Dict[str, Any] = {
        "variant": "semantic_similarity_gated",
        "gate_triggered": False,
        "candidate_rationale": "",
        "discriminator_rationale": "",
        "differentiators": [],
        "candidate_top3": [],
        "pairwise_cosine": {},
        "mean_cosine": None,
    }

    candidate_prompt = candidate_prompt_template.replace(
        "{options_text}", options_text
    ).replace(
        "{case_text}", case_text
    )

    try:
        raw_candidate = call_model(
            client,
            model,
            candidate_prompt,
            retry_attempts,
            retry_delay,
        )
        ranked_indices, candidate_rationale = parse_ranked_indices(
            raw_candidate,
            len(options),
        )
        telemetry["candidate_rationale"] = candidate_rationale
        telemetry["candidate_top3"] = ranked_indices
    except Exception as exc:
        telemetry["candidate_error"] = str(exc)
        return -1, telemetry

    top3_labels = [options[idx] for idx in ranked_indices[:3]]
    try:
        similarity = compute_similarity_for_top3(
            top3_labels,
            threshold=similarity_threshold,
            embedding_service=embedding_service,
        )
    except Exception as exc:
        telemetry["similarity_error"] = str(exc)
        return ranked_indices[0], telemetry

    telemetry["pairwise_cosine"] = similarity.pairwise_cosine
    telemetry["mean_cosine"] = similarity.mean_cosine
    telemetry["gate_triggered"] = similarity.gate_triggered

    if not similarity.gate_triggered:
        return ranked_indices[0], telemetry

    candidate_block = "\n".join(
        f"{position+1}. option {idx+1}: {options[idx]}"
        for position, idx in enumerate(ranked_indices[:3])
    )
    similarity_block = "\n".join(
        [f"Mean cosine similarity: {similarity.mean_cosine:.4f}"]
        + [f"pair {k}: {v:.4f}" for k, v in similarity.pairwise_cosine.items()]
    )

    discriminator_prompt = discriminator_prompt_template.replace(
        "{candidate_block}", candidate_block
    ).replace(
        "{similarity_block}", similarity_block
    ).replace(
        "{case_text}", case_text
    )

    try:
        raw_discriminator = call_model(
            client,
            model,
            discriminator_prompt,
            retry_attempts,
            retry_delay,
        )
        final_choice, differentiators, rationale = parse_discriminator_choice(
            raw_discriminator,
            len(options),
        )
        telemetry["discriminator_rationale"] = rationale
        telemetry["differentiators"] = differentiators
        return final_choice, telemetry
    except Exception as exc:
        telemetry["discriminator_error"] = str(exc)
        return ranked_indices[0], telemetry


def process_batch(
    batch: pd.DataFrame,
    args: argparse.Namespace,
    client,
    embedding_service: JinaEmbeddingService,
    candidate_prompt_template: str,
    discriminator_prompt_template: str,
    random_seed: int,
) -> List[Dict[str, Any]]:
    results = []
    rng = random.Random(random_seed)

    for _, row in batch.iterrows():
        case_text = str(row["case presentation"])
        true_diagnosis = str(row["final diagnosis"])

        try:
            distractors = extract_distractors(row, true_diagnosis)
        except Exception as exc:
            logger.warning("Skipping row due to distractor parsing error: %s", exc)
            continue

        options = [true_diagnosis] + distractors
        rng.shuffle(options)

        correct_index = options.index(true_diagnosis)

        if args.variant == "semantic_similarity_gated":
            predicted_index, telemetry = evaluate_case_semantic_similarity(
                client=client,
                model=args.model,
                case_text=case_text,
                options=options,
                similarity_threshold=args.similarity_threshold,
                embedding_service=embedding_service,
                candidate_prompt_template=candidate_prompt_template,
                discriminator_prompt_template=discriminator_prompt_template,
                retry_attempts=args.retry_attempts,
                retry_delay=args.retry_delay,
            )
        else:
            predicted_index, telemetry = evaluate_case_baseline(
                client=client,
                model=args.model,
                case_text=case_text,
                options=options,
                retry_attempts=args.retry_attempts,
                retry_delay=args.retry_delay,
            )

        generated_diagnosis = options[predicted_index] if 0 <= predicted_index < len(options) else ""

        results.append(
            {
                "Case presentation": case_text,
                "True diagnosis": true_diagnosis,
                "Generated diagnosis": generated_diagnosis,
                "Correct index": correct_index,
                "Predicted index": predicted_index,
                "Correct": correct_index == predicted_index,
                "Variant": telemetry.get("variant", args.variant),
                "Gate Triggered": telemetry.get("gate_triggered", False),
                "Mean Cosine": telemetry.get("mean_cosine"),
                "Pairwise Cosine JSON": json.dumps(telemetry.get("pairwise_cosine", {})),
                "Candidate Top3": json.dumps(telemetry.get("candidate_top3", [])),
                "Candidate Rationale": telemetry.get("candidate_rationale", ""),
                "Discriminator Rationale": telemetry.get("discriminator_rationale", ""),
                "Differentiators JSON": json.dumps(telemetry.get("differentiators", [])),
                "Error": telemetry.get("candidate_error")
                or telemetry.get("similarity_error")
                or telemetry.get("discriminator_error")
                or "",
            }
        )

        time.sleep(args.api_delay)

    return results


def main() -> None:
    args = parse_args()

    ds = pd.read_csv(args.input)
    provider = JudgeProvider(args.provider)
    client = create_client(provider=provider)
    embedding_service = JinaEmbeddingService()

    candidate_prompt_template = load_prompt_template("candidate_mcq")
    discriminator_prompt_template = load_prompt_template("discriminator_mcq")

    all_results: List[Dict[str, Any]] = []

    for batch_num in range(args.n_batches):
        logger.info("Processing batch %s/%s", batch_num + 1, args.n_batches)
        batch = ds.sample(n=args.batch_size, random_state=args.random_seed + batch_num)

        batch_results = process_batch(
            batch=batch,
            args=args,
            client=client,
            embedding_service=embedding_service,
            candidate_prompt_template=candidate_prompt_template,
            discriminator_prompt_template=discriminator_prompt_template,
            random_seed=args.random_seed + batch_num,
        )
        all_results.extend(batch_results)

        logger.info("Completed batch %s/%s", batch_num + 1, args.n_batches)
        if batch_num < args.n_batches - 1:
            time.sleep(10)

    results_df = pd.DataFrame(all_results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)

    accuracy = float(results_df["Correct"].mean()) if len(results_df) else 0.0
    print(f"Saved: {args.output}")
    print(f"Variant: {args.variant}")
    print(f"Rows: {len(results_df)}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
