"""MCQ evaluation with baseline and refinement variants (semantic, discriminative, progressive)."""

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
from refinement.discriminative_questioning import (
    format_evidence_for_prompt,
    parse_answer_extraction,
    parse_discriminative_question,
    parse_integrated_decision_mcq,
    parse_ranked_option_indices,
)
from refinement.similarity_gating import (
    JinaEmbeddingService,
    compute_similarity_for_top3,
)
from refinement.progressive_disclosure import (
    compute_belief_revision_scores,
    format_early_options_for_prompt,
    parse_early_ranked_option_indices_mcq,
    parse_revision_decision_mcq,
    truncate_case_by_fraction,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VALID_VARIANTS = [
    "baseline",
    "semantic_similarity_gated",
    "discriminative_question",
    "progressive_disclosure",
]


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
    parser.add_argument("--disclosure-fraction", type=float, default=0.2)
    parser.add_argument("--early-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--revision-instability-threshold", type=float, default=0.5)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=60.0)
    parser.add_argument("--api-delay", type=float, default=1.0)
    return parser.parse_args()


def load_prompt_template(folder: str, name: str) -> str:
    prompts_dir = Path(__file__).parent / "prompts" / folder
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


def parse_discriminator_choice(
    text: str,
    num_options: int,
    candidate_indices: Sequence[int] | None = None,
) -> Tuple[int, List[str], str]:
    data = extract_json_from_response(text)

    raw_choice = int(data.get("final_choice_index", -1)) - 1
    if candidate_indices:
        # Prefer rank-relative decoding because prompt lists a ranked block.
        if 0 <= raw_choice < len(candidate_indices):
            final_choice = int(candidate_indices[raw_choice])
        elif 0 <= raw_choice < num_options:
            final_choice = raw_choice
        else:
            raise ValueError("Invalid discriminator final_choice_index")
    else:
        final_choice = raw_choice
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
    true_normalized = true_diagnosis.strip().lower()
    for key in ("distractor1", "distractor2", "distractor3", "distractor4"):
        raw_value = row.get(key, "")
        if pd.isna(raw_value):
            continue
        value = str(raw_value).strip()
        if not value:
            continue
        normalized = value.lower()
        if normalized in {"nan", "none", "null"}:
            continue
        if normalized != true_normalized and value not in distractors:
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
            ranked_indices[:3],
        )
        telemetry["discriminator_rationale"] = rationale
        telemetry["differentiators"] = differentiators
        return final_choice, telemetry
    except Exception as exc:
        telemetry["discriminator_error"] = str(exc)
        return ranked_indices[0], telemetry


def evaluate_case_discriminative_question(
    client,
    model: str,
    case_text: str,
    options: Sequence[str],
    candidate_prompt_template: str,
    question_prompt_template: str,
    answer_prompt_template: str,
    integration_prompt_template: str,
    retry_attempts: int,
    retry_delay: float,
) -> Tuple[int, Dict[str, Any]]:
    options_text = "\n".join(f"{i+1}. {option}" for i, option in enumerate(options))

    telemetry: Dict[str, Any] = {
        "variant": "discriminative_question",
        "candidate_top3": [],
        "candidate_rationale": "",
        "discriminative_question": "",
        "question_target_variable": "",
        "extracted_answer": "",
        "answer_confidence": None,
        "evidence_spans": [],
        "integration_summary": "",
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
        ranked_indices, candidate_rationale = parse_ranked_option_indices(
            raw_candidate,
            len(options),
        )
        telemetry["candidate_top3"] = ranked_indices[:3]
        telemetry["candidate_rationale"] = candidate_rationale
    except Exception as exc:
        telemetry["candidate_error"] = str(exc)
        return -1, telemetry

    top2 = ranked_indices[:2]
    candidate_block = "\n".join(
        f"{rank+1}. option {idx+1}: {options[idx]}"
        for rank, idx in enumerate(top2)
    )

    question_prompt = question_prompt_template.replace(
        "{candidate_block}", candidate_block
    ).replace(
        "{case_text}", case_text
    )
    try:
        raw_question = call_model(
            client,
            model,
            question_prompt,
            retry_attempts,
            retry_delay,
        )
        question = parse_discriminative_question(raw_question)
        telemetry["discriminative_question"] = question.question
        telemetry["question_target_variable"] = question.target_variable
    except Exception as exc:
        telemetry["question_error"] = str(exc)
        return ranked_indices[0], telemetry

    answer_prompt = answer_prompt_template.replace(
        "{question}", question.question
    ).replace(
        "{target_variable}", question.target_variable
    ).replace(
        "{case_text}", case_text
    )
    try:
        raw_answer = call_model(
            client,
            model,
            answer_prompt,
            retry_attempts,
            retry_delay,
        )
        answer = parse_answer_extraction(raw_answer)
        telemetry["extracted_answer"] = answer.answer
        telemetry["answer_confidence"] = answer.confidence
        telemetry["evidence_spans"] = answer.evidence_spans
    except Exception as exc:
        telemetry["answer_error"] = str(exc)
        return ranked_indices[0], telemetry

    integration_prompt = integration_prompt_template.replace(
        "{candidate_block}",
        "\n".join(
            f"{rank+1}. option {idx+1}: {options[idx]}"
            for rank, idx in enumerate(ranked_indices[:3])
        ),
    ).replace(
        "{question}", question.question
    ).replace(
        "{answer_block}", format_evidence_for_prompt(answer)
    ).replace(
        "{case_text}", case_text
    )
    try:
        raw_integration = call_model(
            client,
            model,
            integration_prompt,
            retry_attempts,
            retry_delay,
        )
        final_choice, integration_summary, rationale = parse_integrated_decision_mcq(
            raw_integration,
            len(options),
            ranked_indices[:3],
        )
        telemetry["integration_summary"] = integration_summary
        telemetry["discriminator_rationale"] = rationale
        return final_choice, telemetry
    except Exception as exc:
        telemetry["integration_error"] = str(exc)
        return ranked_indices[0], telemetry


def evaluate_case_progressive_disclosure(
    client,
    model: str,
    case_text: str,
    options: Sequence[str],
    early_prompt_template: str,
    revision_prompt_template: str,
    disclosure_fraction: float,
    early_conf_threshold: float,
    revision_instability_threshold: float,
    retry_attempts: int,
    retry_delay: float,
) -> Tuple[int, Dict[str, Any]]:
    options_text = "\n".join(f"{i+1}. {option}" for i, option in enumerate(options))
    early_case_text = truncate_case_by_fraction(case_text, fraction=disclosure_fraction)

    telemetry: Dict[str, Any] = {
        "variant": "progressive_disclosure",
        "early_candidate_top3": [],
        "early_candidate_rationale": "",
        "early_top_confidence": None,
        "revision_summary": "",
        "revision_evidence": "",
        "anchoring_flag": False,
        "confidence_instability_score": 0.0,
        "revision_delta": 0.0,
        "belief_penalty_score": 0.0,
    }

    early_prompt = early_prompt_template.replace("{options_text}", options_text).replace(
        "{early_case_text}", early_case_text
    )
    try:
        raw_early = call_model(client, model, early_prompt, retry_attempts, retry_delay)
        early = parse_early_ranked_option_indices_mcq(raw_early, len(options))
    except Exception as exc:
        telemetry["early_error"] = str(exc)
        return -1, telemetry

    telemetry["early_candidate_top3"] = early.ranked_indices
    telemetry["early_candidate_rationale"] = early.rationale
    telemetry["early_top_confidence"] = early.confidences[0] if early.confidences else None

    early_ranked_block = format_early_options_for_prompt(
        options=options,
        ranked_indices=early.ranked_indices[:3],
        confidences=early.confidences,
    )
    revision_prompt = revision_prompt_template.replace(
        "{early_ranked_block}", early_ranked_block
    ).replace(
        "{early_case_text}", early_case_text
    ).replace(
        "{case_text}", case_text
    )
    try:
        raw_revision = call_model(client, model, revision_prompt, retry_attempts, retry_delay)
        revision = parse_revision_decision_mcq(
            raw_revision,
            len(options),
            early.ranked_indices[:3],
        )
    except Exception as exc:
        telemetry["revision_error"] = str(exc)
        return early.ranked_indices[0], telemetry

    early_top_idx = early.ranked_indices[0]
    early_top_label = options[early_top_idx]
    final_label = options[revision.final_choice_index]
    early_top_conf = early.confidences[0] if early.confidences else 0.0

    scores = compute_belief_revision_scores(
        early_top_label=early_top_label,
        final_label=final_label,
        early_top_confidence=early_top_conf,
        final_confidence=revision.final_confidence,
        contradiction_found=revision.contradiction_found,
        kept_labels=[options[idx] for idx in revision.kept_indices if 0 <= idx < len(options)],
    )

    instability = scores.confidence_instability_score
    if instability < revision_instability_threshold:
        instability = 0.0
    if early_top_conf < early_conf_threshold:
        instability = 0.0
    penalty = (0.5 * float(scores.anchoring_flag)) + (0.3 * instability) + (
        0.2 * (1.0 if (scores.revision_delta >= 1.0 and not revision.contradiction_found) else 0.0)
    )
    penalty = max(0.0, min(1.0, penalty))

    telemetry["revision_summary"] = revision.revision_summary
    telemetry["revision_evidence"] = revision.rationale
    telemetry["anchoring_flag"] = scores.anchoring_flag
    telemetry["confidence_instability_score"] = instability
    telemetry["revision_delta"] = scores.revision_delta
    telemetry["belief_penalty_score"] = penalty

    return revision.final_choice_index, telemetry


def process_batch(
    batch: pd.DataFrame,
    args: argparse.Namespace,
    client,
    embedding_service: JinaEmbeddingService,
    candidate_prompt_template: str,
    discriminator_prompt_template: str,
    question_prompt_template: str,
    answer_prompt_template: str,
    integration_prompt_template: str,
    early_prompt_template: str,
    revision_prompt_template: str,
    random_seed: int,
) -> List[Dict[str, Any]]:
    results = []
    rng = random.Random(random_seed)

    for _, row in batch.iterrows():
        case_text = str(row.get("case presentation", row.get("clean text", "")))
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
        elif args.variant == "discriminative_question":
            predicted_index, telemetry = evaluate_case_discriminative_question(
                client=client,
                model=args.model,
                case_text=case_text,
                options=options,
                candidate_prompt_template=candidate_prompt_template,
                question_prompt_template=question_prompt_template,
                answer_prompt_template=answer_prompt_template,
                integration_prompt_template=integration_prompt_template,
                retry_attempts=args.retry_attempts,
                retry_delay=args.retry_delay,
            )
        elif args.variant == "progressive_disclosure":
            predicted_index, telemetry = evaluate_case_progressive_disclosure(
                client=client,
                model=args.model,
                case_text=case_text,
                options=options,
                early_prompt_template=early_prompt_template,
                revision_prompt_template=revision_prompt_template,
                disclosure_fraction=args.disclosure_fraction,
                early_conf_threshold=args.early_confidence_threshold,
                revision_instability_threshold=args.revision_instability_threshold,
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
                "Discriminative Question": telemetry.get("discriminative_question", ""),
                "Question Target Variable": telemetry.get("question_target_variable", ""),
                "Extracted Answer": telemetry.get("extracted_answer", ""),
                "Answer Confidence": telemetry.get("answer_confidence"),
                "Evidence Spans JSON": json.dumps(telemetry.get("evidence_spans", [])),
                "Integration Summary": telemetry.get("integration_summary", ""),
                "Discriminator Rationale": telemetry.get("discriminator_rationale", ""),
                "Differentiators JSON": json.dumps(telemetry.get("differentiators", [])),
                "Early Candidate Top3": json.dumps(telemetry.get("early_candidate_top3", [])),
                "Early Candidate Rationale": telemetry.get("early_candidate_rationale", ""),
                "Early Top Confidence": telemetry.get("early_top_confidence"),
                "Revision Summary": telemetry.get("revision_summary", ""),
                "Revision Evidence": telemetry.get("revision_evidence", ""),
                "Anchoring Flag": telemetry.get("anchoring_flag", False),
                "Confidence Instability Score": telemetry.get("confidence_instability_score"),
                "Revision Delta": telemetry.get("revision_delta"),
                "Belief Penalty Score": telemetry.get("belief_penalty_score"),
                "Error": telemetry.get("candidate_error")
                or telemetry.get("similarity_error")
                or telemetry.get("early_error")
                or telemetry.get("revision_error")
                or telemetry.get("question_error")
                or telemetry.get("answer_error")
                or telemetry.get("integration_error")
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

    if args.variant == "semantic_similarity_gated":
        prompt_folder = "semantic_similarity"
    elif args.variant == "discriminative_question":
        prompt_folder = "discriminative_question"
    elif args.variant == "progressive_disclosure":
        prompt_folder = "progressive_disclosure"
    else:
        prompt_folder = "semantic_similarity"

    candidate_prompt_template = (
        load_prompt_template(prompt_folder, "candidate_mcq")
        if args.variant in {"semantic_similarity_gated", "discriminative_question"}
        else ""
    )
    discriminator_prompt_template = load_prompt_template(
        "semantic_similarity",
        "discriminator_mcq",
    )
    question_prompt_template = load_prompt_template(prompt_folder, "question_mcq") if args.variant == "discriminative_question" else ""
    answer_prompt_template = load_prompt_template(prompt_folder, "answer_extraction_mcq") if args.variant == "discriminative_question" else ""
    integration_prompt_template = load_prompt_template(prompt_folder, "integrate_mcq") if args.variant == "discriminative_question" else ""
    early_prompt_template = load_prompt_template(prompt_folder, "early_mcq") if args.variant == "progressive_disclosure" else ""
    revision_prompt_template = load_prompt_template(prompt_folder, "revision_mcq") if args.variant == "progressive_disclosure" else ""

    all_results: List[Dict[str, Any]] = []
    effective_batch_size = min(args.batch_size, len(ds))
    if effective_batch_size < args.batch_size:
        logger.warning(
            "Requested batch size (%s) exceeds dataset size (%s); using %s rows per batch",
            args.batch_size,
            len(ds),
            effective_batch_size,
        )

    for batch_num in range(args.n_batches):
        logger.info("Processing batch %s/%s", batch_num + 1, args.n_batches)
        batch = ds.sample(n=effective_batch_size, random_state=args.random_seed + batch_num)

        batch_results = process_batch(
            batch=batch,
            args=args,
            client=client,
            embedding_service=embedding_service,
            candidate_prompt_template=candidate_prompt_template,
            discriminator_prompt_template=discriminator_prompt_template,
            question_prompt_template=question_prompt_template,
            answer_prompt_template=answer_prompt_template,
            integration_prompt_template=integration_prompt_template,
            early_prompt_template=early_prompt_template,
            revision_prompt_template=revision_prompt_template,
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
