"""
DeepSeek Chat multiple-choice QA evaluation script.

Uses DeepSeek Chat as the judge model for multiple-choice diagnosis evaluation.
DeepSeek API is OpenAI-compatible, so we use the OpenAI client with custom base_url.
"""

import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
import time
import random
import argparse
import re

from eval_batching import build_eval_batches

load_dotenv()

# DeepSeek API configuration
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


def get_deepseek_client() -> OpenAI:
    """Create DeepSeek client using OpenAI-compatible API."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def parse_predicted_index(text, num_options):
    stripped = str(text).strip()
    if not stripped:
        return -1

    explicit_matches = []
    for pattern in (
        r"(?:option|choice|answer|index|selected)\s*[:#=\-\s]*\(?\s*(\d+)\s*\)?",
        r"\b(?:is|was)\s*(\d+)\b",
    ):
        explicit_matches.extend(re.findall(pattern, stripped, flags=re.IGNORECASE))

    for match in reversed(explicit_matches):
        idx = int(match) - 1
        if 0 <= idx < num_options:
            return idx

    valid_indices = []
    for match in re.finditer(r"(?<!\d)(\d+)(?!\d)", stripped):
        idx = int(match.group(1)) - 1
        if 0 <= idx < num_options:
            valid_indices.append(idx)

    return valid_indices[-1] if valid_indices else -1


def extract_distractors(row, true_diagnosis):
    distractors = []
    true_normalized = str(true_diagnosis).strip().lower()
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


def process_batch(batch, client, model=DEEPSEEK_MODEL):
    """Process a batch of cases using DeepSeek Chat."""
    results = []
    for _, row in batch.iterrows():
        case_presentation = row['case presentation']
        true_diagnosis = row['final diagnosis']
        distractors = extract_distractors(row, true_diagnosis)

        options = [true_diagnosis] + distractors
        random.shuffle(options)
        options_text = "\n".join([f"{i+1}. {option}" for i, option in enumerate(options)])
        prompt = (
            f"Predict the diagnosis of this case presentation of a patient. "
            f"Return only the correct index from the following list, for example: 3\n"
            f"{options_text}\n"
            f"Case presentation: {case_presentation}"
        )

        while True:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                generated_diagnosis = response.choices[0].message.content.strip()
                predicted_index = parse_predicted_index(generated_diagnosis, len(options))
                break
            except Exception as e:
                print(f"API error: {e}. Waiting for 60 seconds before retrying.")
                time.sleep(60)

        results.append({
            'Case presentation': case_presentation,
            'True diagnosis': true_diagnosis,
            'Generated diagnosis': generated_diagnosis,
            'Correct index': options.index(true_diagnosis),
            'Predicted index': predicted_index,
            'Correct': options.index(true_diagnosis) == predicted_index
        })
        time.sleep(1)  # Sleep for 1 second between each API call

    return results


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek Chat multiple-choice diagnosis evaluation"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="ablation_study_tokens.csv",
        help="Path to input dataset CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/deepseek_multiple_choice_batched.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEEPSEEK_MODEL,
        help="DeepSeek model to use (default: deepseek-chat)",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=4,
        help="Number of batches to sample",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250,
        help="Size of each batch",
    )
    
    args = parser.parse_args()
    
    # Set up DeepSeek client
    client = get_deepseek_client()
    
    # Read the CSV file
    ds = pd.read_csv(args.input)
    print(f"Loaded {len(ds)} cases from {args.input}")
    
    all_results = []
    batches = build_eval_batches(
        ds=ds,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        random_seed=0,
        sampling_mode="unique",
    )

    for batch_num, batch in enumerate(batches, start=1):
        print(f"Processing batch {batch_num}/{len(batches)}")

        batch_results = process_batch(batch, client, model=args.model)
        all_results.extend(batch_results)

        print(f"Completed batch {batch_num}/{len(batches)}")
        if batch_num < len(batches):
            time.sleep(10)  # Sleep for 10 seconds between batches

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save the DataFrame to a CSV file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

    # Calculate and print accuracy
    accuracy = sum(results_df['Correct']) / len(results_df)
    print(f"\nAccuracy: {accuracy:.4f} ({sum(results_df['Correct'])}/{len(results_df)})")


if __name__ == "__main__":
    main()
