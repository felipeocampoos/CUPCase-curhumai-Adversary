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


def process_batch(batch, client, model=DEEPSEEK_MODEL):
    """Process a batch of cases using DeepSeek Chat."""
    results = []
    for _, row in batch.iterrows():
        case_presentation = row['case presentation']
        true_diagnosis = row['final diagnosis']
        distractor2 = row['distractor2']
        distractor3 = row['distractor3']
        distractor4 = row['distractor4']

        options = [true_diagnosis, distractor2, distractor3, distractor4]
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
                try:
                    predicted_index = int(generated_diagnosis[0]) - 1
                except Exception as e:
                    predicted_index = -1
                    print(f"Parse error: {e}")
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

    for batch_num in range(args.n_batches):
        print(f"Processing batch {batch_num + 1}/{args.n_batches}")
        # Randomly sample rows
        batch = ds.sample(n=args.batch_size, random_state=batch_num)

        batch_results = process_batch(batch, client, model=args.model)
        all_results.extend(batch_results)

        print(f"Completed batch {batch_num + 1}/{args.n_batches}")
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
