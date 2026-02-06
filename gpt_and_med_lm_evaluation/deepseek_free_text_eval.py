"""
DeepSeek Chat free-text evaluation script.

Uses DeepSeek Chat as the judge model for open-ended diagnosis evaluation.
DeepSeek API is OpenAI-compatible, so we use the OpenAI client with custom base_url.
"""

import pandas as pd
import bert_score
import os
from dotenv import load_dotenv
from openai import OpenAI
import time
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
        case_presentation = row['clean text']
        true_diagnosis = row['final diagnosis']

        prompt = (
            f"Predict the diagnosis of this case presentation of a patient. "
            f"Return the final diagnosis in one concise sentence without any further elaboration.\n"
            f"For example: <diagnosis name here>\n"
            f"Case presentation: {case_presentation}\n"
            f"Diagnosis:"
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
                break
            except Exception as e:
                print(f"API error: {e}. Waiting for 60 seconds before retrying.")
                time.sleep(60)

        results.append({
            'Case presentation': case_presentation,
            'True diagnosis': true_diagnosis,
            'Generated diagnosis': generated_diagnosis
        })
        time.sleep(1)  # Sleep for 1 second between each API call

    return results


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek Chat free-text diagnosis evaluation"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/Case_report_w_images_dis_VF.csv",
        help="Path to input dataset CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/deepseek_free_text_batched.csv",
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

    # Calculate BERTScore F1 using DeBERTa model
    model_type = "microsoft/deberta-xlarge-mnli"
    predictions = results_df['Generated diagnosis'].tolist()
    references = results_df['True diagnosis'].tolist()
    P, R, F1 = bert_score.score(predictions, references, lang="en", model_type=model_type)

    # Add BERTScore F1 to the DataFrame
    results_df['BERTScore F1'] = F1.tolist()

    # Save the DataFrame to a CSV file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Total cases: {len(results_df)}")
    print(f"  Mean BERTScore F1: {results_df['BERTScore F1'].mean():.4f}")
    print(f"  Std BERTScore F1: {results_df['BERTScore F1'].std():.4f}")


if __name__ == "__main__":
    main()
