from openai import OpenAI
# client = OpenAI() # User should set up their client
import pandas as pd
from tqdm import tqdm
import os
import concurrent.futures
import sys

# Add path to apex_oracle
# Assuming this file is in BOLT-anonymous-release/fine-tuning/peptides/
# and apex_oracle is in BOLT-anonymous-release/optimization/peptides/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../optimization/peptides")))

from apex_oracle.refseqs import REFERENCE_SEQUENCE
# REFERENCE_SEQUENCE = REFERENCE_SEQUENCE[900:920] # Example slice
from collections import defaultdict

# Placeholder for output file
output_file = "./sampled_data/sample_output.jsonl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Placeholder for model ID
model = "ft:gpt-4o-mini-2024-07-18:organization:model-name:id" 

temperature = 0.1
TIMEOUT_SECONDS = 30  # API call timeout
MAX_WORKERS = 20  # Reduced default workers for safety

print(f"Using temperature: {temperature}")
print(f"Using model: {model}")

client = OpenAI()

def process_request(request):
    peptide, query = request
    messages = [
        {"role": "system", "content": f"You are a specialized assistant that modifies peptide sequences to enhance antimicrobial activity. Make up to 25% sequence modifications based on known antimicrobial peptide properties such as: positive charge, hydrophobicity, and amphipathicity."},
        {"role": "user", "content": query}
    ]
    completions = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=TIMEOUT_SECONDS  # Add timeout
    )
    return peptide, completions.choices[0].message.content

# Prepare all requests (Example: 10 requests per peptide)
all_requests = []
for peptide in REFERENCE_SEQUENCE[:5]: # Example: first 5 peptides
    all_requests.extend([(peptide, peptide)] * 10)

total_requests = len(all_requests)
success_threshold = int(total_requests * 1.0)  # 100% completion target
completed_count = 0

# Create DataFrame upfront
results_df = pd.DataFrame({
    "source_peptide": REFERENCE_SEQUENCE[:5], # Matching the slice above
    "generated_answers": [[] for _ in range(len(REFERENCE_SEQUENCE[:5]))]
})


# Process all requests in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_request, req): req for req in all_requests}
    results = []
    
    with tqdm(total=success_threshold, desc="Generating variants") as progress:
        for future in concurrent.futures.as_completed(futures):
            try:
                peptide, result = future.result()
                results.append((peptide, result))
                completed_count += 1
                progress.update(1)
                
                # Early termination at threshold
                if completed_count >= success_threshold:
                    # Cancel remaining futures
                    remaining = [f for f in futures if not f.done()]
                    for f in remaining:
                        f.cancel()
                    print(f"\nReached {completed_count}/{total_requests} requests, terminating early")
                    break
                    
            except Exception as e:
                print(f"Request failed for peptide {peptide}: {str(e)}")
                # Check if we can still reach threshold
                remaining_possible = total_requests - completed_count
                if (completed_count + remaining_possible) < success_threshold:
                    print("Cannot reach threshold, terminating early")
                    break

# Group results by peptide using dictionary
peptide_results = defaultdict(list)
for peptide, result in results:
    peptide_results[peptide].append(result)

# Update DataFrame with collected results
results_df["generated_answers"] = results_df["source_peptide"].map(peptide_results)

# Save results
results_df.to_json(output_file, orient="records", lines=True)
