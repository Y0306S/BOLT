from openai import OpenAI

client = OpenAI()
import pandas as pd
from tqdm import tqdm
import fire
import re

import concurrent.futures

temperature = 0.7
# model = "ft:gpt-4o-mini-2024-07-18:organization:model-name:id"
model = "gpt-4o-mini-2024-07-18"  # Placeholder or default model

print(f"Using temperature: {temperature}")
print(f"Using model: {model}")


def process_request(query):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides efficient orderings for given queries.",
        },
        {"role": "user", "content": query},
    ]
    completions = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    return completions.choices[0].message.content


import os


def extract_task_id(task):
    pattern = r"^.*[a-zA-Z]"
    match = re.search(pattern, task)
    if match:
        return match.group(0)
    else:
        return task


def get_task_description(task):
    # Note: This function assumes a 'workload' directory exists with SQL files.
    # Users should provide their own workload files or adjust the paths below.
    if not os.path.exists("./workload"):
        return "SELECT * FROM table"  # Placeholder return if workload dir is missing

    workload_group = task.split("_")[0].lower()
    task = task.split("_")[1].lower()

    if workload_group == "job":
        with open(f"./workload/job/{task}.sql") as f:
            return f.read()
    elif workload_group == "ceb":  # assume 3k for now, ignore 13k
        # get task id from task, everything before and including last alphabet character (a~z)
        task_id = extract_task_id(task)
        with open(f"./workload/ceb-3k/{task_id}/{task}.sql") as f:
            return f.read()
    return "SELECT * FROM table"


# Example: Load tasks from a file or define them here
# tasks = ["CEB_10A10", "CEB_10A15"]
# For demonstration, we'll try to read from the example CSV if it exists
if os.path.exists("data/example_finetuning_data.csv"):
    df = pd.read_csv("data/example_finetuning_data.csv")
    tasks = df["task"].unique().tolist()[:5]  # Take first 5 tasks
else:
    tasks = ["CEB_10A10"]  # Default task

print(f"Sampling for {len(tasks)} tasks")

# for each task, get the description, and query the model 100 times in parallel
task_descriptions = [get_task_description(task) for task in tasks]

results_df = pd.DataFrame(columns=["src_code"])
results_df["src_code"] = task_descriptions
results_df["task"] = tasks
results_df["generated_answers"] = results_df.apply(lambda x: [], axis=1)

all_generated_answers = []

for tqdm_task, task in tqdm(enumerate(tasks), total=len(tasks)):
    task_description = task_descriptions[tqdm_task]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_request, [task_description] * 50))
        # again for 100 samples
        # results += list(executor.map(process_request, [task_description]*50))
    all_generated_answers.append(results)

results_df["generated_answers"] = all_generated_answers

# save the results to csv
output_file = "samples/sampled_plans.jsonl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
results_df.to_json(output_file, orient="records", lines=True)
