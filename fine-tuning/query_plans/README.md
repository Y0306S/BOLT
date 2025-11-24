# Query Plan Fine-tuning Scripts

This directory contains scripts for generating training data and sampling from fine-tuned models for query plan optimization.

## Files

- `generate_openai_ft_data.py`: Script to generate training data for fine-tuning from a CSV file.
- `sampling_openai.py`: Script to sample query plans from a fine-tuned OpenAI model.
- `data/example_finetuning_data.csv`: Example input CSV data.
- `data/example_finetuning_data.jsonl`: Example output JSONL data for fine-tuning.

## Usage

### Generating Training Data (`generate_openai_ft_data.py`)

1.  Prepare a CSV file with your training data.
2.  Ensure you have the corresponding SQL workload files in a `workload` directory (or adjust the script to point to them).
3.  Update `raw_data_path` and `output_path` in the script.
4.  Run the script:
    ```bash
    python generate_openai_ft_data.py
    ```

### Sampling (`sampling_openai.py`)

1.  Set your OpenAI API key in the environment.
2.  Update the `model` variable with your fine-tuned model ID.
3.  Run the script:
    ```bash
    python sampling_openai.py
    ```

### OpenAI Fine-tuning

The generated JSONL files can be uploaded directly to the [OpenAI Fine-tuning Dashboard](https://platform.openai.com/finetune) to create a fine-tuned model.
