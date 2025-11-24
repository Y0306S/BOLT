# Peptide Fine-tuning Scripts

This directory contains scripts for generating training data and sampling from fine-tuned models for peptide optimization.

## Files

- `efficient_sample.py`: Script to sample sequences from a fine-tuned OpenAI model.
- `generate_gpt4_ft_data.py`: Script to generate training data for fine-tuning from a CSV file.
- `example_finetuning_data.jsonl`: An example of the JSONL format required for OpenAI fine-tuning.

## Usage

### Sampling (`efficient_sample.py`)

1.  Set your OpenAI API key in the environment or in the script.
2.  Update the `model` variable with your fine-tuned model ID.
3.  Update `output_file` to your desired output path.
4.  Run the script:
    ```bash
    python efficient_sample.py
    ```

### Generating Training Data (`generate_gpt4_ft_data.py`)

1.  Prepare a CSV file with columns `sequence` (target) and `reference_sequence` (source).
2.  Update `data_path` to point to your CSV file.
3.  Update `save_path` to your desired output JSONL path.
4.  Run the script:
    ```bash
    python generate_gpt4_ft_data.py
    ```

## Dependencies

- `openai`
- `pandas`
- `tqdm`
- `token_count` (optional, for counting tokens)
