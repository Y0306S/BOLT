# Query Plan Fine-tuning Scripts

This directory contains scripts for generating training data and sampling from fine-tuned models for query plan optimization.

## Files

- `generate_openai_ft_data.py`: Script to generate training data for fine-tuning from a CSV file.
- `sampling_openai.py`: Script to sample query plans from a fine-tuned OpenAI model.
- `data/example_finetuning_data.csv`: Example input CSV data.
- `data/example_finetuning_data.jsonl`: Example output JSONL data for fine-tuning.
- `torchtune_config/`: Directory containing configuration files for fine-tuning with Torchtune.

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

### Torchtune Fine-tuning

We provide configuration files for fine-tuning Llama 3.1 8B and Qwen 2.5 7B models using [Torchtune](https://github.com/pytorch/torchtune).

1.  **Install Torchtune**: Follow the installation instructions in the [Torchtune repository](https://github.com/pytorch/torchtune).
2.  **Download Model Weights**: Download the base model weights (e.g., `meta-llama/Meta-Llama-3-8B-Instruct` or `Qwen/Qwen2.5-7B-Instruct`) using the `tune download` command or from Hugging Face.
3.  **Update Configuration**: Edit the YAML files in `torchtune_config/` to point to your downloaded model weights, tokenizer, and dataset.
    *   `checkpointer.checkpoint_dir`: Path to the model weights.
    *   `tokenizer.path`: Path to the tokenizer file.
    *   `dataset.data_files`: Path to your training data (e.g., `data/example_finetuning_data.jsonl`).
    *   `output_dir`: Directory where checkpoints will be saved.
4.  **Run Fine-tuning**:
    ```bash
    tune run --nnodes 1 --nproc_per_node <NUM_GPUS> full_finetune_distributed --config torchtune_config/llama3_1_8B_instruct_full.yaml
    ```
    Replace `<NUM_GPUS>` with the number of GPUs available.
