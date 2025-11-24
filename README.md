# BOLT-anonymous-release

## Repository Structure

The repository is organized into two main directories corresponding to the different parts of BOLT:

### `fine-tuning/`
This directory contains the necessary scripts and data to fine-tune Large Language Models (LLMs) for generating better initializations.
- **`peptides/`**: Contains scripts for generating training data and sampling from models for peptide design.
- **`query_plans/`**: Contains scripts for generating training data and sampling for database query plan optimization. It also includes `torchtune_config/` with YAML configurations for fine-tuning Llama 3.1 and Qwen 2.5 models.

### `optimization/`
This directory contains the core BO algorithms.
- **`peptides/`**: Implements the BO loop for peptide design.
  - **`lolbo/`**: Core logic for Latent Space BO.
  - **`apex_oracle/`**: Oracle for evaluating peptide properties.
  - **`uniref_vae/`**: VAE models for peptide sequences.
- **`query_plans/`**: Implements the BO loop for query planning.
  - **`query_plan_optimization/`**: Main package containing the VAE, oracle interfaces, and BO logic.
  - **`tasks/`**: Definitions of the query plan BO tasks.