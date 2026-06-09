# LOLBO Query Plan Optimization - Quick Start Guide

## Overview

This guide explains how to run query plan optimization using LOLBO (Latent Space Bayesian Optimization) with your trained CEB_64 VAE model.

## What is LOLBO?

LOLBO is a latent space optimization framework that combines:
1. **VAE (Variational Autoencoder)**: Encodes discrete query plans into continuous latent space
2. **Bayesian Optimization**: Efficiently searches the latent space for optimal query plans
3. **Trust Region Methods (TuRBO)**: Constrains search to promising regions

### How LOLBO Works Here:

```
Query Plan (discrete) ←→ VAE Encoder ←→ Latent Space (continuous, 64-dim) ←→ VAE Decoder ←→ Query Plan
                              ↑                    ↓
                         Evaluation           Bayesian Optimization
                         (Oracle)             (Acquisition Function)
```

**The Loop:**
1. Initialize with existing data (from Bao, LLM, or random)
2. Encode initial plans to latent space using VAE
3. Train surrogate model (GP) on latent points and their scores
4. Use acquisition function (Thompson Sampling or EI) to propose new latent points
5. Decode latent points back to query plans using VAE
6. Evaluate query plans on database (oracle call)
7. Update surrogate model with new data
8. Repeat until budget exhausted

## Prerequisites

### Required Files:
1. **VAE Checkpoint**: Place your trained `CEB_64.ckpt` file at:
   ```
   /workspace/optimization/query_plans/query_plan_optimization/vae/CEB_64.ckpt
   ```

2. **Initialization Data** (one of the following):
   - Bao data: `../initialization_data/bao_censored_initializations.csv`
   - Random runs: `../initialization_data/random_runs.csv`
   - LLM data: `../initialization_data/{N}tasks_samples_temp07.jsonl`

### Database Setup:
- PostgreSQL database with IMDB schema
- Proper user permissions for query execution
- Database connection configured in workload settings

## Configuration

### Key Parameters Explained:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_initialization_points` | 10 | Number of initial data points from Bao/LLM/random |
| `max_n_oracle_calls` | 100 | Total optimization budget (init + BO iterations) |
| `acq_func` | "ts" | Acquisition function: "ts" (Thompson Sampling) or "ei" (Expected Improvement) |
| `bsz` | 1 | Batch size (1 = sequential BO) |
| `vanilla_bo` | False | Use TuRBO (trust region) vs vanilla BO |
| `init_w_bao` | True | Initialize with Bao data |
| `init_w_llm` | False | Initialize with LLM-generated queries |
| `init_w_random` | False | Initialize with random queries |
| `learning_rte` | 0.01 | Learning rate for model updates |
| `init_n_update_epochs` | 80 | Epochs to train surrogate on initial data |
| `num_update_epochs` | 2 | Epochs per BO iteration update |
| `e2e_freq` | 10 | Frequency of end-to-end updates (if update_e2e=True) |
| `update_e2e` | False | Whether to update VAE end-to-end (LOLBO vs TuRBO) |
| `allow_cross_joins` | False | Allow cross joins in query plans |

## Running Optimization

### Method 1: Using the Python Script (Recommended)

```bash
cd /workspace/optimization/query_plans/query_plan_optimization/lolbo_scripts

# Basic usage
python run_CEB_64_100tasks.py \
    --workload_name CEB_1A10 \
    --vae_ckpt_path ../vae/CEB_64.ckpt \
    --wandb_entity your_wandb_username \
    --seed 42

# Disable wandb tracking
python run_CEB_64_100tasks.py \
    --workload_name CEB_1A10 \
    --track_with_wandb False

# Different workload
python run_CEB_64_100tasks.py \
    --workload_name CEB_2B5 \
    --wandb_entity your_wandb_username \
    --seed 123
```

### Method 2: Direct Command Line

```bash
cd /workspace/optimization/query_plans/query_plan_optimization/lolbo_scripts

python info_transformer_vae_optimization.py \
    --workload_name CEB_1A10 \
    --path_to_vae_statedict ../vae/CEB_64.ckpt \
    --dim 64 \
    --num_initialization_points 10 \
    --max_n_oracle_calls 100 \
    --init_w_bao True \
    --init_w_llm False \
    --acq_func ts \
    --bsz 1 \
    --vanilla_bo False \
    --allow_cross_joins False \
    --wandb_entity your_wandb_username \
    --track_with_wandb True \
    --verbose True \
    - run_lolbo - done
```

## Available Workloads

Check available workloads in the `workload/` directory:
- **CEB-3K**: Queries in `workload/ceb-3k/` (e.g., CEB_1A10, CEB_2B5, etc.)
- **JOB**: IMDB join order benchmark queries
- **Stack Overflow**: SO_PAST, SO_FUTURE, SO_SHIFTED
- **DSB**: Decision Support Benchmark

## Output

### Results Saved:
- CSV file with all collected data: `optimization_all_collected_data/`
- Format: `{project_name}_{run_name}_{workload}_all-data-collected.csv`
- Contains: train_x (query plans), train_y (runtimes), censoring indicators

### Wandb Tracking (if enabled):
- Best score found over time
- Number of oracle calls
- Runtime statistics
- Model update timings
- Trust region length

## Troubleshooting

### Common Issues:

1. **VAE Checkpoint Not Found**
   ```
   FileNotFoundError: [Errno 2] No such file or directory: '../vae/CEB_64.ckpt'
   ```
   **Solution**: Ensure your trained VAE checkpoint is placed at the correct path.

2. **No Initialization Data**
   ```
   AssertionError: no bao init data for workload CEB_1A10
   ```
   **Solution**: Check that initialization data exists for your workload in `../initialization_data/`.

3. **Database Connection Error**
   ```
   psycopg2.OperationalError: could not connect to server
   ```
   **Solution**: Verify PostgreSQL is running and connection parameters are correct.

4. **Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce batch size or use CPU-only mode.

## Modifying the Configuration

To change the optimization parameters, edit `run_CEB_64_100tasks.py`:

```python
optimizer = InfoTransformerVAEOptimization(
    # Change these values:
    num_initialization_points=10,  # Initial data points
    max_n_oracle_calls=100,        # Total budget
    
    # Try different acquisition functions:
    acq_func="ei",  # Expected Improvement instead of Thompson Sampling
    
    # Enable end-to-end updates (true LOLBO):
    update_e2e=True,
    e2e_freq=10,
    
    # Allow more exploration:
    vanilla_bo=True,  # Disable trust regions
)
```

## Next Steps

1. **Upload your VAE**: Place `CEB_64.ckpt` in the vae directory
2. **Verify initialization data**: Check what's available in `initialization_data/`
3. **Test run**: Run with `--track_with_wandb False` first to verify setup
4. **Full optimization**: Enable wandb and run full optimization
5. **Analyze results**: Check saved CSV files and wandb dashboard

## Additional Information

For more details on the LOLBO algorithm, see:
- `lolbo/lolbo.py`: Core LOLBO state management
- `lolbo/latent_space_objective.py`: Base objective class
- `lolbo/info_transformer_vae_objective.py`: VAE-specific objective
- `utils/bo_utils/turbo.py`: Trust region Bayesian optimization implementation
