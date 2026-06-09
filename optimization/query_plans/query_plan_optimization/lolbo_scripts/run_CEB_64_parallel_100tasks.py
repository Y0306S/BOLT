#!/usr/bin/env python3
"""
Run LOLBO Optimization with Parallel Database Instances
Configuration: 4 parallel databases, batch size 4, 100 total configurations
This significantly speeds up optimization by evaluating 4 queries simultaneously
"""

import sys
import os
import multiprocessing as mp
from functools import partial

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from info_transformer_vae_optimization import InfoTransformerVAEOptimization

def run_parallel_optimization(
    workload_name="CEB_1A10",
    vae_ckpt_path="../vae/CEB_64.ckpt",
    wandb_entity="xxx",
    seed=42,
    track_with_wandb=True,
    num_db_instances=4,
):
    """
    Run LOLBO optimization with parallel database instances.
    
    Args:
        workload_name: Name of the workload to optimize
        vae_ckpt_path: Path to your trained VAE checkpoint file
        wandb_entity: Your wandb username for tracking
        seed: Random seed for reproducibility
        track_with_wandb: Whether to track experiments with wandb
        num_db_instances: Number of parallel PostgreSQL instances (default: 4)
    """
    
    # Configure database ports for parallel instances
    db_ports = [5432 + i for i in range(1, num_db_instances + 1)]
    
    print(f"Setting up optimization with {num_db_instances} parallel database instances")
    print(f"Database ports: {db_ports}")
    print(f"Batch size: {num_db_instances} (evaluating {num_db_instances} queries per iteration)")
    print("")
    
    optimizer = InfoTransformerVAEOptimization(
        # VAE Configuration
        path_to_vae_statedict=vae_ckpt_path,
        dim=64,  # Latent space dimension (matches CEB_64)
        
        # Workload Configuration
        workload_name=workload_name,
        which_query_language="aliases",
        allow_cross_joins=False,
        
        # Database Configuration for Parallel Execution
        db_ports=db_ports,  # Use multiple database instances
        
        # Initialization Configuration
        init_w_bao=True,       # Use Bao data for initialization
        init_w_llm=False,      # Don't use LLM-generated data
        init_w_random=False,   # Don't use random initialization
        
        # Budget Configuration - KEY PARAMETERS
        num_initialization_points=10,  # 10 initial points from Bao
        max_n_oracle_calls=100,        # Total budget: 10 init + 90 BO = 100
        
        # Bayesian Optimization Configuration - PARALLEL MODE
        acq_func="ts",         # Thompson Sampling acquisition function
        bsz=num_db_instances,  # Batch size = number of DB instances (parallel BO)
        vanilla_bo=False,      # Use TuRBO (Trust Region BO)
        
        # Model Training Configuration
        learning_rte=0.01,
        init_n_update_epochs=80,    # Epochs to train on initial data
        num_update_epochs=2,        # Epochs per BO iteration
        e2e_freq=10,                # End-to-end update frequency
        update_e2e=False,           # Don't update end-to-end (TuRBO mode)
        k=100,                      # Track top k points
        
        # Timeout/Censoring Configuration
        censored_observations=True,
        censored_obs_is_max=True,
        timeout_strategy="ours",
        timeout_percentile=0.1,
        constant_timeout=1_000_000_000,
        
        # Logging Configuration
        verbose=True,
        save_freq=10,               # Save results every 10 iterations
        flag_correct_adding_vae_decode_time=True,
        
        # Weights & Biases Configuration
        seed=seed,
        track_with_wandb=track_with_wandb,
        wandb_entity=wandb_entity,
        wandb_project_name=f"optimize-{workload_name}-parallel-{num_db_instances}x",
    )
    
    # Run the optimization
    optimizer.run_lolbo()
    
    print(f"\nOptimization completed for workload: {workload_name}")
    print(f"Total oracle calls made: {optimizer.lolbo_state.objective.num_calls}")
    print(f"Best score found: {optimizer.lolbo_state.best_score_seen}")
    print(f"Speedup achieved: ~{num_db_instances}x compared to sequential execution")
    
    return optimizer


if __name__ == "__main__":
    import fire
    
    # Allow command-line execution with customizable parameters
    fire.Fire(run_parallel_optimization)
    
    # Example usage:
    # python run_CEB_64_parallel_100tasks.py --workload_name CEB_1A10 --wandb_entity your_username --num_db_instances 4
    # python run_CEB_64_parallel_100tasks.py --workload_name CEB_2B5 --wandb_entity your_username --num_db_instances 4 --seed 123
