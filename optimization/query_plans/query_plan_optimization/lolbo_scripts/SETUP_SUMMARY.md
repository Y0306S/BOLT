# Summary: LOLBO Query Plan Optimization Setup

## What I Found

I've explored the BOLT repository structure at `/workspace/optimization/query_plans/query_plan_optimization/` and located all the relevant components for running query plan optimization with LOLBO.

### Key Files Located:

1. **LOLBO Implementation**: `/workspace/optimization/query_plans/query_plan_optimization/lolbo/`
   - `lolbo.py`: Core LOLBO state management and optimization loop
   - `latent_space_objective.py`: Base class for latent space objectives
   - `info_transformer_vae_objective.py`: VAE-specific objective for query plans

2. **Scripts**: `/workspace/optimization/query_plans/query_plan_optimization/lolbo_scripts/`
   - `optimize.py`: Main optimization runner class
   - `info_transformer_vae_optimization.py`: Task-specific optimization for InfoTransformer VAE
   - `run_baoinit_CEB_1A10.sh`: Example bash script for running optimization

3. **VAE Directory**: `/workspace/optimization/query_plans/query_plan_optimization/vae/`
   - `model.py`: VAE model definition
   - `CEB_64.ckpt.placeholder`: Placeholder file (you need to upload your actual checkpoint here)

4. **Workloads**: `/workspace/optimization/query_plans/query_plan_optimization/workload/`
   - CEB-3K queries in `ceb-3k/` directory
   - Support for JOB, Stack Overflow, DSB benchmarks

5. **Initialization Data**: `/workspace/optimization/query_plans/query_plan_optimization/initialization_data/`
   - Contains placeholder files for Bao and BOLT LLM initialization data
   - These are Git LFS files that need to be pulled or replaced with actual data

## What I Created For You

### 1. Ready-to-Run Script: `run_CEB_64_100tasks.py`

Location: `/workspace/optimization/query_plans/query_plan_optimization/lolbo_scripts/run_CEB_64_100tasks.py`

This script is configured exactly as you requested:
- ✅ **10 initialization points** from Bao data
- ✅ **90 BO iterations** 
- ✅ **Total of 100 configurations**
- ✅ Uses your **CEB_64 VAE** (64-dimensional latent space)
- ✅ Thompson Sampling acquisition function
- ✅ TuRBO (Trust Region Bayesian Optimization)

**How to run:**
```bash
cd /workspace/optimization/query_plans/query_plan_optimization/lolbo_scripts

python run_CEB_64_100tasks.py \
    --workload_name CEB_1A10 \
    --vae_ckpt_path ../vae/CEB_64.ckpt \
    --wandb_entity your_wandb_username
```

### 2. Documentation: `README_LOLBO_SETUP.md`

Location: `/workspace/optimization/query_plans/query_plan_optimization/lolbo_scripts/README_LOLBO_SETUP.md`

Comprehensive guide covering:
- How LOLBO works
- Configuration parameters explained
- Multiple ways to run optimization
- Troubleshooting tips
- Available workloads

## How LOLBO Works Here

The optimization process follows this loop:

```
1. Initialize with Bao data (10 query plans with known runtimes)
        ↓
2. Encode plans to 64-dim latent space using your VAE encoder
        ↓
3. Train Gaussian Process surrogate model on (latent_z, runtime) pairs
        ↓
4. Use Thompson Sampling to propose new latent point z_new
        ↓
5. Decode z_new → query plan using your VAE decoder
        ↓
6. Execute query plan on database (oracle call) → get runtime
        ↓
7. Add (z_new, runtime) to training data
        ↓
8. Update Gaussian Process model
        ↓
9. Repeat steps 4-8 for 90 iterations (until 100 total evaluations)
```

Key innovation: Instead of searching discrete query plan space directly, LOLBO searches in the continuous latent space learned by your VAE, making Bayesian optimization much more efficient.

## What You Need To Do Next

### 1. Upload Your VAE Checkpoint ⚠️ CRITICAL

Place your trained VAE file at:
```
/workspace/optimization/query_plans/query_plan_optimization/vae/CEB_64.ckpt
```

Currently there's only a placeholder file there.

### 2. Ensure Initialization Data Exists

The code expects one of these initialization sources:

**Option A: Bao Data (Recommended - currently configured)**
- File: `../initialization_data/bao_censored_initializations.csv`
- Or: `../initialization_data/ceb_3k_bao_initialization.csv` (fallback)

**Option B: LLM-generated Data from BOLT**
- File: `../initialization_data/{N}tasks_samples_temp07.jsonl`
- Where N = number of tasks (e.g., 150tasks_samples_temp07.jsonl)

**Option C: Random Initialization**
- File: `../initialization_data/random_runs.csv`

Check what's available and update the script if needed:
```python
# In run_CEB_64_100tasks.py, modify these lines:
init_w_bao=True,       # Set to True for Bao data
init_w_llm=False,      # Set to True for LLM data
init_w_random=False,   # Set to True for random data
```

### 3. Database Configuration

Ensure you have:
- PostgreSQL running with IMDB schema loaded
- Database user with proper permissions
- Connection parameters configured in workload settings

The default configuration expects:
- Database name: `imdb`
- User: `imdb`

### 4. Install Dependencies

From `/workspace/optimization/query_plans/`:
```bash
pip install -e .
# or with uv
uv pip install -e .
```

Required packages include:
- torch, botorch, gpytorch (for BO)
- lightning (for VAE)
- psycopg2-binary (for PostgreSQL)
- wandb (optional, for tracking)
- fire (for command-line interface)

### 5. (Optional) Configure Weights & Biases

If you want to track experiments:
- Replace `wandb_entity="xxx"` with your actual wandb username
- Or set `track_with_wandb=False` to disable tracking

## Additional Information You Requested

### About LOLBO Implementation

The LOLBO algorithm in this codebase:

1. **Uses Trust Regions (TuRBO)**: Constrains search to local regions in latent space, expanding/contracting based on progress
2. **Handles Censored Observations**: Query timeouts are treated as censored data points (query took ≥ timeout seconds)
3. **Supports End-to-End Updates**: Can optionally fine-tune the VAE during optimization (disabled by default in your config)
4. **Multiple Acquisition Functions**: Thompson Sampling (default) or Expected Improvement
5. **Batch Processing**: Supports batched evaluations (though your config uses batch size 1)

Key files to understand the implementation:
- `lolbo/lolbo.py` (lines 325-380): Main optimization loop
- `lolbo/utils/bo_utils/turbo.py`: Trust region BO implementation
- `lolbo/utils/eulbo_utils.py`: End-to-end update utilities

### Customization Options

You can easily modify `run_CEB_64_100tasks.py` to:

```python
# Try different acquisition functions
acq_func="ei"  # Expected Improvement instead of "ts"

# Enable true LOLBO with end-to-end VAE updates
update_e2e=True,
e2e_freq=10,

# More aggressive exploration
vanilla_bo=True,  # Disable trust regions

# Larger batches (if you can parallelize oracle calls)
bsz=4,  # Evaluate 4 points per iteration

# Different initialization
num_initialization_points=20,  # More initial data
max_n_oracle_calls=200,        # Larger budget
```

## Questions?

If you need help with:
- Setting up the database
- Pulling Git LFS files for initialization data
- Understanding specific LOLBO parameters
- Debugging connection issues
- Analyzing results after optimization runs

Just let me know!
