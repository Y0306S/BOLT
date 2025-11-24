1. Clone repository using git lfs to also clone all large files

2. Install python dependencies using your favorite pyproject.toml-compatible tool, e.g., `uv sync -U`

3. Optimization needs to execute against a PostgreSQL database pre-loaded with the target dataset. A docker container for testing can be built from `imdb_postgres/`. Supply the connection details using environment variables:

   - `DB_HOST=172.17.0.2`
   - `DB_USER=imdb`
   - `DB_PASSWORD=imdb`.

4. Run the following script to start an optimization run for a particular workload specified by the --workload_name argument.

```Bash
cd bayes_lqo/lolbo_scripts
```

If want to track with weights and biases, this command is fine. Otherwise, set the env var `WANDB_MODE="offline"` or `WANDB_MODE="disabled"`.

```Bash
python info_transformer_vae_optimization.py --workload_name CEB_1A10 --allow_cross_joins False --init_w_bao True --wandb_entity xxx - run_lolbo - done
```

Workload names are i.e. CEB_8A94, CEB_1A1272, CEB_2B118, etc. A list of CEB workloads can be found at ```tasks/CEB_tasks.txt```