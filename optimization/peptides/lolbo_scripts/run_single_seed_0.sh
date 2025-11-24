CUDA_VISIBLE_DEVICES="0" python info_transformer_vae_optimization.py \
    --max_n_oracle_calls 200000 \
    --bsz 50 \
    --track_with_wandb False \
    --constraint_function_ids "[similarity]" \
    --constraint_thresholds "[0.75]" \
    --wandb_entity xxx \
    --num_initialization_points 1000 \
    --max_string_length 30 \
    --init_n_update_epochs 20 \
    --task_specific_args "[bacteria_0]" \
    --init_data_path ../apex_oracle/init_data/seed_0_init.txt \
    --init_scores_path ../apex_oracle/init_data/seed_0_scores.csv \
    --wandb_run_tags "[seed_0_single_run]" \
    --constraint_types "[0]" \
    --task_id "apex" \
    run_lolbo
