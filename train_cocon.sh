#!/bin/sh

export TRAIN_FILE=data/gpt2output/medium-345M-k40.train.jsonl
export TEST_FILE=data/gpt2output/medium-345M-k40.test.jsonl

python traininfer_cocon.py \
    --do_train \
    --output_dir=models/COCON \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-medium \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$TEST_FILE \
    --per_gpu_train_batch_size 16 \
    --per_gpu_train_lm_batch_size 16 \
    --per_gpu_eval_batch_size 1 \
    --overwrite_output_dir \
    --num_train_epochs 2 \
    --track_loss_gradnorms \
    --min_hs_tis_split_offset -2 \
    --max_hs_tis_split_offset 2 \
    --save_total_limit 10 \
    --save_steps 999999999 \
    --per_gpu_train_cycle_ar_cocon_recon_batch_size 16 \
    --epoch_ind_to_start_cycle_ar_cocon_recon 0 \
    --epoch_ind_to_start_other_context_cocon 0 \
    --epoch_ind_to_start_adv 0 \
    --step_ind_to_start_cycle_ar_cocon_recon 2000 \
    --step_ind_to_start_other_context_cocon 2000 \
    --step_ind_to_start_adv 2000 \
    --disc_update_interval 5 \
    --output_hidden_for_cocon_after_block_ind 6 \
    --logging_steps 500 \
    --lambda_self_cocon_lm_loss 1 \
    --self_token_mask_prob 1 \
    --lambda_cycle_ar_cocon_recon_lm_loss 1 \
    --lambda_other_context_cocon_lm_loss 0 \
    --lambda_adv 1 \
    --lambda_hist_cocon_lm_loss 1 \
    --eval_compute_without_checkpoint \
    --num_cocon_generate 100 \

