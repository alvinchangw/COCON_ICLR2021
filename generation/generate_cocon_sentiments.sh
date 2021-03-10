#!/bin/sh

export TRAIN_FILE=data/gpt2output/medium-345M-k40.train.jsonl
export TEST_FILE=data/gpt2output/medium-345M-k40.test.jsonl
export HISTORY_SOURCE_FILE=prompts/sentiment_prefixes
export CONTEXT_SOURCE_FILE=attr_markers/is_perfect

python traininfer_cocon.py \
    --do_cocon_compute \
    --output_dir=models/COCON \
    --cocon_output_filename perfect_cocon_output.txt \
    --cocon_output_jsonl_filename perfect_cocon_output.jsonl \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-medium \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$TEST_FILE \
    --per_gpu_eval_batch_size 1 \
    --output_hidden_for_cocon_after_block_ind 6 \
    --cocon_compute_history_source_data_file=$HISTORY_SOURCE_FILE \
    --cocon_compute_context_source_data_file=$CONTEXT_SOURCE_FILE \
    --prepend_bos_token_to_line \
    --eval_compute_without_checkpoint \
    --gen_cs_len 5 \
    --generate_length 50 \
    --line_by_line_cs \
    --line_by_line_hs \
    --enumerate_all_cs_for_each_hs \


export CONTEXT_SOURCE_FILE=attr_markers/is_horrible

python traininfer_cocon.py \
    --do_cocon_compute \
    --output_dir=models/COCON \
    --cocon_output_filename horrible_cocon_output.txt \
    --cocon_output_jsonl_filename horrible_cocon_output.jsonl \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-medium \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$TEST_FILE \
    --per_gpu_eval_batch_size 1 \
    --output_hidden_for_cocon_after_block_ind 6 \
    --cocon_compute_history_source_data_file=$HISTORY_SOURCE_FILE \
    --cocon_compute_context_source_data_file=$CONTEXT_SOURCE_FILE \
    --prepend_bos_token_to_line \
    --eval_compute_without_checkpoint \
    --gen_cs_len 5 \
    --generate_length 50 \
    --line_by_line_cs \
    --line_by_line_hs \
    --enumerate_all_cs_for_each_hs \

