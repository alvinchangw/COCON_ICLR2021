#!/bin/sh

export TRAIN_FILE=data/gpt2output/medium-345M-k40.train.jsonl
export TEST_FILE=data/gpt2output/medium-345M-k40.test.jsonl
export HISTORY_SOURCE_FILE=prompts/topic_prefixes
export CONTEXT_SOURCE_FILE=BOW/only_computers.txt

python traininfer_cocon.py \
    --do_cocon_compute \
    --output_dir=models/COCON \
    --cocon_output_filename computers_cocon_output.txt \
    --cocon_output_jsonl_filename computers_cocon_output.jsonl \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-medium \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$TEST_FILE \
    --output_hidden_for_cocon_after_block_ind 6 \
    --per_gpu_eval_batch_size 1 \
    --cocon_compute_history_source_data_file=$HISTORY_SOURCE_FILE \
    --cocon_compute_context_source_data_file=$CONTEXT_SOURCE_FILE \
    --prepend_bos_token_to_line \
    --eval_compute_without_checkpoint \
    --gen_cs_len 5 \
    --generate_length 80 \
    --line_by_line_cs \
    --line_by_line_hs \
    --enumerate_all_cs_for_each_hs \
    --seed 42 \




export CONTEXT_SOURCE_FILE=BOW/only_legal.txt

python traininfer_cocon.py \
    --do_cocon_compute \
    --output_dir=models/COCON \
    --cocon_output_filename legal_cocon_output.txt \
    --cocon_output_jsonl_filename legal_cocon_output.jsonl \
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
    --generate_length 80 \
    --line_by_line_cs \
    --line_by_line_hs \
    --enumerate_all_cs_for_each_hs \
    --seed 42 \


export CONTEXT_SOURCE_FILE=BOW/only_politician.txt

python traininfer_cocon.py \
    --do_cocon_compute \
    --output_dir=models/COCON \
    --cocon_output_filename politician1wWbos_gen80_cocon_output_wprependbaseline.txt \
    --cocon_output_jsonl_filename politician1wWbos_gen80_cocon_output_wprependbaseline.jsonl \
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
    --generate_length 80 \
    --line_by_line_cs \
    --line_by_line_hs \
    --enumerate_all_cs_for_each_hs \
    --seed 42 \


export CONTEXT_SOURCE_FILE=BOW/only_religion.txt

python traininfer_cocon.py \
    --do_cocon_compute \
    --output_dir=models/COCON \
    --cocon_output_filename religion_cocon_output.txt \
    --cocon_output_jsonl_filename religion_cocon_output.jsonl \
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
    --generate_length 80 \
    --line_by_line_cs \
    --line_by_line_hs \
    --enumerate_all_cs_for_each_hs \
    --seed 42 \


export CONTEXT_SOURCE_FILE=BOW/only_scientist.txt

python traininfer_cocon.py \
    --do_cocon_compute \
    --output_dir=models/COCON \
    --cocon_output_filename science_cocon_output.txt \
    --cocon_output_jsonl_filename science_cocon_output.jsonl \
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
    --generate_length 80 \
    --line_by_line_cs \
    --line_by_line_hs \
    --enumerate_all_cs_for_each_hs \
    --seed 42 \

