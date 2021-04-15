# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
import json
import wget
import math

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import get_rank, get_world_size
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    CoconBlock,
    HDiscriminator,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from collections import OrderedDict

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}

class JsonlCoconDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, cs_len, hs_len, tis_len, block_size=None, text_json_key="text", evaluate=False, prepended_text_to_remove=None):
        assert os.path.isfile(file_path)

        self.cs_len = cs_len
        self.hs_len = hs_len
        self.tis_len = tis_len

        if block_size is None:
            block_size = hs_len + max(cs_len, tis_len)
        self.block_size = block_size

        directory, filename = os.path.split(file_path)
        if evaluate and text_json_key != 'text':
            cached_features_file = os.path.join(
                directory, args.model_type + "_cached_cocon_" + str(block_size) + text_json_key + "_" + filename
            )
        else:
            cached_features_file = os.path.join(
                directory, args.model_type + "_cached_cocon_" + str(block_size) + "_" + filename
            )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            if prepended_text_to_remove is not None:
                if ';' in prepended_text_to_remove:
                    prepended_texts = prepended_text_to_remove.split(';')
                    logger.info("prepended_texts: {}".format(prepended_texts))
                else:
                    prepended_texts = [prepended_text_to_remove]
            else:
                prepended_texts = None
            
            lines = []
            with open(file_path, encoding="utf-8") as f:
                for jsonl in tqdm(f):
                    json_dict = json.loads(jsonl)
                    if 'length' in json_dict.keys() and evaluate == False:
                        if json_dict['length'] >= block_size:
                            line = json_dict[text_json_key]
                            if prepended_text_to_remove is not None and len(prepended_texts) == 1 and prepended_text_to_remove in line:
                                line = line[line.index(prepended_text_to_remove)+len(prepended_text_to_remove):]
                            else:
                                if prepended_texts is not None:
                                    for prepended_text in prepended_texts:
                                        if prepended_text in line:
                                            line = line[line.index(prepended_text_to_remove)+len(prepended_text_to_remove):]
                                            break
                            lines.append(line)
                    else:
                        line = json_dict[text_json_key]
                        if prepended_text_to_remove is not None:
                            if len(prepended_texts) == 1 and prepended_text_to_remove in line:
                                line = line[line.index(prepended_text_to_remove)+len(prepended_text_to_remove):]
                            else:
                                for prepended_text in prepended_texts:
                                    if prepended_text in line:
                                        line = line[len(prepended_text):]
                                        break

                        lines.append(line)

            logger.info("Encoding with tokenizer")
            self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=None)["input_ids"]

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]
        overflow_len = len(example) - self.block_size
        if overflow_len > 0:
            random_ind = random.randint(0, overflow_len) # random integer between 0 and overflow_len (both inclusive)
        else:
            random_ind = 0
        example_block = example[random_ind:random_ind+self.block_size]

        return torch.tensor(example_block, dtype=torch.long)

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512, prepend_bos_token=False):
        assert os.path.isfile(file_path)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            if prepend_bos_token:
                lines = [tokenizer.bos_token + line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            else:
                lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        
        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False, file_path=None, generate=False, line_by_line=False, prepend_bos_token=False, text_json_key="text", prepended_text_to_remove=None):
    if generate:
        cs_len = args.gen_cs_len
        hs_len = args.gen_hs_len
        tis_len = args.gen_tis_len
    else:
        cs_len = args.cs_len
        hs_len = args.hs_len
        tis_len = args.tis_len
    
    if file_path is None:
        file_path = args.eval_data_file if evaluate else args.train_data_file
    
    if line_by_line:
        logger.info("Creating LineByLineTextDataset")
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size, prepend_bos_token=prepend_bos_token)
    else:
        if evaluate:
            logger.info("Creating JsonlCoconDataset for eval")
            return JsonlCoconDataset(tokenizer, args, file_path=file_path, block_size=args.block_size, text_json_key=text_json_key, cs_len=cs_len, hs_len=hs_len, tis_len=tis_len, evaluate=True, prepended_text_to_remove=prepended_text_to_remove)
        else:
            return JsonlCoconDataset(tokenizer, args, file_path=file_path, cs_len=cs_len, hs_len=hs_len, tis_len=tis_len)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def _clear_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)

    for checkpoint in checkpoints_sorted:
        logger.info("Deleting older checkpoint [{}] before rerunning training".format(checkpoint))
        shutil.rmtree(checkpoint)

# https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/22
def to_one_hot(y, n_dims=None, debug=False):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y
    y_tensor = y_tensor.type(torch.LongTensor).reshape(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)

    if debug:
        y_compare = torch.argmax(y_one_hot, dim=-1)
        logger.info( "y_compare: {}".format(y_compare)) 
        logger.info( "u: {}".format(y)) 
        
    return y_one_hot

def num_correct(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def train_cocon(args, train_dataset, model, tokenizer, cocon_block, disc_model=None, model_config=None, transform_h_after_layernorm=False):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_log_dir = os.path.join(args.output_dir, 'runs')
        tb_writer = SummaryWriter(tb_log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # check if max/min_hs_tis_split_offset is out of range of hs_len or tis_len
    offset_hs_tis_split = False
    if args.min_hs_tis_split_offset != 0:
        offset_hs_tis_split = True
        if (args.min_hs_tis_split_offset+args.hs_len < 0):
            raise ValueError(
                "min_hs_tis_split_offset is out of bound"
            )
    if args.max_hs_tis_split_offset != 0:
        offset_hs_tis_split = True
        if (min(args.cs_len, args.tis_len) - args.max_hs_tis_split_offset < 0):
            raise ValueError(
                "max_hs_tis_split_offset is out of bound"
            )

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay) for cocon_block
    no_decay = ["bias", "LayerNorm.weight"]
    cocon_block_optimizer_grouped_parameters = [
        {
            "params": [p for n, p in cocon_block.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in cocon_block.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    cocon_block_optimizer = AdamW(cocon_block_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    cocon_block_scheduler = get_linear_schedule_with_warmup(
        cocon_block_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.lambda_adv > 0:
        # Prepare optimizer and schedule (linear warmup and decay) for disc_model
        no_decay = ["bias", "LayerNorm.weight"]
        disc_model_optimizer_grouped_parameters = [
            {
                "params": [p for n, p in disc_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in disc_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        disc_model_optimizer = AdamW(disc_model_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        disc_model_scheduler = get_linear_schedule_with_warmup(
            disc_model_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    # Prepare optimizer and schedule (linear warmup and decay) for lm model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "cocon_block_optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "cocon_block_scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        cocon_block_optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "cocon_block_optimizer.pt")))
        cocon_block_scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "cocon_block_scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        cocon_block, cocon_block_optimizer = amp.initialize(cocon_block, cocon_block_optimizer, opt_level=args.fp16_opt_level)
        if args.lambda_adv > 0:
            disc_model, disc_model_optimizer = amp.initialize(disc_model, disc_model_optimizer, opt_level=args.fp16_opt_level)
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        cocon_block = torch.nn.DataParallel(cocon_block)
        if args.lambda_adv > 0:
            disc_model = torch.nn.DataParallel(disc_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        cocon_block = torch.nn.parallel.DistributedDataParallel(
            cocon_block, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        if args.lambda_adv > 0:
            disc_model = torch.nn.parallel.DistributedDataParallel(
                disc_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
            )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    disc_loss, logging_disc_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility

    start_self_cocon_lm = False
    start_hist_cocon_lm = False
    start_cycle_ar_cocon_recon = False
    start_other_context_cocon = False
    start_adv = False
    adv_gen_opt_step = 0
    adv_disc_opt_step = 0
    first_save = True
    for epoch_ind in train_iterator:
        logger.info( "epoch_ind: {}".format(epoch_ind))

        if args.lambda_cycle_ar_cocon_recon_lm_loss > 0 and args.per_gpu_train_cycle_ar_cocon_recon_batch_size is not None and epoch_ind == args.epoch_ind_to_start_cycle_ar_cocon_recon:
            args.train_cycle_ar_cocon_recon_batch_size = args.per_gpu_train_cycle_ar_cocon_recon_batch_size * max(1, args.n_gpu)
            logger.info( "Changing train_batch_size to {} due to start_cycle_ar_cocon_recon".format(args.train_cycle_ar_cocon_recon_batch_size))

            train_dataloader = DataLoader(
                train_dataset, sampler=train_sampler, batch_size=args.train_cycle_ar_cocon_recon_batch_size, collate_fn=collate
            )

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, lm_labels = (batch, batch)
            
            # Skip batch if seq len is shorter than hs_len, i.e. no tis or cs text
            if inputs.shape[1] < args.hs_len:
                logger.info("inputs.shape[1] < args.hs_len, skipping batch")
                continue

            # Split train samples into hs, tis, cs segments
            # variable split offset
            if offset_hs_tis_split:
                hs_tis_split_ind = random.randint(args.min_hs_tis_split_offset, args.max_hs_tis_split_offset)
                hs_len = args.hs_len + hs_tis_split_ind
                cs_len = args.cs_len - hs_tis_split_ind
                tis_len = args.tis_len - hs_tis_split_ind
            else:
                hs_len = args.hs_len
                cs_len = args.cs_len
                tis_len = args.tis_len

            lm_labels = lm_labels[:, :hs_len+tis_len]
            inputs = inputs.to(args.device)
            lm_labels = lm_labels.to(args.device)

            original_context_seq = inputs[:, hs_len:hs_len+cs_len]
            original_history_seq = inputs[:, :hs_len]
            original_transform_input_seq = inputs[:, hs_len:hs_len+tis_len]

            # use batch with + 1 index as other sample
            other_sample_inputs = torch.cat([inputs[-1:], inputs[:-1]], dim=0)
            other_sample_lm_labels = other_sample_inputs[:, :hs_len+tis_len]

            other_sample_context_seq = other_sample_inputs[:, hs_len:hs_len+cs_len]
            other_sample_history_seq = other_sample_inputs[:, :hs_len]
            other_sample_transform_input_seq = other_sample_inputs[:, hs_len:hs_len+tis_len]

            model.eval()
            
            cocon_block.train()
            if args.lambda_adv > 0:
                disc_model.train()
            
            if args.gradient_accumulation_steps == 1:
                # reset grad in model
                model.zero_grad()
                cocon_block.zero_grad()
                if args.lambda_adv > 0 and start_adv:
                    disc_model.zero_grad()

            with torch.no_grad():
                if transform_h_after_layernorm:                    
                    hidden_states = model(inputs, output_after_block_ind=args.output_hidden_for_cocon_after_block_ind, return_point='next_block_ln_1')[0]  # [N, L, C]
                    context_seq_hidden_states = model(original_context_seq, output_after_block_ind=args.output_hidden_for_cocon_after_block_ind, return_point='next_block_ln_1')[0]  # [N, L, C]
                else:
                    hidden_states = model(inputs, output_after_block_ind=args.output_hidden_for_cocon_after_block_ind)[0]  # [N, L, C]
                    context_seq_hidden_states = model(original_context_seq, output_after_block_ind=args.output_hidden_for_cocon_after_block_ind)[0]  # [N, L, C]

            original_hidden_states = hidden_states
            original_history_seq_hidden_states = original_hidden_states[:, :hs_len]
            original_transform_input_seq_hidden_states = original_hidden_states[:, hs_len:hs_len+tis_len]
            original_context_seq_hidden_states = context_seq_hidden_states

            # use batch with + 1 index as other sample
            other_sample_hidden_states = torch.cat([hidden_states[-1:], hidden_states[:-1]], dim=0)
            other_sample_history_seq_hidden_states = other_sample_hidden_states[:, :hs_len]
            other_sample_transform_input_seq_hidden_states = other_sample_hidden_states[:, hs_len:hs_len+tis_len]

            other_sample_context_seq_hidden_states = torch.cat([context_seq_hidden_states[-1:], context_seq_hidden_states[:-1]], dim=0)



            # self_cocon_lm_loss computation, CS: original_context_seq_hidden_states, HS: original_history_seq_hidden_states, TIS: original_transform_input_seq_hidden_states
            # single FF pass, no need for AR 
            # cs & tis mask computation
            if args.self_cocon_lm_cs_mask_prob > 0 or args.self_cocon_lm_tis_mask_prob > 0:
                if args.self_cocon_lm_mutual_exc_mask:
                    max_cs_tis_len = max(original_context_seq_hidden_states.shape[1], original_transform_input_seq_hidden_states.shape[1]) 
                    total_mask_prob = args.self_cocon_lm_cs_mask_prob + args.self_cocon_lm_tis_mask_prob
                    if total_mask_prob > 1:
                        logger.warning("self_cocon_lm_mask_prob > 1, bounding it to 1")
                        total_mask_prob = 1
                    
                    prob_matrix = torch.full([original_context_seq_hidden_states.shape[0], max_cs_tis_len], total_mask_prob)
                    all_masked_indices = torch.bernoulli(prob_matrix).bool()

                    prob_allocated_cs = args.self_cocon_lm_cs_mask_prob / total_mask_prob
                    # prob_allocated_tis = args.self_cocon_lm_tis_mask_prob / total_mask_prob
                    allocated_cs_prob_matrix = torch.full([original_context_seq_hidden_states.shape[0], max_cs_tis_len], prob_allocated_cs)
                    allocated_cs_indices = torch.bernoulli(allocated_cs_prob_matrix).bool()

                    cs_masked_indices = all_masked_indices & allocated_cs_indices
                    tis_masked_indices = all_masked_indices & ~allocated_cs_indices

                    if original_context_seq_hidden_states.shape[1] != max_cs_tis_len:
                        cs_masked_indices = cs_masked_indices[:, original_context_seq_hidden_states.shape[1]]
                    elif original_transform_input_seq_hidden_states.shape[1] != max_cs_tis_len:
                        tis_masked_indices = tis_masked_indices[:, original_transform_input_seq_hidden_states.shape[1]]

                else:
                    cs_prob_matrix = torch.full(original_context_seq_hidden_states.shape[:-1], args.self_cocon_lm_cs_mask_prob)
                    cs_masked_indices = torch.bernoulli(cs_prob_matrix).bool()
                    tis_prob_matrix = torch.full(original_transform_input_seq_hidden_states.shape[:-1], args.self_cocon_lm_tis_mask_prob)
                    tis_masked_indices = torch.bernoulli(tis_prob_matrix).bool()
            
                self_cocon_hidden_states = cocon_block(original_transform_input_seq_hidden_states, context_seq=original_context_seq_hidden_states, history_seq=original_history_seq_hidden_states, include_sos_output=True, cs_masked_indices=cs_masked_indices, tis_masked_indices=tis_masked_indices, cs_self_attn_mask_prob=args.self_token_mask_prob) # [N, L, C]        
            else:
                self_cocon_hidden_states = cocon_block(original_transform_input_seq_hidden_states, context_seq=original_context_seq_hidden_states, history_seq=original_history_seq_hidden_states, include_sos_output=True, cs_self_attn_mask_prob=args.self_token_mask_prob) # [N, L, C]

            # concat cocon output with original history_seq
            self_cocon_lm_tail_input = torch.cat([original_history_seq_hidden_states[:, :-1], self_cocon_hidden_states], dim=1)

            # compute lm loss only on cocon-transformed hidden_states
            lm_logit_first_index = original_history_seq_hidden_states.shape[1] -1
            lm_labels_first_index = lm_logit_first_index + 1
            
            # compute lm tail logits output and loss values
            if transform_h_after_layernorm:    
                self_cocon_lm_tail_outputs = model(input_hidden_state=self_cocon_lm_tail_input, labels=lm_labels, lm_logit_first_index=lm_logit_first_index, lm_labels_first_index=lm_labels_first_index, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1, input_point='current_block_ln_1')
            else:
                self_cocon_lm_tail_outputs = model(input_hidden_state=self_cocon_lm_tail_input, labels=lm_labels, lm_logit_first_index=lm_logit_first_index, lm_labels_first_index=lm_labels_first_index, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1)

            self_cocon_lm_loss = self_cocon_lm_tail_outputs[0]
                    
            if args.track_loss_gradnorms and args.local_rank in [-1, 0] and args.logging_steps > 0 and (global_step+1) % args.logging_steps == 0:
                self_cocon_lm_loss_grad = torch.autograd.grad(self_cocon_lm_loss, cocon_block.cocon_attn.c_attn.weight, retain_graph=True)[0]
                self_cocon_lm_loss_gradnorm = torch.norm(self_cocon_lm_loss_grad)

            if args.lambda_self_cocon_lm_loss > 0:
                total_loss = args.lambda_self_cocon_lm_loss * self_cocon_lm_loss
            else:
                total_loss = 0


            if args.lambda_hist_cocon_lm_loss > 0:
                # Check whether it is time to start adv training   
                if start_hist_cocon_lm == False and epoch_ind == args.epoch_ind_to_start_hist_cocon_lm and step == args.step_ind_to_start_hist_cocon_lm:
                    logger.info( "starting hist_cocon_lm_loss training") 
                    logger.info( "step_ind_to_start_hist_cocon_lm: {}, step: {}".format(args.step_ind_to_start_hist_cocon_lm, step)) 
                    logger.info( "epoch_ind_to_start_hist_cocon_lm: {}, epoch_ind: {}".format(args.epoch_ind_to_start_hist_cocon_lm, epoch_ind)) 
                    start_hist_cocon_lm = True

            if start_hist_cocon_lm or (args.track_hist_cocon_lm_loss and args.local_rank in [-1, 0] and args.logging_steps > 0 and (global_step+1) % args.logging_steps == 0):
                hist_cocon_hidden_states = cocon_block(original_transform_input_seq_hidden_states, context_seq=None, history_seq=original_history_seq_hidden_states, include_sos_output=True) # [N, L, C]

                # concat cocon output with original history_seq
                hist_cocon_lm_tail_input = torch.cat([original_history_seq_hidden_states[:, :-1], hist_cocon_hidden_states], dim=1)

                # compute lm loss only on cocon-transformed hidden_states
                lm_logit_first_index = original_history_seq_hidden_states.shape[1] -1
                lm_labels_first_index = lm_logit_first_index + 1
                
                # compute lm tail logits output and loss values
                if transform_h_after_layernorm:    
                    hist_cocon_lm_tail_outputs = model(input_hidden_state=hist_cocon_lm_tail_input, labels=lm_labels, lm_logit_first_index=lm_logit_first_index, lm_labels_first_index=lm_labels_first_index, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1, input_point='current_block_ln_1')
                else:
                    hist_cocon_lm_tail_outputs = model(input_hidden_state=hist_cocon_lm_tail_input, labels=lm_labels, lm_logit_first_index=lm_logit_first_index, lm_labels_first_index=lm_labels_first_index, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1)

                hist_cocon_lm_loss = hist_cocon_lm_tail_outputs[0]
                        
                if args.track_loss_gradnorms and args.local_rank in [-1, 0] and args.logging_steps > 0 and (global_step+1) % args.logging_steps == 0:
                    hist_cocon_lm_loss_grad = torch.autograd.grad(hist_cocon_lm_loss, cocon_block.cocon_attn.c_attn.weight, retain_graph=True)[0]
                    hist_cocon_lm_loss_gradnorm = torch.norm(hist_cocon_lm_loss_grad)

                if args.lambda_hist_cocon_lm_loss > 0:
                    total_loss += args.lambda_hist_cocon_lm_loss * hist_cocon_lm_loss


            if args.lambda_adv > 0:
                # Check whether it is time to start adv training   
                if start_adv == False and epoch_ind == args.epoch_ind_to_start_adv and step == args.step_ind_to_start_adv:
                    logger.info( "starting adversarial learning") 
                    logger.info( "step_ind_to_start_adv: {}, step: {}".format(args.step_ind_to_start_adv, step)) 
                    logger.info( "epoch_ind_to_start_adv: {}, epoch_ind: {}".format(args.epoch_ind_to_start_adv, epoch_ind)) 
                    start_adv = True

            # cycle_ar_cocon_recon_lm_loss computation, Step 1 CS: original_context_seq_hidden_states, HS: other_sample_history_seq_hidden_states, TIS: None (AR generation)
            if args.lambda_cycle_ar_cocon_recon_lm_loss > 0 and epoch_ind == args.epoch_ind_to_start_cycle_ar_cocon_recon and step == args.step_ind_to_start_cycle_ar_cocon_recon:
                logger.info( "starting cycle_ar_cocon_recon_lm learning") 
                logger.info( "step_ind_to_start_cycle_ar_cocon_recon: {}, step: {}".format(args.step_ind_to_start_cycle_ar_cocon_recon, step)) 
                logger.info( "epoch_ind_to_start_cycle_ar_cocon_recon: {}, epoch_ind: {}".format(args.epoch_ind_to_start_cycle_ar_cocon_recon, epoch_ind)) 
                start_cycle_ar_cocon_recon = True
                
            if start_cycle_ar_cocon_recon or start_adv:
                cur_len = 0
                cocon_block_output = None
                cocon_th_gen_input = None
                cocon_th_gen_output = None

                cocon_output_embeds = None
                lm_tail_past=None
                lm_head_past=None
                
                max_cocon_AR_length = min(args.max_cocon_AR_length, original_transform_input_seq_hidden_states.shape[1]) # limit cocon output length to hidden states' length

                other_sample_history_seq_one_hot_prob = to_one_hot(other_sample_history_seq, n_dims=model_config.vocab_size).to(args.device)
                other_sample_history_seq_embeds = torch.matmul(other_sample_history_seq_one_hot_prob, model.transformer.wte.weight)

                # cocon_th_gen_output: autoreg generation with CS- & HS-conditioned cocon op
                while cur_len < max_cocon_AR_length:
                    cocon_transformed_hidden_states = cocon_block(cocon_th_gen_input, context_seq=original_context_seq_hidden_states, history_seq=other_sample_history_seq_hidden_states, include_sos_output=True, cs_self_attn_mask_prob=args.cycle_self_token_mask_prob) # [N, L, C]

                    if cur_len == 0:
                        cocon_block_output = cocon_transformed_hidden_states[:, -1:]
                    else:
                        cocon_block_output = torch.cat([cocon_block_output, cocon_transformed_hidden_states[:, -1:]], dim=1)

                    if args.use_only_last_cocon_output_for_ar:
                        if cocon_th_gen_input is not None:
                            hist_plus_cocon_hidden_states = torch.cat([other_sample_history_seq_hidden_states, cocon_th_gen_input[:, :-1], cocon_transformed_hidden_states[:, -1:]], dim=1)
                        else:
                            hist_plus_cocon_hidden_states = torch.cat([other_sample_history_seq_hidden_states[:, :-1], cocon_transformed_hidden_states[:, -1:]], dim=1)
                    else:
                        hist_plus_cocon_hidden_states = torch.cat([other_sample_history_seq_hidden_states[:, :-1], cocon_transformed_hidden_states], dim=1)

                    # optimized tail computation
                    if transform_h_after_layernorm:    
                        lm_tail_inputs = model.prepare_hidden_state_inputs_for_generation(input_hidden_state=hist_plus_cocon_hidden_states, past=lm_tail_past, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1, input_point='current_block_ln_1')
                    else:
                        lm_tail_inputs = model.prepare_hidden_state_inputs_for_generation(input_hidden_state=hist_plus_cocon_hidden_states, past=lm_tail_past, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1)

                    tail_outputs = model(**lm_tail_inputs)
                    next_token_logits = tail_outputs[0]  # [N,L,C] where C is vocab_size
                    if next_token_logits.shape[1] > 1:
                        next_token_logits = next_token_logits[:, -1:]
                    lm_tail_past = tail_outputs[1]

                    if args.gen_gumbel_softmax:
                        next_cocon_output_prob = torch.nn.functional.gumbel_softmax(next_token_logits, dim=-1)
                    else:
                        next_cocon_output_prob = torch.nn.functional.softmax(next_token_logits, dim=-1)

                    next_cocon_output_embed = torch.matmul(next_cocon_output_prob, model.transformer.wte.weight) # [N, 1, C]
            
                    if cur_len == 0:
                        cocon_output_embeds = next_cocon_output_embed
                        hist_plus_cocon_output_embeds = torch.cat([other_sample_history_seq_embeds, next_cocon_output_embed], dim=1)
                    else:
                        cocon_output_embeds = torch.cat([cocon_output_embeds, next_cocon_output_embed], dim=1)
                        hist_plus_cocon_output_embeds = torch.cat([hist_plus_cocon_output_embeds, next_cocon_output_embed], dim=1)

                    # optimized head computation
                    if transform_h_after_layernorm:  
                        lm_head_inputs = model.prepare_embeds_inputs_for_generation(inputs_embeds=hist_plus_cocon_output_embeds, past=lm_head_past, input_ids=None, output_after_block_ind=args.output_hidden_for_cocon_after_block_ind, return_point='next_block_ln_1')
                    else:
                        lm_head_inputs = model.prepare_embeds_inputs_for_generation(inputs_embeds=hist_plus_cocon_output_embeds, past=lm_head_past, input_ids=None, output_after_block_ind=args.output_hidden_for_cocon_after_block_ind)  

                    head_outputs = model(**lm_head_inputs)
                    cocon_gen_output_h = head_outputs[0]
                    if cocon_gen_output_h.shape[1] > 1:
                        next_h = cocon_gen_output_h[:, -1:]
                    else:
                        next_h = cocon_gen_output_h
                    
                    lm_head_past = head_outputs[1]
                    if cur_len % args.train_cycle_detach_interval == 0:
                        h_to_cat_input = next_h.detach()
                    else:
                        h_to_cat_input = next_h

                    if cur_len == 0:
                        cocon_th_gen_input = h_to_cat_input
                        cocon_th_gen_output = next_h
                    else:
                        cocon_th_gen_input = torch.cat([cocon_th_gen_input, h_to_cat_input], dim=1)
                        cocon_th_gen_output = torch.cat([cocon_th_gen_output, next_h], dim=1)

                    cur_len = cocon_th_gen_input.shape[1]

                if start_cycle_ar_cocon_recon:
                    ar_cocon_final_output_embeds = cocon_output_embeds

                    if transform_h_after_layernorm:    
                        ar_cocon_output_hidden_states = model(input_ids=None, inputs_embeds=ar_cocon_final_output_embeds, output_after_block_ind=args.output_hidden_for_cocon_after_block_ind, return_point='next_block_ln_1')[0]  # [N, L, C]  
                    else:
                        ar_cocon_output_hidden_states = model(input_ids=None, inputs_embeds=ar_cocon_final_output_embeds, output_after_block_ind=args.output_hidden_for_cocon_after_block_ind)[0]  # [N, L, C]   

                    # tis mask computation
                    if args.cycle_ar_cocon_recon_lm_tis_mask_prob > 0:
                        tis_prob_matrix = torch.full(original_transform_input_seq_hidden_states.shape[:-1], args.cycle_ar_cocon_recon_lm_tis_mask_prob)
                        tis_masked_indices = torch.bernoulli(tis_prob_matrix).bool()

                        cycle_ar_cocon_recon_hidden_states = cocon_block(original_transform_input_seq_hidden_states, context_seq=ar_cocon_output_hidden_states, history_seq=original_history_seq_hidden_states, include_sos_output=True, tis_masked_indices=tis_masked_indices, cs_self_attn_mask_prob=args.cycle_self_token_mask_prob) # [N, L, C]
                    else:
                        cycle_ar_cocon_recon_hidden_states = cocon_block(original_transform_input_seq_hidden_states, context_seq=ar_cocon_output_hidden_states, history_seq=original_history_seq_hidden_states, include_sos_output=True, cs_self_attn_mask_prob=args.cycle_self_token_mask_prob) # [N, L, C]

                    # concat cocon output with original history_seq, replace original_history_seq_hidden_states' last hidden state with first element of cycle_ar_cocon_recon_hidden_states
                    cycle_ar_cocon_recon_lm_tail_input = torch.cat([original_history_seq_hidden_states[:, :-1], cycle_ar_cocon_recon_hidden_states], dim=1)

                    # compute lm loss only on cocon-transformed hidden_states
                    lm_logit_first_index = original_history_seq_hidden_states.shape[1] -1
                    lm_labels_first_index = lm_logit_first_index + 1
                    
                    # compute lm tail logits output and loss values
                    if transform_h_after_layernorm:    
                        cycle_ar_cocon_recon_lm_tail_outputs = model(input_hidden_state=cycle_ar_cocon_recon_lm_tail_input, labels=lm_labels, lm_logit_first_index=lm_logit_first_index, lm_labels_first_index=lm_labels_first_index, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1, input_point='current_block_ln_1')
                    else:
                        cycle_ar_cocon_recon_lm_tail_outputs = model(input_hidden_state=cycle_ar_cocon_recon_lm_tail_input, labels=lm_labels, lm_logit_first_index=lm_logit_first_index, lm_labels_first_index=lm_labels_first_index, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1)

                    cycle_ar_cocon_recon_lm_loss = cycle_ar_cocon_recon_lm_tail_outputs[0]
                            
                    if args.track_loss_gradnorms and args.local_rank in [-1, 0] and args.logging_steps > 0 and (global_step+1) % args.logging_steps == 0:
                        cycle_ar_cocon_recon_lm_loss_grad = torch.autograd.grad(cycle_ar_cocon_recon_lm_loss, cocon_block.cocon_attn.c_attn.weight, retain_graph=True)[0]
                        cycle_ar_cocon_recon_lm_loss_gradnorm = torch.norm(cycle_ar_cocon_recon_lm_loss_grad)

                    if args.lambda_cycle_ar_cocon_recon_lm_loss > 0:
                        total_loss += args.lambda_cycle_ar_cocon_recon_lm_loss * cycle_ar_cocon_recon_lm_loss




            # context_ar_cocon_lm_loss computation, Step 1 CS: other_sample_context_seq_hidden_states, HS: other_sample_history_seq_hidden_states, TIS: cocon_th_gen_output (AR generated)
            # cat other_sample_history_seq_embeds with ar_cocon_final_output_embeds
            # hist_plus_cocon_output_hidden_states = torch.cat([other_sample_history_seq_hidden_states, cocon_th_gen_output], dim=1)
            if args.lambda_other_context_cocon_lm_loss > 0 and epoch_ind == args.epoch_ind_to_start_other_context_cocon and step == args.step_ind_to_start_other_context_cocon:
                logger.info( "starting cycle_ar_cocon_recon_lm learning") 
                logger.info( "step_ind_to_start_other_context_cocon: {}, step: {}".format(args.step_ind_to_start_other_context_cocon, step)) 
                logger.info( "epoch_ind_to_start_other_context_cocon: {}, epoch_ind: {}".format(args.epoch_ind_to_start_other_context_cocon, epoch_ind)) 
                start_other_context_cocon = True
                
            if start_other_context_cocon:
                other_context_cocon_hidden_states = cocon_block(cocon_th_gen_output, context_seq=original_context_seq_hidden_states, history_seq=other_sample_history_seq_hidden_states, include_sos_output=True, cs_self_attn_mask_prob=args.other_context_self_token_mask_prob) # [N, L, C]

                # concat cocon output with original history_seq
                other_context_cocon_lm_tail_input = torch.cat([other_sample_history_seq_hidden_states[:, :-1], other_context_cocon_hidden_states], dim=1)

                # compute lm loss only on cocon-transformed hidden_states
                lm_logit_first_index = other_sample_history_seq_hidden_states.shape[1] -1
                lm_labels_first_index = lm_logit_first_index + 1
                
                # compute lm tail logits output and loss values
                if transform_h_after_layernorm:    
                    other_context_cocon_lm_tail_outputs = model(input_hidden_state=other_context_cocon_lm_tail_input, labels=other_sample_lm_labels, lm_logit_first_index=lm_logit_first_index, lm_labels_first_index=lm_labels_first_index, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1, input_point='current_block_ln_1')
                else:
                    other_context_cocon_lm_tail_outputs = model(input_hidden_state=other_context_cocon_lm_tail_input, labels=other_sample_lm_labels, lm_logit_first_index=lm_logit_first_index, lm_labels_first_index=lm_labels_first_index, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1)

                other_context_cocon_lm_loss = other_context_cocon_lm_tail_outputs[0]
                    
                if args.track_loss_gradnorms and args.local_rank in [-1, 0] and args.logging_steps > 0 and (global_step+1) % args.logging_steps == 0:
                    other_context_cocon_lm_loss_grad = torch.autograd.grad(other_context_cocon_lm_loss, cocon_block.cocon_attn.c_attn.weight, retain_graph=True)[0]
                    other_context_cocon_lm_loss_gradnorm = torch.norm(other_context_cocon_lm_loss_grad)
                
                if args.lambda_other_context_cocon_lm_loss > 0:
                    total_loss += args.lambda_other_context_cocon_lm_loss * other_context_cocon_lm_loss


            if args.lambda_adv > 0:
                # Compute adv LOSS :  cocon_th_gen_output or cocon_block_output for adv training
                if start_adv:
                    if args.adv_use_th_gen_output:
                        disc_fake_input = cocon_th_gen_output
                    else:
                        disc_fake_input = cocon_block_output
                    disc_real_input = original_transform_input_seq_hidden_states

                    # detach for disc opt step later
                    if step % args.disc_update_interval == 0:
                        disc_fake_input_detached = disc_fake_input.detach()
                        disc_real_input_detached = disc_real_input.detach()

                    if step % args.gen_update_interval == 0:
                        # Adversarial cocon training step: train GEN
                        real_disc_label = torch.ones(hidden_states.shape[0], 1).to(args.device)
                        fake_disc_label = torch.zeros(hidden_states.shape[0], 1).to(args.device)
                        
                        d_fake_loss, d_fake_logits = disc_model(disc_fake_input, fake_disc_label)
                        d_real_loss, d_real_logits = disc_model(disc_real_input, real_disc_label)

                        if args.track_loss_gradnorms and args.local_rank in [-1, 0] and args.logging_steps > 0 and (global_step+1) % args.logging_steps == 0:
                            adv_loss_grad = torch.autograd.grad((-1*d_fake_loss - d_real_loss), cocon_block.cocon_attn.c_attn.weight, retain_graph=True)[0]
                            adv_loss_gradnorm = torch.norm(adv_loss_grad)
                        
                        if args.lambda_adv > 0:
                            total_loss += args.lambda_adv * ( -1*d_fake_loss - d_real_loss)
                        adv_gen_opt_step += 1


            if args.n_gpu > 1:
                total_loss = total_loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                total_loss = total_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(total_loss, cocon_block_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()

            tr_loss += total_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(cocon_block_optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(cocon_block.parameters(), args.max_grad_norm)

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and (global_step+1) % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, (global_step+1))

                    tb_writer.add_scalar("LR/lm_lr", cocon_block_scheduler.get_lr()[0], (global_step+1))
                    tb_writer.add_scalar("LOSS/total_loss", (tr_loss - logging_loss) / args.logging_steps, (global_step+1))

                    tb_writer.add_scalar("LOSS/self_cocon_lm_loss", self_cocon_lm_loss.item(), (global_step+1))
                    
                    if args.lambda_hist_cocon_lm_loss > 0 or args.track_hist_cocon_lm_loss:
                        tb_writer.add_scalar("LOSS/self_cocon_lm_loss", self_cocon_lm_loss.item(), (global_step+1))

                    if (args.lambda_hist_cocon_lm_loss > 0 and start_hist_cocon_lm) or args.track_hist_cocon_lm_loss:
                        tb_writer.add_scalar("LOSS/hist_cocon_lm_loss", hist_cocon_lm_loss, (global_step+1))
                        
                    if args.lambda_adv > 0 and start_adv:
                        tb_writer.add_scalar("LOSS/d_fake_loss", d_fake_loss.item(), (global_step+1))
                        tb_writer.add_scalar("LOSS/d_real_loss", d_real_loss.item(), (global_step+1))

                    if start_cycle_ar_cocon_recon:
                        tb_writer.add_scalar("LOSS/cycle_ar_cocon_recon_lm_loss", cycle_ar_cocon_recon_lm_loss.item(), (global_step+1))

                    if start_other_context_cocon:
                        tb_writer.add_scalar("LOSS/other_context_cocon_lm_loss", other_context_cocon_lm_loss.item(), (global_step+1))

                    tb_writer.add_scalar("GRADIENT/TOTALLOSS_model_attn_c_attn_gradnorm", torch.norm(cocon_block.cocon_attn.c_attn.weight.grad), (global_step+1))
                    if args.track_loss_gradnorms:
                        tb_writer.add_scalar("GRADIENT/self_cocon_lm_loss_gradnorm", self_cocon_lm_loss_gradnorm, (global_step+1))

                        if (args.lambda_hist_cocon_lm_loss > 0 and start_hist_cocon_lm) or args.track_hist_cocon_lm_loss:
                            tb_writer.add_scalar("GRADIENT/hist_cocon_lm_loss_gradnorm", hist_cocon_lm_loss_gradnorm, (global_step+1))
                        if start_cycle_ar_cocon_recon:
                            tb_writer.add_scalar("GRADIENT/cycle_ar_cocon_recon_lm_loss_gradnorm", cycle_ar_cocon_recon_lm_loss_gradnorm, (global_step+1))
                        if start_other_context_cocon:
                            tb_writer.add_scalar("GRADIENT/other_context_cocon_lm_loss_gradnorm", other_context_cocon_lm_loss_gradnorm, (global_step+1))                            
                        if start_adv:
                            tb_writer.add_scalar("GRADIENT/adv_loss_gradnorm", adv_loss_gradnorm, (global_step+1))
                        
                    logging_loss = tr_loss

                cocon_block_optimizer.step() # opt.step() does not zero_grad, need to zero.grad() manually 
                cocon_block_scheduler.step() # Update learning rate schedule
                

                cocon_block.zero_grad()
                model.zero_grad()


            # Adv DISC opt step
            if step % args.disc_update_interval == 0 and start_adv:
                if args.gradient_accumulation_steps == 1:
                    disc_model.zero_grad()

                # Adversarial cocon training step: train DISC
                real_disc_label = torch.ones(hidden_states.shape[0], 1).to(args.device)
                fake_disc_label = torch.zeros(hidden_states.shape[0], 1).to(args.device)
                d_fake_loss, d_fake_logits = disc_model(disc_fake_input_detached, fake_disc_label)
                d_real_loss, d_real_logits = disc_model(disc_real_input_detached, real_disc_label)
                total_disc_loss = d_fake_loss + d_real_loss

                disc_fake_num_correct = torch.sum(d_fake_logits < 0)
                disc_fake_acc = disc_fake_num_correct.type(torch.float64) / len(fake_disc_label)
                disc_real_num_correct = torch.sum(d_real_logits > 0)
                disc_real_acc = disc_real_num_correct.type(torch.float64) / len(real_disc_label)
            
                if args.n_gpu > 1:
                    total_disc_loss = total_disc_loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    total_disc_loss = total_disc_loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(total_disc_loss, disc_model_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_disc_loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(disc_model_optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(disc_model.parameters(), args.max_grad_norm)
                if args.local_rank in [-1, 0]: 
                    if args.logging_steps > 0 and step % args.logging_steps == 0: # to sync with disc update step
                        tb_writer.add_scalar("LR/disc_lr", disc_model_scheduler.get_lr()[0], (global_step+1))
                        tb_writer.add_scalar("LOSS/disc_loss", total_disc_loss.item(), (global_step+1))

                        tb_writer.add_scalar("LOSS/disc_fake_loss", d_fake_loss.item(), (global_step+1))
                        tb_writer.add_scalar("LOSS/disc_real_loss", d_real_loss.item(), (global_step+1))

                        tb_writer.add_scalar("GRADIENT/DISCLOSS_discmodel_conv3_gradnorm", torch.norm(disc_model.conv_layers[2].weight.grad), (global_step+1))

                        tb_writer.add_scalar("ACC/disc_fake_acc", disc_fake_acc, (global_step+1))
                        tb_writer.add_scalar("ACC/disc_real_acc", disc_real_acc, (global_step+1))

                        disc_loss = logging_disc_loss
                    elif (adv_disc_opt_step * args.disc_update_interval) < args.steps_to_closely_monitor_adv and step % (args.logging_steps // 50) == 0:
                        tb_writer.add_scalar("LR/disc_lr", disc_model_scheduler.get_lr()[0], (global_step+1))
                        tb_writer.add_scalar("LOSS/disc_loss", total_disc_loss.item(), (global_step+1))

                        tb_writer.add_scalar("LOSS/disc_fake_loss", d_fake_loss.item(), (global_step+1))
                        tb_writer.add_scalar("LOSS/disc_real_loss", d_real_loss.item(), (global_step+1))

                        tb_writer.add_scalar("GRADIENT/DISCLOSS_discmodel_conv3_gradnorm", torch.norm(disc_model.conv_layers[2].weight.grad), (global_step+1))

                        tb_writer.add_scalar("ACC/disc_fake_acc", disc_fake_acc, (global_step+1))
                        tb_writer.add_scalar("ACC/disc_real_acc", disc_real_acc, (global_step+1))


                disc_model_optimizer.step()
                disc_model_scheduler.step()  # Update learning rate schedule
                
                disc_model.zero_grad()
                adv_disc_opt_step += 1


            # Save model
            if args.local_rank in [-1, 0] and args.save_steps > 0 and step % args.save_steps == 0: # to sync with gen/disc update step          
                checkpoint_prefix = "cocon_block_checkpoint"
                if first_save:
                    _clear_checkpoints(args, checkpoint_prefix)
                    first_save = False

                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, (global_step+1)))
                os.makedirs(output_dir, exist_ok=True)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving cocon_block model checkpoint to %s", output_dir)

                _rotate_checkpoints(args, checkpoint_prefix)

                cocon_block_weights_name = "cocon_block_pytorch_model.bin"
                output_cocon_block_model_file = os.path.join(output_dir, cocon_block_weights_name)
                torch.save(cocon_block.state_dict(), output_cocon_block_model_file)
                logger.info("cocon_block model weights saved in {}".format(output_cocon_block_model_file))

                torch.save(cocon_block_optimizer.state_dict(), os.path.join(output_dir, "cocon_block_optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "cocon_block_scheduler.pt"))
                logger.info("Saving cocon_block optimizer and scheduler states to %s", output_dir)

            global_step += 1

            if (args.max_steps > 0 and global_step > args.max_steps) or (args.epoch_max_steps > 0 and step > args.epoch_max_steps):
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step





def train_lm(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    if args.per_gpu_train_lm_batch_size <= 0:
        args.per_gpu_train_lm_batch_size = args.per_gpu_train_batch_size
    args.train_lm_batch_size = args.per_gpu_train_lm_batch_size * max(1, args.n_gpu)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_lm_batch_size, collate_fn=collate
    )

    if args.lm_max_steps > 0:
        t_total = args.lm_max_steps
        args.num_lm_train_epochs = args.lm_max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_lm_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running LM training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_lm_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_lm_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_lm_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    first_save = True
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, labels = (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)


            if args.output_meanvars:
                all_meanvars = outputs[-1] 
                all_meanvars_tensor = []

                for block_ind, meanvars_in_block in enumerate(all_meanvars):
                    for layer_ind, meanvars_in_layer in enumerate(meanvars_in_block):
                        for stats_ind, stats in enumerate(meanvars_in_layer): # stats.shape: [batch_size, n_embd], mean & var
                            all_meanvars_tensor.append(stats)

                all_meanvars = torch.stack(all_meanvars_tensor, dim=1)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("LM/lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("LM/loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    if first_save:
                        _clear_checkpoints(args, checkpoint_prefix)
                        first_save = False

                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving LM model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving LM optimizer and scheduler states to %s", output_dir)

            if args.lm_max_steps > 0 and global_step > args.lm_max_steps:
                epoch_iterator.close()
                break
        if args.lm_max_steps > 0 and global_step > args.lm_max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


# evaluate perplexity
def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True, text_json_key=args.text_json_key, prepended_text_to_remove=args.prepended_text_to_remove)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        if labels.shape[1] < 2:
            continue
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()

        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, args.eval_output_filename)
    with open(output_eval_file, "w") as writer:
        logger.info("***** PPL Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def count_ngram(text_samples, n, tokenizer=None):
    """
    Count the number of unique n-grams
    :param text_samples: list, a list of samples
    :param n: int, n-gram
    :return: the number of unique n-grams in text_samples
    """
    if len(text_samples) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    ngram = set()
    for sample in text_samples:
        if len(sample) < n:
            continue

        sample = list(map(str, sample))
        for i in range(len(sample) - n + 1):
            ng = ' '.join(sample[i: i + n])

            ngram.add(' '.join(ng))
    return len(ngram)

# evaluate Dist-K scores
def evaluate_dist_scores(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True, text_json_key=args.text_json_key, prepended_text_to_remove=args.prepended_text_to_remove)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    dist_eval_samples = []
    num_tokens = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        sample_flattened = batch.reshape(-1)
        dist_eval_samples.append(sample_flattened.tolist())
        num_tokens += len(sample_flattened)

        nb_eval_steps += 1

        if nb_eval_steps == args.dist_eval_max_samples:
            logger.info("breaking iteration @ sample # {}".format(nb_eval_steps))
            break 

    dist1_score = count_ngram(dist_eval_samples, 1) / float(num_tokens)
    dist2_score = count_ngram(dist_eval_samples, 2) / float(num_tokens)
    dist3_score = count_ngram(dist_eval_samples, 3) / float(num_tokens)

    result = {"Dist-1": dist1_score, "Dist-2": dist2_score, "Dist-3": dist3_score}
    
    output_filename = "distK_" + args.eval_output_filename
    output_eval_file = os.path.join(eval_output_dir, prefix, output_filename)
    with open(output_eval_file, "w") as writer:
        logger.info("***** Dist-1,2,3 Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result
    
def fix_state_dict_naming(state_dict):
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        if 'con2' in key:
            new_key = key.replace('con2', 'cocon')
        # new_key = key_transformation(key)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict

# Use to generate cocon-edited text with either trained or simple cocon op
def generate_cocon_compute(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, cocon_block=None, prefix="", 
    random_sample_data=False, use_only_first_context_source_batch=False, use_only_first_custom_mu_s_input_batch=False, transform_h_after_layernorm=False, prepend_history_seq=True) -> Dict:

    eval_output_dir = args.output_dir

    cocon_output_file_path = os.path.join(args.output_dir, args.cocon_output_filename)
    if os.path.exists(cocon_output_file_path):
        if args.append_cocon_output_files:
            logger.info("Append to existing cocon output file")
        else:
            logger.info("Removing existing cocon output file")
            os.remove(cocon_output_file_path)
    else:
        logger.info("Creating new cocon output file")

    if args.cocon_output_jsonl_filename is not None:
        cocon_output_jsonl_file_path = os.path.join(args.output_dir, args.cocon_output_jsonl_filename)
        if os.path.exists(cocon_output_jsonl_file_path):
            if args.append_cocon_output_files:
                logger.info("Append to existing cocon output jsonl file")
            else:
                logger.info("Removing existing cocon output jsonl file")
                os.remove(cocon_output_jsonl_file_path)
        else:
            logger.info("Creating new cocon output jsonl file")
    else:
        cocon_output_jsonl_file_path = None
            
    if args.line_by_line_hs:
        history_source_dataset = load_and_cache_examples(args, tokenizer, file_path=args.cocon_compute_history_source_data_file, generate=True, line_by_line=True, prepend_bos_token=args.prepend_bos_token_to_line)
    else:
        history_source_dataset = load_and_cache_examples(args, tokenizer, file_path=args.cocon_compute_history_source_data_file, generate=True)
    
    if args.line_by_line_cs:
        context_source_dataset = load_and_cache_examples(args, tokenizer, file_path=args.cocon_compute_context_source_data_file, generate=True, line_by_line=True)
    else:
        context_source_dataset = load_and_cache_examples(args, tokenizer, file_path=args.cocon_compute_context_source_data_file, generate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    if random_sample_data == True:
        history_source_sampler = RandomSampler(history_source_dataset) if args.local_rank == -1 else DistributedSampler(history_source_dataset)
        context_source_sampler = RandomSampler(context_source_dataset) if args.local_rank == -1 else DistributedSampler(context_source_dataset)
    else:
        history_source_sampler = SequentialSampler(history_source_dataset)
        context_source_sampler = SequentialSampler(context_source_dataset)

    history_source_dataloader = DataLoader(
        history_source_dataset, sampler=history_source_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )
    context_source_dataloader = DataLoader(
        context_source_dataset, sampler=context_source_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )
    context_source_dataloader_iter = iter(context_source_dataloader)
            
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Generate cocon samples!
    logger.info("***** Running cocon generation {} *****".format(prefix))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_generate_cocon_steps = 0
    model.eval()
    cocon_block.eval()

    if use_only_first_context_source_batch and args.use_history_source_as_context_source_for_gen == False:
        context_source_batch = next(context_source_dataloader_iter)
        context_source_inputs = context_source_batch

    for batch_ind, batch in enumerate(tqdm(history_source_dataloader, desc="Generating")):
        inputs = batch

        if args.use_history_source_as_context_source_for_gen:
            history_source_inputs = inputs[:(inputs.shape[0] // 2)]
            context_source_inputs = inputs[(inputs.shape[0] // 2):]
            inputs = history_source_inputs
        inputs = inputs.to(args.device)


        if args.line_by_line_hs:
            original_history_seq = inputs
            original_context_seq = None
            original_transform_input_seq = None
        else:
            original_history_seq = inputs[:, :args.gen_hs_len]
            original_context_seq = inputs[:, args.gen_hs_len:args.gen_hs_len+args.gen_cs_len]
            original_transform_input_seq = inputs[:, args.gen_hs_len:args.gen_hs_len+args.gen_tis_len]

        if use_only_first_context_source_batch == False and args.use_history_source_as_context_source_for_gen == False:
            if args.enumerate_all_cs_for_each_hs:
                for context_batch_ind, context_source_batch in enumerate(tqdm(context_source_dataloader, desc="Enumerating context source")):
                    context_source_inputs = context_source_batch

                    if args.line_by_line_cs:
                        context_seq = context_source_inputs
                    else:
                        context_seq = context_source_inputs[:, args.gen_hs_len:args.gen_hs_len+args.gen_cs_len]
                    context_seq = context_seq.to(args.device)

                    with open(cocon_output_file_path, "a", encoding='utf-8') as f:
                        f.writelines("***HS #{},    CS #{}***\n".format(batch_ind, context_batch_ind))
                    generate_cocon_sample(context_seq, original_history_seq, original_context_seq, inputs, cocon_output_file_path, args, model, tokenizer, cocon_block, 
                        cocon_output_jsonl_file_path=cocon_output_jsonl_file_path, transform_h_after_layernorm=transform_h_after_layernorm, prepend_history_seq=prepend_history_seq)

                # go to next hs
                continue

            else:
                context_source_batch = next(context_source_dataloader_iter)
                context_source_inputs = context_source_batch

        if args.line_by_line_cs:
            context_seq = context_source_inputs
        else:
            context_seq = context_source_inputs[:, args.gen_hs_len:args.gen_hs_len+args.gen_cs_len]
        context_seq = context_seq.to(args.device)

        with open(cocon_output_file_path, "a", encoding='utf-8') as f:
            f.writelines("***HS #{}***\n".format(batch_ind))

        generate_cocon_sample(context_seq, original_history_seq, original_context_seq, inputs, cocon_output_file_path, args, model, tokenizer, cocon_block, 
            cocon_output_jsonl_file_path=cocon_output_jsonl_file_path, transform_h_after_layernorm=transform_h_after_layernorm, prepend_history_seq=prepend_history_seq)


        if nb_generate_cocon_steps >= args.num_cocon_generate - 1:
            break


        nb_generate_cocon_steps += 1

    return nb_generate_cocon_steps


def generate_cocon_sample(context_seq, original_history_seq, original_context_seq, inputs, cocon_output_file_path, args, model, tokenizer, cocon_block, 
    cocon_output_jsonl_file_path=None, transform_h_after_layernorm=False, prepend_history_seq=True, 
    original_dia_history_seq=None, dia_context_seq=None, original_dia_context_seq=None, end_of_text_id=None, single_generation=False, do_cocon_wgpt2genas2ndcs=True, wgpt2genas2ndcs_double_context_len=30,
    cocon_wgpt2genas2ndcs_cs_attn_biases=[1, 2, 5, 10], cocon_wgpt2genas2ndcs_gpt2out_attn_biases=[-1, -2, -5, -10]):

    with torch.no_grad():
        encoded_prompt = inputs[:, 0:0]

        # Cocon generation with context_seq as cs
        cocon_gen_ar_output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=args.generate_length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
            do_cocon=True,
            cocon_block=cocon_block,
            cocon_context_inputs=context_seq,
            cocon_history_inputs=original_history_seq,
            cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind, 
            transform_h_after_layernorm=transform_h_after_layernorm,
            use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
        )
        if prepend_history_seq:
            cocon_gen_ar_output_sequences = torch.cat([original_history_seq, cocon_gen_ar_output_sequences], dim=1)
        # Remove the batch dimension when returning multiple sequences
        if len(cocon_gen_ar_output_sequences.shape) > 2:
            cocon_gen_ar_output_sequences.squeeze_()



        if args.context_attn_bias != 0:
            # Cocon generation with context_seq as cs, with context_attn_bias
            cocon_gen_conbias_ar_output_sequences = model.generate(
                input_ids=encoded_prompt,
                max_length=args.generate_length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
                do_cocon=True,
                cocon_block=cocon_block,
                cocon_context_inputs=context_seq,
                cocon_history_inputs=original_history_seq,
                cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind, 
                transform_h_after_layernorm=transform_h_after_layernorm,
                use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
                context_attn_bias=args.context_attn_bias
            )
            if prepend_history_seq:
                cocon_gen_conbias_ar_output_sequences = torch.cat([original_history_seq, cocon_gen_conbias_ar_output_sequences], dim=1)
            # Remove the batch dimension when returning multiple sequences
            if len(cocon_gen_conbias_ar_output_sequences.shape) > 2:
                cocon_gen_conbias_ar_output_sequences.squeeze_()

        # Cocon generation with original_context_seq as cs
        if args.line_by_line_hs == False and original_context_seq is not None:
            self_cocon_gen_ar_output_sequences = model.generate(
                input_ids=encoded_prompt,
                max_length=args.generate_length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
                do_cocon=True,
                cocon_block=cocon_block,
                cocon_context_inputs=original_context_seq,
                cocon_history_inputs=original_history_seq,
                cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind, 
                transform_h_after_layernorm=transform_h_after_layernorm,
                use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
            )
            if prepend_history_seq:
                self_cocon_gen_ar_output_sequences = torch.cat([original_history_seq, self_cocon_gen_ar_output_sequences], dim=1)
            # Remove the batch dimension when returning multiple sequences
            if len(self_cocon_gen_ar_output_sequences.shape) > 2:
                self_cocon_gen_ar_output_sequences.squeeze_()

        # Prepend Context GPT-2 generation
        # Sanity check: autoregressive text generation with context_seq prepended on the prompt
        encoded_prompt = torch.cat([context_seq, original_history_seq], dim=1)
        prependgpt2_gen_ar_output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=args.generate_length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
        )


        # Original GPT-2 generation
        # Sanity check: autoregressive text generation
        encoded_prompt = original_history_seq
        gen_ar_output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=args.generate_length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
        )
        # Remove the batch dimension when returning multiple sequences
        if len(gen_ar_output_sequences.shape) > 2:
            gen_ar_output_sequences.squeeze_()

        # Cocon generation (wgpt2genas2ndcs): with gpt2 generations as 2nd context sequence
        if do_cocon_wgpt2genas2ndcs:
            gpt2_gen_output = gen_ar_output_sequences[ :, len(original_history_seq[0]): ]
            cocon_wgpt2genas2ndcs_context_input = [context_seq, gpt2_gen_output]
            cocon_wgpt2genas2ndcs_context_attn_bias = [0, 0]
            encoded_prompt = inputs[:, 0:0]
            cocon_wgpt2genas2ndcs_gen_ar_output_sequences = model.generate(
                input_ids=encoded_prompt,
                max_length=args.generate_length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
                do_cocon=True,
                cocon_block=cocon_block,
                cocon_context_inputs=cocon_wgpt2genas2ndcs_context_input,
                cocon_history_inputs=original_history_seq,
                cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind, 
                transform_h_after_layernorm=transform_h_after_layernorm,
                use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
                context_attn_bias=cocon_wgpt2genas2ndcs_context_attn_bias,
            )
            if prepend_history_seq:
                cocon_wgpt2genas2ndcs_gen_ar_output_sequences = torch.cat([original_history_seq, cocon_wgpt2genas2ndcs_gen_ar_output_sequences], dim=1)
            # Remove the batch dimension when returning multiple sequences
            if len(cocon_wgpt2genas2ndcs_gen_ar_output_sequences.shape) > 2:
                cocon_wgpt2genas2ndcs_gen_ar_output_sequences.squeeze_()


            # Cocon generation (wgpt2genas2ndcs): with gpt2 generations as 2nd context sequence, with double context generation cut off: cocon with one context after wgpt2genas2ndcs_double_context_len for better generation quality
            gpt2_gen_output = gen_ar_output_sequences[ :, len(original_history_seq[0]): ]
            cocon_wgpt2genas2ndcs_2parts1st_context_input = [context_seq, gpt2_gen_output]
            encoded_prompt = inputs[:, 0:0]
            # Part 1 generation
            cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences = model.generate(
                input_ids=encoded_prompt,
                max_length=wgpt2genas2ndcs_double_context_len + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
                do_cocon=True,
                cocon_block=cocon_block,
                cocon_context_inputs=cocon_wgpt2genas2ndcs_2parts1st_context_input,
                cocon_history_inputs=original_history_seq,
                cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind, 
                transform_h_after_layernorm=transform_h_after_layernorm,
                use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
                context_attn_bias=cocon_wgpt2genas2ndcs_context_attn_bias,
            )
            if prepend_history_seq:
                cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences = torch.cat([original_history_seq, cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences], dim=1)
            # Remove the batch dimension when returning multiple sequences
            if len(cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences.shape) > 2:
                cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences.squeeze_()

            encoded_prompt = inputs[:, 0:0]
            # Part 2 generation: with only original context_seq as context input
            cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences = model.generate(
                input_ids=encoded_prompt,
                max_length=args.generate_length - wgpt2genas2ndcs_double_context_len + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
                do_cocon=True,
                cocon_block=cocon_block,
                cocon_context_inputs=context_seq,
                cocon_history_inputs=cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences,
                cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind, 
                transform_h_after_layernorm=transform_h_after_layernorm,
                use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
            )
            if prepend_history_seq:
                cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences = torch.cat([cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences, cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences], dim=1)
            # Remove the batch dimension when returning multiple sequences
            if len(cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences.shape) > 2:
                cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences.squeeze_()


            # With varying cs context_attn_bias values, Cocon generation (wgpt2genas2ndcs): with gpt2 generations as 2nd context sequence, with double context generation cut off: cocon with one context after wgpt2genas2ndcs_double_context_len
            cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences_list = []
            for cs_attn_bias in cocon_wgpt2genas2ndcs_cs_attn_biases:
                cocon_wgpt2genas2ndcs_context_attn_bias = [cs_attn_bias, 0]
            
                gpt2_gen_output = gen_ar_output_sequences[ :, len(original_history_seq[0]): ]
                cocon_wgpt2genas2ndcs_2parts1st_context_input = [context_seq, gpt2_gen_output]
                encoded_prompt = inputs[:, 0:0]
                # Part 1 generation
                cs_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences = model.generate(
                    input_ids=encoded_prompt,
                    max_length=wgpt2genas2ndcs_double_context_len + len(encoded_prompt[0]),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    num_return_sequences=args.num_return_sequences,
                    do_cocon=True,
                    cocon_block=cocon_block,
                    cocon_context_inputs=cocon_wgpt2genas2ndcs_2parts1st_context_input,
                    cocon_history_inputs=original_history_seq,
                    cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind, 
                    transform_h_after_layernorm=transform_h_after_layernorm,
                    use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
                    context_attn_bias=cocon_wgpt2genas2ndcs_context_attn_bias,
                )
                if prepend_history_seq:
                    cs_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences = torch.cat([original_history_seq, cs_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences], dim=1)

                encoded_prompt = inputs[:, 0:0]
                # Part 2 generation: with only original context_seq as context input
                cs_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences = model.generate(
                    input_ids=encoded_prompt,
                    max_length=args.generate_length - wgpt2genas2ndcs_double_context_len + len(encoded_prompt[0]),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    num_return_sequences=args.num_return_sequences,
                    do_cocon=True,
                    cocon_block=cocon_block,
                    cocon_context_inputs=context_seq,
                    cocon_history_inputs=cs_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences,
                    cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind, 
                    transform_h_after_layernorm=transform_h_after_layernorm,
                    use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
                )
                if prepend_history_seq:
                    cs_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences = torch.cat([cs_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences, cs_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences], dim=1)
                # Remove the batch dimension when returning multiple sequences
                if len(cs_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences.shape) > 2:
                    cs_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences.squeeze_()
                cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences_list.append(cs_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences)


            # With varying gpt2 output context_attn_bias values, Cocon generation (wgpt2genas2ndcs): with gpt2 generations as 2nd context sequence, with double context generation cut off: cocon with one context after wgpt2genas2ndcs_double_context_len
            gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences_list = []
            for gpt2out_attn_bias in cocon_wgpt2genas2ndcs_gpt2out_attn_biases:
                cocon_wgpt2genas2ndcs_context_attn_bias = [0, gpt2out_attn_bias]
            
                gpt2_gen_output = gen_ar_output_sequences[ :, len(original_history_seq[0]): ]
                cocon_wgpt2genas2ndcs_2parts1st_context_input = [context_seq, gpt2_gen_output]
                encoded_prompt = inputs[:, 0:0]
                # Part 1 generation
                gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences = model.generate(
                    input_ids=encoded_prompt,
                    max_length=wgpt2genas2ndcs_double_context_len + len(encoded_prompt[0]),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    num_return_sequences=args.num_return_sequences,
                    do_cocon=True,
                    cocon_block=cocon_block,
                    cocon_context_inputs=cocon_wgpt2genas2ndcs_2parts1st_context_input,
                    cocon_history_inputs=original_history_seq,
                    cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind, 
                    transform_h_after_layernorm=transform_h_after_layernorm,
                    use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
                    context_attn_bias=cocon_wgpt2genas2ndcs_context_attn_bias,
                )
                if prepend_history_seq:
                    gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences = torch.cat([original_history_seq, gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences], dim=1)

                encoded_prompt = inputs[:, 0:0]
                # Part 2 generation: with only original context_seq as context input
                gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences = model.generate(
                    input_ids=encoded_prompt,
                    max_length=args.generate_length - wgpt2genas2ndcs_double_context_len + len(encoded_prompt[0]),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    num_return_sequences=args.num_return_sequences,
                    do_cocon=True,
                    cocon_block=cocon_block,
                    cocon_context_inputs=context_seq,
                    cocon_history_inputs=gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences,
                    cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind, 
                    transform_h_after_layernorm=transform_h_after_layernorm,
                    use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
                )
                if prepend_history_seq:
                    gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences = torch.cat([gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences, gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences], dim=1)
                # Remove the batch dimension when returning multiple sequences
                if len(gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences.shape) > 2:
                    gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences.squeeze_()
                gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences_list.append(gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences)



        cocon_output_text_lines_dict = {}
        for generated_sequence_idx, generated_sequence in enumerate(cocon_gen_ar_output_sequences):
            if cocon_output_jsonl_file_path is not None:
                cocon_jsonl_output_dict = {}
            # Decode and log original_input_text
            original_input_sequence = inputs[generated_sequence_idx]
            original_input_sequence = original_input_sequence.tolist()
            original_input_text = tokenizer.decode(original_input_sequence, clean_up_tokenization_spaces=True)
            cocon_output_text_lines_dict[generated_sequence_idx] = ["original_input_text: {} \n".format(original_input_text)]
            if cocon_output_jsonl_file_path is not None:
                cocon_jsonl_output_dict["original_input_text"] = original_input_text

            # Decode and log original_history_seq
            original_history_sequence = original_history_seq[generated_sequence_idx]
            original_history_sequence = original_history_sequence.tolist()
            original_history_text = tokenizer.decode(original_history_sequence, clean_up_tokenization_spaces=True)
            cocon_output_text_lines_dict[generated_sequence_idx].append("original_history_text: {} \n".format(original_history_text))
            if cocon_output_jsonl_file_path is not None:
                cocon_jsonl_output_dict["original_history_text"] = original_history_text

            # Decode and log context_seq
            if type(context_seq) == list:
                context_seq = torch.cat(context_seq, dim=1)
            
            context_sequence = context_seq[generated_sequence_idx]
            context_sequence = context_sequence.tolist()

            context_text = tokenizer.decode(context_sequence, clean_up_tokenization_spaces=True)
            cocon_output_text_lines_dict[generated_sequence_idx].append("context_text: {} \n".format(context_text))
            if cocon_output_jsonl_file_path is not None:
                cocon_jsonl_output_dict["context_text"] = context_text

            # Decode and log original_context_seq
            if args.line_by_line_hs == False and original_context_seq is not None:
                original_context_sequence = original_context_seq[generated_sequence_idx]
                original_context_sequence = original_context_sequence.tolist()
                original_context_text = tokenizer.decode(original_context_sequence, clean_up_tokenization_spaces=True)
                cocon_output_text_lines_dict[generated_sequence_idx].append("original_context_text: {} \n".format(original_context_text))
                if cocon_output_jsonl_file_path is not None:
                    cocon_jsonl_output_dict["original_context_text"] = original_context_text
            else:
                cocon_output_text_lines_dict[generated_sequence_idx].append("original_context_text: None \n")

            # Decode and log cocon AR generated text
            cocon_gen_ar_output_sequence = cocon_gen_ar_output_sequences[generated_sequence_idx]
            cocon_gen_ar_output_sequence = cocon_gen_ar_output_sequence.tolist()
            cocon_gen_ar_output_text = tokenizer.decode(cocon_gen_ar_output_sequence, clean_up_tokenization_spaces=True)
            cocon_output_text_lines_dict[generated_sequence_idx].append("Cocon AR output: {} \n".format(cocon_gen_ar_output_text))
            if cocon_output_jsonl_file_path is not None:
                cocon_jsonl_output_dict["cocon_output"] = cocon_gen_ar_output_text


            if args.context_attn_bias != 0:
                # Decode and log cocon AR generated text, with context_attn_bias
                cocon_gen_conbias_ar_output_sequence = cocon_gen_conbias_ar_output_sequences[generated_sequence_idx]
                cocon_gen_conbias_ar_output_sequence = cocon_gen_conbias_ar_output_sequence.tolist()
                cocon_gen_conbias_ar_output_text = tokenizer.decode(cocon_gen_conbias_ar_output_sequence, clean_up_tokenization_spaces=True)
                cocon_output_text_lines_dict[generated_sequence_idx].append("Cocon AR output, context_attn_bias {}: {} \n".format(args.context_attn_bias, cocon_gen_conbias_ar_output_text))
                if cocon_output_jsonl_file_path is not None:
                    cocon_jsonl_output_dict["cocon_conbias_output"] = cocon_gen_conbias_ar_output_text
                    cocon_jsonl_output_dict["context_attn_bias"] = args.context_attn_bias


            # Decode and log self cocon AR generated text
            if args.line_by_line_hs == False and original_context_seq is not None:
                self_cocon_gen_ar_output_sequence = self_cocon_gen_ar_output_sequences[generated_sequence_idx]
                self_cocon_gen_ar_output_sequence = self_cocon_gen_ar_output_sequence.tolist()
                self_cocon_gen_ar_output_text = tokenizer.decode(self_cocon_gen_ar_output_sequence, clean_up_tokenization_spaces=True)
                cocon_output_text_lines_dict[generated_sequence_idx].append("(Self) Cocon AR output: {} \n".format(self_cocon_gen_ar_output_text))
                if cocon_output_jsonl_file_path is not None:
                    cocon_jsonl_output_dict["self_cocon_output"] = self_cocon_gen_ar_output_text


            # Sanity check (SC) prependgpt2_gen_ar_output_sequences: Decode and log AR generated text
            prependgpt2_gen_ar_output_sequence = prependgpt2_gen_ar_output_sequences[generated_sequence_idx]
            prependgpt2_gen_ar_output_sequence = prependgpt2_gen_ar_output_sequence.tolist()
            prependgpt2_gen_output_text = tokenizer.decode(prependgpt2_gen_ar_output_sequence, clean_up_tokenization_spaces=True)
            cocon_output_text_lines_dict[generated_sequence_idx].append("SC prependgpt2 Autoreg-generated output: {} \n".format(prependgpt2_gen_output_text))
            if cocon_output_jsonl_file_path is not None:
                cocon_jsonl_output_dict["prependgpt2_ar_gen"] = prependgpt2_gen_output_text


            # Sanity check (SC): Decode and log AR generated text
            gen_ar_output_sequence = gen_ar_output_sequences[generated_sequence_idx]
            gen_ar_output_sequence = gen_ar_output_sequence.tolist()
            gen_output_text = tokenizer.decode(gen_ar_output_sequence, clean_up_tokenization_spaces=True)
            cocon_output_text_lines_dict[generated_sequence_idx].append("SC Autoreg-generated output: {} \n".format(gen_output_text))
            if cocon_output_jsonl_file_path is not None:
                cocon_jsonl_output_dict["sc_gpt2_ar_gen"] = gen_output_text


            # Cocon generation (wgpt2genas2ndcs): with gpt2 generations as 2nd context sequence
            if do_cocon_wgpt2genas2ndcs:
                cocon_wgpt2genas2ndcs_gen_ar_output_sequence = cocon_wgpt2genas2ndcs_gen_ar_output_sequences[generated_sequence_idx]
                cocon_wgpt2genas2ndcs_gen_ar_output_sequence = cocon_wgpt2genas2ndcs_gen_ar_output_sequence.tolist()
                cocon_wgpt2genas2ndcs_gen_ar_output_text = tokenizer.decode(cocon_wgpt2genas2ndcs_gen_ar_output_sequence, clean_up_tokenization_spaces=True)
                cocon_output_text_lines_dict[generated_sequence_idx].append("Cocon wgpt2genas2ndcs AR output: {} \n".format(cocon_wgpt2genas2ndcs_gen_ar_output_text))
                if cocon_output_jsonl_file_path is not None:
                    cocon_jsonl_output_dict["cocon_wgpt2genas2ndcs_output"] = cocon_wgpt2genas2ndcs_gen_ar_output_text

                cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequence = cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences[generated_sequence_idx]
                cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequence = cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequence.tolist()
                cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_text = tokenizer.decode(cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequence, clean_up_tokenization_spaces=True)
                cocon_output_text_lines_dict[generated_sequence_idx].append("Cocon wgpt2genas2ndcs (2 parts) AR output: {} \n".format(cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_text))
                if cocon_output_jsonl_file_path is not None:
                    cocon_jsonl_output_dict["cocon_wgpt2genas2ndcs_2parts_output"] = cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_text


                for bias_ind, cs_attn_bias in enumerate(cocon_wgpt2genas2ndcs_cs_attn_biases):
                    cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences = cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences_list[bias_ind]
                    cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequence = cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences[generated_sequence_idx]
                    cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequence = cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequence.tolist()
                    cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_text = tokenizer.decode(cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequence, clean_up_tokenization_spaces=True)
                    cocon_output_text_lines_dict[generated_sequence_idx].append("Cocon wgpt2genas2ndcs (2 parts) AR output, cs_attn_bias {}: {} \n".format(cs_attn_bias, cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_text))
                    if cocon_output_jsonl_file_path is not None:
                        cocon_jsonl_output_dict["cocon_wgpt2genas2ndcs_2parts_output_cs_attn_bias{}".format(cs_attn_bias)] = cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_text

                for bias_ind, gpt2out_attn_bias in enumerate(cocon_wgpt2genas2ndcs_gpt2out_attn_biases):
                    gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences = gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences_list[bias_ind]
                    gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequence = gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences[generated_sequence_idx]
                    gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequence = gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequence.tolist()
                    gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_text = tokenizer.decode(gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequence, clean_up_tokenization_spaces=True)
                    cocon_output_text_lines_dict[generated_sequence_idx].append("Cocon wgpt2genas2ndcs (2 parts) AR output, gpt2out_attn_bias {}: {} \n".format(gpt2out_attn_bias, gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_text))
                    if cocon_output_jsonl_file_path is not None:
                        cocon_jsonl_output_dict["cocon_wgpt2genas2ndcs_2parts_output_gpt2out_attn_bias{}".format(gpt2out_attn_bias)] = gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_text
            
            if cocon_output_jsonl_file_path is not None:
                with open(cocon_output_jsonl_file_path, "a") as f:
                    json.dump(cocon_jsonl_output_dict, f)
                    f.write('\n')

    cocon_output_text_lines = []
    for sample_ind in range(inputs.shape[0]):
        cocon_output_text_lines = cocon_output_text_lines + cocon_output_text_lines_dict[sample_ind] + ["----------\n"]

    with open(cocon_output_file_path, "a", encoding='utf-8') as f:
        f.writelines(cocon_output_text_lines)

    if args.context_attn_bias != 0:
        return cocon_gen_conbias_ar_output_text
    else:
        return cocon_gen_ar_output_text



# Use to generate cocon-edited text with either trained or simple cocon op
def generate_single_cocon_example(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, cocon_block=None, prefix="", 
    random_sample_data=False, use_only_first_context_source_batch=False, use_only_first_custom_mu_s_input_batch=False, transform_h_after_layernorm=False, prepend_history_seq=True) -> Dict:

    eval_output_dir = args.output_dir

    cocon_output_file_path = os.path.join(args.output_dir, args.cocon_output_filename)
    if os.path.exists(cocon_output_file_path):
        if args.append_cocon_output_files:
            logger.info("Append to existing cocon output file")
        else:
            logger.info("Removing existing cocon output file")
            os.remove(cocon_output_file_path)
    else:
        logger.info("Creating new cocon output file")

    if args.cocon_output_jsonl_filename is not None:
        cocon_output_jsonl_file_path = os.path.join(args.output_dir, args.cocon_output_jsonl_filename)
        if os.path.exists(cocon_output_jsonl_file_path):
            if args.append_cocon_output_files:
                logger.info("Append to existing cocon output jsonl file")
            else:
                logger.info("Removing existing cocon output jsonl file")
                os.remove(cocon_output_jsonl_file_path)
        else:
            logger.info("Creating new cocon output jsonl file")
    else:
        cocon_output_jsonl_file_path = None        

    prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")
    if args.prepend_bos_token_to_line:
        prompt_text = tokenizer.bos_token + prompt_text
    logger.info("prompt_text: {}".format(prompt_text))

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    original_history_seq = encoded_prompt.to(args.device)

    content_input_text = args.content_input if args.content_input else input("Content input >>> ")
    logger.info("content_input: {}".format(content_input_text))

    if args.content_input_delimit is not None and args.content_input_delimit in content_input_text:
        content_input_texts = content_input_text.split(args.content_input_delimit)
        logger.info("content_input_texts: {}".format(content_input_texts))
        context_seq = []
        for content_input_text in content_input_texts:
            encoded_content_input = tokenizer.encode(content_input_text, add_special_tokens=False, return_tensors="pt")
            context_seq.append(encoded_content_input.to(args.device))
    else:
        encoded_content_input = tokenizer.encode(content_input_text, add_special_tokens=False, return_tensors="pt")
        context_seq = encoded_content_input.to(args.device)

    original_context_seq = None
    inputs = original_history_seq

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Generate cocon samples!
    logger.info("***** Running single cocon generation {} *****".format(prefix))

    model.eval()
    cocon_block.eval()

    inputs = encoded_prompt
    inputs = inputs.to(args.device)

    cocon_output_text = generate_cocon_sample(context_seq, original_history_seq, original_context_seq, inputs, cocon_output_file_path, args, model, tokenizer, cocon_block, 
        cocon_output_jsonl_file_path=cocon_output_jsonl_file_path, transform_h_after_layernorm=transform_h_after_layernorm, prepend_history_seq=prepend_history_seq, single_generation=True)

    logger.info("cocon_output_text: {} *****".format(cocon_output_text))

    return cocon_output_text



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )

    parser.add_argument(
        "--cs_len",
        default=20,
        type=int,
        help="Context sequence length."
    )
    parser.add_argument(
        "--hs_len",
        default=10,
        type=int,
        help="History sequence length."
    )
    parser.add_argument(
        "--tis_len",
        default=20,
        type=int,
        help="Transformation input sequence length."
    )


    parser.add_argument(
        "--gen_cs_len",
        default=None,
        type=int,
        help="Context sequence length for generation."
    )
    parser.add_argument(
        "--gen_hs_len",
        default=None,
        type=int,
        help="History sequence length for generation."
    )
    parser.add_argument(
        "--gen_tis_len",
        default=None,
        type=int,
        help="Transformation input sequence length for generation."
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_train_lm_batch_size", default=0, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )

    parser.add_argument(
        "--epoch_max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps per epoch to perform. Override num_train_epochs.",
    )

    
    parser.add_argument(
        "--num_lm_train_epochs", default=0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--lm_max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument(
        "--compute_meanvars_before_layernorm",
        action="store_true",
        help="Whether to compute mean and var before layernorm",
    )
    parser.add_argument(
        "--output_meanvars",
        action="store_true",
        default=True,
        help="Whether to output hidden states' mean and var values across channels",
    )
    parser.add_argument(
        "--start_sample_ind",
        type=int,
        default=0,
        help="Index to start computing hidden state stats.",
    )
    parser.add_argument(
        "--num_meanvar_compute",
        type=int,
        default=99999999,
        help="Number of data samples to compute meanvars.",
    )
    parser.add_argument(
        "--meanvar_output_filename",
        type=str,
        default='mean_var.npy',
        help="The output file to save data sample mean/var values.",
    )

    parser.add_argument(
        "--compute_meanvars_random_sample",
        action="store_true",
        help="Whether to sample randomly while computing mean/var.",
    )

    parser.add_argument(
        "--eval_output_filename",
        type=str,
        default="eval_results.txt",
        help="The output file to save eval results.",
    )

    parser.add_argument(
        "--cocon_output_filename",
        type=str,
        default="cocon_output.txt",
        help="The output file to save cocon generated text.",
    )

    parser.add_argument(
        "--cocon_output_jsonl_filename",
        type=str,
        default="cocon_output.jsonl",
        help="The output jsonl file to save cocon generated text.",
    )

    parser.add_argument(
        "--w_mapper_num_layers", type=int, default=5, help="Number of fc layers for mapping z into w"
    )
    parser.add_argument(
        "--w_mapper_dropout_prob", type=float, default=None, help="Dropout probability for z to w mapper fc layers"
    )
    parser.add_argument(
        "--gen_update_interval", type=int, default=1, help="Number of lm steps before a gen update step"
    )
    parser.add_argument(
        "--disc_update_interval", type=int, default=1, help="Number of lm steps before a disc update step"
    )

    parser.add_argument(
        "--step_ind_to_start_cycle_ar_cocon_recon", type=int, default=0, help="Step number to start own_style_cocon learning, 0 is the first step"
    )
    parser.add_argument(
        "--epoch_ind_to_start_cycle_ar_cocon_recon", type=int, default=0, help="Training epoch number to start own_style_cocon learning, 0 is the first epoch"
    )

    parser.add_argument(
        "--step_ind_to_start_other_context_cocon", type=int, default=0, help="Step number to start own_style_cocon learning, 0 is the first step"
    )
    parser.add_argument(
        "--epoch_ind_to_start_other_context_cocon", type=int, default=0, help="Training epoch number to start own_style_cocon learning, 0 is the first epoch"
    )

    parser.add_argument(
        "--step_ind_to_start_adv", type=int, default=0, help="LM step number to start adversarial learning, 0 is the first step"
    )
    parser.add_argument(
        "--epoch_ind_to_start_adv", type=int, default=0, help="Training epoch number to start adversarial learning, 0 is the first epoch"
    )

    parser.add_argument(
        "--step_ind_to_start_hist_cocon_lm", type=int, default=0, help="Step number to start hist_cocon_lm learning, 0 is the first step"
    )
    parser.add_argument(
        "--epoch_ind_to_start_hist_cocon_lm", type=int, default=0, help="Training epoch number to start hist_cocon_lm learning, 0 is the first epoch"
    )


    parser.add_argument(
        "--steps_to_closely_monitor_adv", type=int, default=10000, help="Training epoch number to start adversarial learning, 0 is the first epoch"
    )
    
    parser.add_argument("--no_adv_gen_train_original_lm", action="store_true", help="Whether to backprop loss through original LM feed forward.")
    parser.add_argument("--use_original_text_as_disc_real", action="store_true", help="Whether to use original text as real input of the disc.")
    parser.add_argument("--gen_gumbel_softmax", action="store_true", help="Whether to use gumbel softmax for computing probs from gen output logits.")
    
    parser.add_argument(
        "--latent_mixing_prob", type=float, default=0, help="Probability of mixing 2 generated_cocon_vector during cocon generation"
    )

    parser.add_argument('--block_indices', 
        nargs='+', 
        type=int,
        default=None,
        help="0,1,2,3,4,5,6,7,8,9,10,11,12 where 12 is the final FF layer without self-attn",
    )
    parser.add_argument('--layer_indices', 
        nargs='+', 
        type=int,
        default=None,
        help="0,1 where 0 is the hidden state stats before self-attn layer and 1 is stats before FF layer",
    )
    parser.add_argument('--stat_indices', 
        nargs='+', 
        type=int,
        default=None,
        help="0,1 where 0 is mean and 1 is var",
    )

    parser.add_argument("--do_cocon_compute", action="store_true", help="Whether to generate text with cocon.")

    parser.add_argument("--eval_compute_without_checkpoint", action="store_true", help="Whether to use saved checkpoint or use pretrained LM to evaluate and compute stats.")

    parser.add_argument(
        "--distance_metric",
        type=str,
        default="l2",
        help="Distance metric used to compute loss value for hidden values, can be l2 (l2 distance) or cos (cosine similarity).",
    )   

    parser.add_argument(
        "--stats_distance_metric",
        type=str,
        default=None,
        help="Distance metric used to compute loss value, can be 'cos' (cosine similarity), 'normalized_l2' (l2 distance between normalized vectors) or 'l2' (l2 distance between unnormalized vectors).",
    )

    parser.add_argument(
        "--lambda_self_cocon_lm_loss", type=float, default=1, help="Lambda value of self_cocon_lm_loss optimization"
    )
    parser.add_argument(
        "--lambda_hist_cocon_lm_loss", type=float, default=0, help="Lambda value of hist_cocon_lm_loss optimization"
    )
    parser.add_argument(
        "--lambda_cycle_ar_cocon_recon_lm_loss", type=float, default=0, help="Lambda value of cycle_ar_cocon_recon_lm_loss optimization"
    )
    parser.add_argument(
        "--lambda_other_context_cocon_lm_loss", type=float, default=0, help="Lambda value of other_context_cocon_lm_loss optimization"
    )
    parser.add_argument(
        "--lambda_adv", type=float, default=0, help="Lambda value of adversarial loss optimization"
    )

    parser.add_argument("--per_gpu_train_cycle_ar_cocon_recon_batch_size", default=None, type=int, help="Batch size per GPU/CPU for training when cycle_recon training starts.")
    
    parser.add_argument("--adv_use_th_gen_output", action="store_true", help="Whether to use cocon > GPT2 tail-head output as fake examples.")

    parser.add_argument("--track_loss_gradnorms", action="store_true", help="Whether to log all loss gradnorm to tb.")
    
    parser.add_argument("--save_lm_model", action="store_true", help="Whether to save (GPT-2) lm model.")
    parser.add_argument("--only_lm", action="store_true", help="Whether to train and infer only lm model, without cocon.")

    parser.add_argument(
        "--cocon_compute_history_source_data_file",
        type=str,
        default="data/gpt2output/webtext.valid.jsonl",
        help="The file for content source data.",
    )

    parser.add_argument(
        "--cocon_compute_context_source_data_file",
        type=str,
        default="data/gpt2output/webtext.test.jsonl",
        help="The file for content source data.",
    )

    parser.add_argument(
        "--num_cocon_generate",
        type=int,
        default=99999999,
        help="Number of cocon samples to generate.",
    )

    parser.add_argument(
        "--output_hidden_for_cocon_after_block_ind", type=int, default=6, help="Block index to output hidden state for cocon computation"
    )
    
    parser.add_argument("--transform_h_after_layernorm", action="store_true", help="Whether to do cocon after layer norm op, generated text results are poorer in this setting.")

    parser.add_argument("--use_only_first_context_source_batch", action="store_true", help="Whether to use only the first style source batch for cocon.")
    
    parser.add_argument("--use_token_gate", action="store_true", help="Whether to use token gate for cocon_block.")
    parser.add_argument("--use_global_gate", action="store_true", help="Whether to use global sequence gate for cocon_block.")
    parser.add_argument("--split_c_proj", action="store_true", help="Whether to use separate c_proj after attn op for mu and sigma.")    

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument("--generate_length", type=int, default=20)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")

    #  interpolation of mean and variance latent vectors
    parser.add_argument(
        "--mean_attr_direction_npy_filename", default=None,
        type=str, help="npy file that stores mean attr latent vector direction."
    )
    parser.add_argument(
        "--mean_start_distance", type=float, default=-50, help="Min magnitude to add mean attr latent vector to original vector"
    )
    parser.add_argument(
        "--mean_end_distance", type=float, default=50, help="Max magnitude to add mean attr latent vector to original vector"
    )
    parser.add_argument(
        "--var_attr_direction_npy_filename", default=None,
        type=str, help="npy file that stores var attr latent vector direction."
    )
    parser.add_argument(
        "--var_start_distance", type=float, default=-50, help="Min magnitude to add var attr latent vector to original vector"
    )
    parser.add_argument(
        "--var_end_distance", type=float, default=50, help="Max magnitude to add var attr latent vector to original vector"
    )
    parser.add_argument(
        "--num_interpolation", type=int, default=9, help="Number of interpolations for attr direction addition to latent vector"
    )
    parser.add_argument(
        "--encoded_prompt_len_cocon_gen", type=int, default=2, help="Length of prompt input ids to use during cocon generation."
    )

    parser.add_argument(
        "--include_zero_prompt",
        action="store_true",
        help="Whether include generated text samples with zero prompt, similar to encoded_prompt_len_cocon_gen=0",
    )

    parser.add_argument(
        "--custom_context_input_text_data_file",
        type=str,
        default=None,
        help="text file for sequences to use for custom mu_s generation",
    )

    parser.add_argument(
        "--train_cycle_detach_interval", type=int, default=1, help="Interval to detach cycle generated hidden states"
    )

    parser.add_argument("--use_unopt_cycle_recon_cocon_training", action="store_true", help="Whether to use unoptimized cycle recon training code.")
    
    parser.add_argument(
        "--cocon_block_type",
        type=str,
        default="1",
        help="Cocon block type , can be one of [1, 2, 3].",
    )
    
    parser.add_argument("--max_cocon_AR_length", type=int, default=100)

    parser.add_argument(
        "--self_cocon_lm_cs_mask_prob", type=float, default=0, help="Ratio of cs' hidden states for self_cocon_lm_loss computation"
    )
    parser.add_argument(
        "--self_cocon_lm_tis_mask_prob", type=float, default=0, help="Ratio of tis' hidden states for self_cocon_lm_loss computation"
    )
    parser.add_argument("--self_cocon_lm_mutual_exc_mask", action="store_true", help="Whether to use mutually exclusive masks for cs and tis for for self_cocon_lm_loss computation.")
    
    parser.add_argument(
        "--cycle_ar_cocon_recon_lm_tis_mask_prob", type=float, default=0, help="Ratio of tis' hidden states for cycle_ar_cocon_recon_lm_loss computation"
    )

    parser.add_argument("--use_only_last_cocon_output_for_ar", action="store_true", help="Whether to use_only_last_cocon_output_for_ar rather than the whole cocon output.")
    
    parser.add_argument("--use_history_source_as_context_source_for_gen", action="store_true", help="Whether to use history_source_data_file as context_source_data_file.")

    parser.add_argument(
        "--self_token_mask_prob", type=float, default=0, help="Probability to mask own token in context seq during self_cocon_lm_loss computation"
    )
    parser.add_argument(
        "--cycle_self_token_mask_prob", type=float, default=0, help="Probability to mask own position's token in context seq during cycle_ar_cocon_recon_lm_loss computation"
    )
    parser.add_argument(
        "--other_context_self_token_mask_prob", type=float, default=0, help="Probability to mask own position's token in context seq during other_context_cocon_lm_loss computation"
    )

    parser.add_argument("--min_hs_tis_split_offset", type=int, default=0, help="Min number of index to offset from hs_len to split train samples into hs/tis")
    parser.add_argument("--max_hs_tis_split_offset", type=int, default=0, help="Max number of index to offset from hs_len to split train samples into hs/tis")
    
    parser.add_argument("--track_hist_cocon_lm_loss", action="store_true", help="Whether to track hist_cocon_lm_loss for logging even without using for training.")
    
    parser.add_argument(
        "--line_by_line_cs",
        action="store_true",
        help="Whether distinct lines of text in the context seq dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--line_by_line_hs",
        action="store_true",
        help="Whether distinct lines of text in the history seq (prompt text) dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--enumerate_all_cs_for_each_hs",
        action="store_true",
        help="Whether to enumerate all context sequences for each history seq (prompt text) during cocon generation.",
    )
    
    parser.add_argument(
        "--prepend_bos_token_to_line",
        action="store_true",
        help="Whether to prepend bos_token to history seq (prompt text) during cocon generation.",
    )
    
    
    parser.add_argument("--text_json_key", type=str, default="text", help="key for sample text in data json object")

    parser.add_argument(
        "--prepended_text_to_remove",
        type=str,
        default=None,
        help="Prepended text to remove during data loading for evaluation, use ; to delimit a list of prepended_texts",
    )

    parser.add_argument("--do_eval_dist", action="store_true", help="Whether to run dist-1,2,3 eval on the dev set.")
    parser.add_argument("--dist_eval_max_samples", type=int, default=-1, help="Defaults to -1 which has no max limit.")


    parser.add_argument(
        "--context_attn_bias", type=float, default=0, help="Value to bias context_attn during cocon forward ops, for generation."
    )

    parser.add_argument(
        "--content_input",
        type=str,
        default=None,
        help="Content input for single COCON generation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Content input for single COCON generation",
    )
    parser.add_argument("--content_input_delimit", 
        type=str,
        default=';',
        help="Delimiter for multiple content inputs",
    )
    parser.add_argument("--do_single_cocon_generation", action="store_true", help="Whether to generate single text with cocon.")
    parser.add_argument("--append_cocon_output_files", action="store_true", help="Whether to append to existing cocon_output_file and cocon_output_jsonl.")


    args = parser.parse_args()    

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        logger.info("Loading tokenizer from pretrained, {}".format(args.model_name_or_path))
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        if args.output_meanvars and ('gpt2' in args.model_name_or_path):
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir,
                output_meanvars=True,
                compute_meanvars_before_layernorm=args.compute_meanvars_before_layernorm
            )
        else:
            logger.info("Loading model from pretrained weights, {}".format(args.model_name_or_path))
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir,
            )
    else:
        logger.info("Training new model from scratch")
        if args.output_meanvars:
            model = model_class(config=config, output_meanvars=True, compute_meanvars_before_layernorm=args.compute_meanvars_before_layernorm)
        else:
            model = model_class(config=config)

    model.to(args.device)

    if args.only_lm == False:
        # Set up CoconBlock
        cocon_block = CoconBlock(config.n_ctx, config, scale=True)
        cocon_block.to(args.device)
        
        if args.lambda_adv > 0:
            # Set up disc_model model
            disc_model = HDiscriminator(config=config)
            disc_model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        if args.num_lm_train_epochs > 0:
            global_step, tr_loss = train_lm(args, train_dataset, model, tokenizer)


        if args.only_lm == False:
            if args.lambda_adv > 0: 
                global_step, tr_loss = train_cocon(args, train_dataset, model, tokenizer, cocon_block=cocon_block, disc_model=disc_model, model_config=config, transform_h_after_layernorm=args.transform_h_after_layernorm)
            else:
                global_step, tr_loss = train_cocon(args, train_dataset, model, tokenizer, cocon_block=cocon_block, model_config=config, transform_h_after_layernorm=args.transform_h_after_layernorm)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        if args.num_lm_train_epochs > 0 or args.save_lm_model:
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Load a trained model and vocabulary that you have fine-tuned
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = tokenizer_class.from_pretrained(args.output_dir)
            model.to(args.device)

        if args.only_lm == False:
            # Save cocon_block model
            cocon_block_weights_name = "cocon_block_pytorch_model.bin"
            output_cocon_block_model_file = os.path.join(args.output_dir, cocon_block_weights_name)
            torch.save(cocon_block.state_dict(), output_cocon_block_model_file)
            logger.info("cocon_block model weights saved in {}".format(output_cocon_block_model_file))

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Compute cocon text: generate cocon text
    results = {}
    if args.do_cocon_compute and args.local_rank in [-1, 0] and args.only_lm == False:
        if args.gen_cs_len is None:
            args.gen_cs_len = args.cs_len
        if args.gen_hs_len is None:
            args.gen_hs_len = args.hs_len
        if args.gen_tis_len is None:
            args.gen_tis_len = args.tis_len

        if not args.eval_compute_without_checkpoint:
            checkpoints = [args.output_dir]
        else:
            checkpoints = ["pretrained"]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            if not args.eval_compute_without_checkpoint:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

                if args.output_meanvars:
                    model = model_class.from_pretrained(
                        checkpoint,
                        output_meanvars=True,
                        compute_meanvars_before_layernorm=args.compute_meanvars_before_layernorm
                    )
                else:
                    model = model_class.from_pretrained(checkpoint)
            else:
                global_step = 0
                prefix = ""

            # Load cocon_block model
            cocon_block_weights_name = "cocon_block_pytorch_model.bin"
            output_cocon_block_model_file = os.path.join(args.output_dir, cocon_block_weights_name)

            cocon_state_dict = torch.load(output_cocon_block_model_file)
            new_cocon_state_dict = fix_state_dict_naming(cocon_state_dict)
            cocon_block.load_state_dict(new_cocon_state_dict)

            model.to(args.device)
            cocon_block.to(args.device)
            
            generate_steps = generate_cocon_compute(args, model, tokenizer, cocon_block=cocon_block, prefix=prefix, use_only_first_context_source_batch=args.use_only_first_context_source_batch, transform_h_after_layernorm=args.transform_h_after_layernorm)

    # Single cocon generation: generate single cocon text
    results = {}
    if args.do_single_cocon_generation and args.local_rank in [-1, 0] and args.only_lm == False:
        if not args.eval_compute_without_checkpoint:
            checkpoints = [args.output_dir]
        else:
            checkpoints = ["pretrained"]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            if not args.eval_compute_without_checkpoint:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

                if args.output_meanvars:
                    model = model_class.from_pretrained(
                        checkpoint,
                        output_meanvars=True,
                        compute_meanvars_before_layernorm=args.compute_meanvars_before_layernorm
                    )
                else:
                    model = model_class.from_pretrained(checkpoint)
            else:
                global_step = 0
                prefix = ""

            # Load cocon_block model
            cocon_block_weights_name = "cocon_block_pytorch_model.bin"
            output_cocon_block_model_file = os.path.join(args.output_dir, cocon_block_weights_name)

            cocon_block.load_state_dict(torch.load(output_cocon_block_model_file), strict=False) # to deal with earlier cocon weights without h_mask and self_token_mask 
            model.to(args.device)
            cocon_block.to(args.device)
            
            generate_steps = generate_single_cocon_example(args, model, tokenizer, cocon_block=cocon_block, prefix=prefix, use_only_first_context_source_batch=args.use_only_first_context_source_batch, transform_h_after_layernorm=args.transform_h_after_layernorm)

    # Evaluation: evaluate model on loss values
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if not args.eval_compute_without_checkpoint:
            checkpoints = [args.output_dir]
        else:
            checkpoints = ["pretrained"]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            if not args.eval_compute_without_checkpoint:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

                if args.output_meanvars:
                    model = model_class.from_pretrained(
                        checkpoint,
                        output_meanvars=True,
                        compute_meanvars_before_layernorm=args.compute_meanvars_before_layernorm
                    )
                else:
                    model = model_class.from_pretrained(checkpoint)
            else:
                global_step = 0
                prefix = ""
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

     # Evaluation: evaluate model on loss values
    if args.do_eval_dist and args.local_rank in [-1, 0]:
        if not args.eval_compute_without_checkpoint:
            checkpoints = [args.output_dir]
        else:
            checkpoints = ["pretrained"]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            if not args.eval_compute_without_checkpoint:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

                if args.output_meanvars:
                    model = model_class.from_pretrained(
                        checkpoint,
                        output_meanvars=True,
                        compute_meanvars_before_layernorm=args.compute_meanvars_before_layernorm
                    )
                else:
                    model = model_class.from_pretrained(checkpoint)
            else:
                global_step = 0
                prefix = ""
            model.to(args.device)
            result = evaluate_dist_scores(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
