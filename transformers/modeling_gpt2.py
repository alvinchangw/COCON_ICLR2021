# coding=utf-8
"""PyTorch OpenAI GPT-2 model."""


import logging
import math
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from .activations import gelu_new
from .configuration_gpt2 import GPT2Config
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_utils import Conv1D, PreTrainedModel, SequenceSummary, prune_conv1d_layer


logger = logging.getLogger(__name__)

GPT2_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin",
    "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch_model.bin",
    "gpt2-large": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-pytorch_model.bin",
    "gpt2-xl": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-pytorch_model.bin",
    "distilgpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-pytorch_model.bin",
}


def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd : ns, :ns]
        w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu_new
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False, output_meanvars=False, compute_meanvars_before_layernorm=False):
        super().__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.instance_norm = nn.InstanceNorm1d(nx, affine=False, track_running_stats=False)
        self.output_meanvars = output_meanvars
        self.compute_meanvars_before_layernorm = compute_meanvars_before_layernorm

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, adaIN=False, adaIN_vector=None, adaIN_before_ln=False, return_point=None, input_point=None):
        if input_point is None:
            input_ln_1 = x

            x_ln_1 = self.ln_1(input_ln_1)

            if return_point == 'ln_1':
                return x_ln_1
        elif input_point == 'ln_1':
            x_ln_1 = x

        if input_point is None or input_point == 'ln_1':
            if self.output_meanvars:
                if self.compute_meanvars_before_layernorm and input_point != 'ln_1':
                    first_latent_mean = input_ln_1.mean(1)
                    first_latent_var = input_ln_1.var(1)
                else:
                    first_latent_mean = x_ln_1.mean(1)
                    first_latent_var = x_ln_1.var(1)

            x_1_output = x_ln_1
            
            output_attn = self.attn(
                x_1_output, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask
            )

            a = output_attn[0]  # output_attn: a, present, (attentions)
            
            x = x + a
            
            input_ln_2 = x

        if return_point == 'attn':
            return input_ln_2

        if input_point == 'attn':
            input_ln_2 = x

        if input_point is None or input_point == 'attn' or input_point == 'ln_1':
            x_ln_2 = self.ln_2(input_ln_2)
        
        if return_point == 'ln_2':
            return x_ln_2

        if input_point == 'ln_2':
            x_ln_2 = x

        if self.output_meanvars:
            if self.compute_meanvars_before_layernorm and input_point != 'ln_2':
                second_latent_mean = input_ln_2.mean(1)
                second_latent_var = input_ln_2.var(1)
            else:
                second_latent_mean = x_ln_2.mean(1)
                second_latent_var = x_ln_2.var(1)

        x_2_output = x_ln_2

        m = self.mlp(x_2_output)

        x = x + m

        if self.output_meanvars:
            outputs = [x] + output_attn[1:] + [(first_latent_mean, first_latent_var), (second_latent_mean, second_latent_var)]
            return outputs  # x, present, (attentions),  ((x_ln_1_mean, x_ln_1_var)), ((x_ln_2_mean, x_ln_2_var))
        else:
            outputs = [x] + output_attn[1:]
            return outputs  # x, present, (attentions)


class CoconBlock(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        logger.info( "CoconBlock initialized")
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

        self.sos_h = nn.Parameter(torch.zeros(nx))
        self.mask_h = nn.Parameter(torch.zeros(nx))

        self.cocon_attn = CoconAttention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.instance_norm = nn.InstanceNorm1d(nx, affine=False, track_running_stats=False)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        
        self.config = config

        self.init_weights()

    def forward(self, x, context_seq=None, history_seq=None, layer_past=None, attention_mask=None, head_mask=None, include_sos_output=False, cs_masked_indices=None, tis_masked_indices=None, cs_self_attn_mask_prob=0, context_attn_bias=0, context_seq_len_list=None):
        if cs_masked_indices is not None and context_seq is not None:
            context_seq = context_seq.clone() # avoid overwrite original context_seq with mask_h
            context_seq[cs_masked_indices] = self.mask_h

        if tis_masked_indices is not None and x is not None:
            x = x.clone() # avoid overwrite original x with mask_h
            x[tis_masked_indices] = self.mask_h

        if history_seq is not None:
            history_seq_len = history_seq.shape[1]
            if x is not None:
                cocon_attn_input = torch.cat([history_seq, x], dim=1)
            else:
                cocon_attn_input = history_seq
        elif x is not None:
            history_seq_len = 0
            batch_size = x.shape[0]
            sos_h = self.sos_h.view(1, 1, -1).expand(batch_size, -1, -1)
            cocon_attn_input = torch.cat([sos_h, x], dim=1)

        x = cocon_attn_input


        cocon_attn_input_ln_1 = self.ln_1(cocon_attn_input)
        x_1_output = cocon_attn_input_ln_1

        output_attn = self.cocon_attn(
            x_1_output, context_seq, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, cs_self_attn_mask_prob=cs_self_attn_mask_prob, history_seq_len=history_seq_len, 
            context_attn_bias=context_attn_bias, context_seq_len_list=context_seq_len_list
        )
        a = output_attn[0]  # output_attn: (a), present, (attentions)
        # H^L_preconv
        x = x + a

        # Skip history_seq computation if history_seq_len > 1
        if history_seq_len > 1:
            x = x[:, history_seq_len-1:]


        x_ln_2 = self.ln_2(x)
        x_2_output = x_ln_2
        m = self.mlp(x_2_output)
        # H^L
        x = x + m

        if include_sos_output:
            cocon_output = x
        else:
            cocon_output = x[:, 1:, :]

        return cocon_output


    def init_weights(self):
        """ Initialize weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) and module.bias is not None:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class CoconAttention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))

        self_token_mask = torch.ones(n_ctx, n_ctx)
        self_token_mask.fill_diagonal_(0)
        self.register_buffer("self_token_mask", self_token_mask.view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.ref_source_attn = Conv1D(n_state * 2, nx)
        self.c_attn = Conv1D(n_state * 3, nx) # input has dim of nx
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, cs_self_attn_mask_prob=0, history_seq_len=None, context_seq_present=True, context_seq_len=0, context_attn_bias=0, context_seq_len_list=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd : ns, :ns]
        w = w * b - 1e4 * (1 - b)

        # self_token_mask computation
        if cs_self_attn_mask_prob > 0 and context_seq_present:
            if history_seq_len == 0:
                history_seq_offset = 0
            else:
                history_seq_offset = history_seq_len - 1
            self_token_mask = self.self_token_mask[:, :, :nd, history_seq_offset:history_seq_offset+ns]
            self_token_mask = self_token_mask.repeat(w.shape[0],1,1,1)

            if cs_self_attn_mask_prob != 1:
                # compute unmasked indices
                self_token_unmask_prob = 1 - cs_self_attn_mask_prob
                unmask_prob_matrix = torch.full(self_token_mask.shape[:-1], self_token_unmask_prob)
                unmasked_indices = torch.bernoulli(unmask_prob_matrix).bool()
                self_token_mask[unmasked_indices] = 1

            w = w * self_token_mask - 1e4 * (1 - self_token_mask)
            
        
        if context_attn_bias != 0:
            if context_seq_len_list is None:
                context_attn_bias_mask = torch.ones(w.shape) # N, H, Q, V
                context_attn_bias_mask[:,:,:, :context_seq_len] = 0
                context_attn_bias_mask = context_attn_bias_mask.to(w.device)
                w = w + context_attn_bias * (1 - context_attn_bias_mask)     
            else:
                current_context_start_ind = 0
                for cs_ind, current_context_seq_len in enumerate(context_seq_len_list):
                    current_context_attn_bias = context_attn_bias[cs_ind]
                    context_attn_bias_mask = torch.ones(w.shape)
                    context_attn_bias_mask[:,:,:, current_context_start_ind:(current_context_start_ind+current_context_seq_len)] = 0
                    context_attn_bias_mask = context_attn_bias_mask.to(w.device)
                    w = w + current_context_attn_bias * (1 - context_attn_bias_mask)
                    current_context_start_ind = current_context_start_ind + current_context_seq_len

            
        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, context_seq, layer_past=None, attention_mask=None, head_mask=None, cs_self_attn_mask_prob=0, history_seq_len=None, context_attn_bias=0, context_seq_len_list=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)

        if context_seq is not None:
            context_seq_len = context_seq.shape[1]
            context_seq = self.ref_source_attn(context_seq)
            key_context_seq, value_context_seq = context_seq.split(self.split_size, dim=2)

            # Prepend keys and values with context_seq keys and values
            prepended_key = torch.cat([key_context_seq, key], dim=1)
            prepended_value = torch.cat([value_context_seq, value], dim=1)
            context_seq_present = True
        else:
            context_seq_len = 0
            prepended_key = key
            prepended_value = value
            context_seq_present = False

        query = self.split_heads(query)
        prepended_key = self.split_heads(prepended_key, k=True)
        prepended_value = self.split_heads(prepended_value)

        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        attn_outputs = self._attn(query, prepended_key, prepended_value, attention_mask, head_mask, cs_self_attn_mask_prob=cs_self_attn_mask_prob, history_seq_len=history_seq_len, context_seq_present=context_seq_present, 
                                    context_seq_len=context_seq_len, context_attn_bias=context_attn_bias, context_seq_len_list=context_seq_len_list)

        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs

        return outputs


class GPT2PreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = GPT2Config
    pretrained_model_archive_map = GPT2_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


GPT2_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.GPT2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

GPT2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.GPT2Tokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `past` output below). Can be used to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        input_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config, output_meanvars=False, compute_meanvars_before_layernorm=False):
        super().__init__(config)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.output_past = config.output_past

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True, output_meanvars=output_meanvars, compute_meanvars_before_layernorm=compute_meanvars_before_layernorm) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.output_meanvars = output_meanvars
        self.compute_meanvars_before_layernorm = compute_meanvars_before_layernorm

        self.instance_norm = nn.InstanceNorm1d(config.n_embd, affine=False, track_running_stats=False)
        self.n_layer = config.n_layer
        self.init_weights()


    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        input_hidden_state=None,
        input_before_block_ind=None,
        output_after_block_ind=None,
        adaIN=False,
        adaIN_vector=None,
        adaIN_before_ln=False,
        input_point=None, # 'current_block_ln_1' or None
        return_point=None, # 'next_block_ln_1' or None
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import GPT2Tokenizer, GPT2Model
        import torch

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        elif input_hidden_state is not None:
            input_shape = input_hidden_state.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids, inputs_embeds or input_hidden_state")

        if input_before_block_ind is not None:
            if output_after_block_ind is not None:
                raise ValueError("You cannot specify both input_before_block_ind and output_after_block_ind")
            elif input_hidden_state is None:
                raise ValueError("You must specify input_hidden_state with input_before_block_ind")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None or len(past) == 0:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
            
            # pad layer pasts with None
            if len(past) != len(self.h):
                layer_past_pads = (None,) * (len(self.h) - len(past))
                if type(past) != tuple:
                    past = tuple(past)
                if input_before_block_ind is not None:
                    past = layer_past_pads + past
                elif output_after_block_ind is not None:
                    past = past + layer_past_pads

        if position_ids is None:
            if input_ids is not None:
               device = input_ids.device
            elif inputs_embeds is not None: 
               device = inputs_embeds.device
            else:
               device = input_hidden_state.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer


        if input_hidden_state is not None and input_before_block_ind is not None:
            hidden_states = input_hidden_state
        else:
            if inputs_embeds is None:
                inputs_embeds = self.wte(input_ids)
            position_embeds = self.wpe(position_ids)
            if token_type_ids is not None:
                token_type_embeds = self.wte(token_type_ids)
            else:
                token_type_embeds = 0
            hidden_states = inputs_embeds + position_embeds + token_type_embeds
            hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_meanvars = []
        all_hidden_states = ()     
        for block_ind, (block, layer_past) in enumerate(zip(self.h, past)):     
            if input_before_block_ind is not None:
                if input_point == 'current_block_ln_1' and block_ind == input_before_block_ind:
                    if self.output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)
                        
                    outputs = block(
                        hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[block_ind], input_point='ln_1'
                    )                   

                elif block_ind >= input_before_block_ind:
                    if self.output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

                    outputs = block(
                        hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[block_ind]
                    )
                else:
                    continue

            else:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

                if output_after_block_ind is not None and (output_after_block_ind+1 == block_ind) and return_point == 'next_block_ln_1':
                    hidden_states = block(
                        hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[block_ind], return_point='ln_1'
                    )                               
                    hidden_states = hidden_states.view(*output_shape)
                    outputs = (hidden_states,)
                    return outputs

                outputs = block(
                    hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[block_ind]
                )

            hidden_states, present = outputs[:2]

            if self.output_past:
                presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

            if self.output_meanvars:
                all_meanvars.append(outputs[-2:])

            if output_after_block_ind == block_ind and return_point != 'next_block_ln_1':
                hidden_states = hidden_states.view(*output_shape)
                outputs = (hidden_states,)
                if self.output_past:
                    outputs = outputs + (presents,)
                if self.output_hidden_states:
                    outputs = outputs + (all_hidden_states,)
                if self.output_attentions:
                    # let the number of heads free (-1) so we can extract attention even after head pruning
                    attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
                    all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
                    outputs = outputs + (all_attentions,)
                return outputs

        if self.output_meanvars and self.compute_meanvars_before_layernorm == True:
            final_h_mean = hidden_states.mean(1)
            final_h_var = hidden_states.var(1)
            all_meanvars.append(((final_h_mean, final_h_var), ))


        hidden_states = self.ln_f(hidden_states)


        if self.output_meanvars and self.compute_meanvars_before_layernorm == False:
            final_h_mean = hidden_states.mean(1)
            final_h_var = hidden_states.var(1)
            all_meanvars.append(((final_h_mean, final_h_var), ))

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        if self.output_meanvars:
            outputs = outputs + (all_meanvars,)
        return outputs  # last hidden state, (presents), (all hidden_states), (all_attentions), (all_meanvars

### Beacon!
@add_start_docstrings(
    """The GPT2 Model transformer with a language modeling head on top
    (linear layer with weights tied to the input embeddings). """,
    GPT2_START_DOCSTRING,
)
class GPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config, output_meanvars=False, compute_meanvars_before_layernorm=False):
        super().__init__(config)
        self.transformer = GPT2Model(config, output_meanvars=output_meanvars, compute_meanvars_before_layernorm=compute_meanvars_before_layernorm)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.output_meanvars = output_meanvars
        self.compute_meanvars_before_layernorm = compute_meanvars_before_layernorm

        self.n_layer = config.n_layer
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if "past" in kwargs and kwargs["past"]:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        inputs = {"input_ids": input_ids}
        inputs.update(kwargs)
        return inputs

    def prepare_embeds_inputs_for_generation(self, inputs_embeds, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if "past" in kwargs and kwargs["past"]:
            inputs_embeds = inputs_embeds[:, -1:, :]

        inputs = {"inputs_embeds": inputs_embeds}
        inputs.update(kwargs)
        return inputs

    def prepare_hidden_state_inputs_for_generation(self, input_hidden_state, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if "past" in kwargs and kwargs["past"]:
            input_hidden_state = input_hidden_state[:, -1:, :]

        inputs = {"input_hidden_state": input_hidden_state}
        inputs.update(kwargs)
        return inputs

    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        input_hidden_state=None,
        input_before_block_ind=None,
        output_after_block_ind=None,
        adaIN=False,
        adaIN_vector=None,
        input_point=None,
        return_point=None,
        lm_logit_first_index=0,
        lm_logit_last_index=-1,
        lm_labels_first_index=1,
        lm_labels_last_index=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when ``labels`` is provided)
            Language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2LMHeadModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

        """

        if input_before_block_ind == self.n_layer and input_point == 'current_block_ln_1':
            hidden_states = input_hidden_state
        else:
            if output_after_block_ind is not None:
                transformer_outputs = self.transformer(
                    input_ids,
                    past=past,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_after_block_ind=output_after_block_ind,
                    return_point=return_point,
                )
                hidden_states = transformer_outputs[0] # previously only (hidden_states, ) now (hidden_states, present, .. ) 
                return transformer_outputs
            elif input_before_block_ind is not None:
                transformer_outputs = self.transformer(
                    input_ids,
                    past=past,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    input_hidden_state=input_hidden_state,
                    input_before_block_ind=input_before_block_ind,
                    input_point=input_point,
                )
            else:
                transformer_outputs = self.transformer(
                    input_ids,
                    past=past,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                )
            hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        if input_before_block_ind == self.n_layer and input_point == 'current_block_ln_1':
            outputs = (lm_logits,)
        else:
            outputs = (lm_logits,) + transformer_outputs[1:]
            
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., lm_logit_first_index:lm_logit_last_index, :].contiguous() # default lm_logit_first_index=0, lm_logit_last_index=-1,
            shift_labels = labels[..., lm_labels_first_index:lm_labels_last_index].contiguous() # default lm_labels_first_index=1, lm_labels_last_index=None,

            loss_fct = CrossEntropyLoss()

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (all_attentions), (all_meanvars)


@add_start_docstrings(
    """The GPT2 Model transformer with a language modeling and a multiple-choice classification
    head on top e.g. for RocStories/SWAG tasks. The two heads are two linear layers.
    The language modeling head has its weights tied to the input embeddings,
    the classification head takes as input the input of a specified classification token index in the input sequence).
""",
    GPT2_START_DOCSTRING,
)
class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
    ):
        r"""
        mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input)
            Index of the classification token in each input sequence.
            Selected in the range ``[0, input_ids.size(-1) - 1[``.
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`)
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`, defaults to :obj:`None`)
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``lm_labels`` is provided):
            Language modeling loss.
        mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`multiple_choice_labels` is provided):
            Multiple choice classification loss.
        lm_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

        # Add a [CLS] to the vocabulary (we should train it also!)
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
        logger.info(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary

        choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        encoded_choices = [tokenizer.encode(s) for s in choices]
        cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

        input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

        outputs = model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

        """
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)

class HDiscriminator(nn.Module):
    def __init__(self, config, filter_widths=[1,2,3], conv_out_channel=128, dropout_prob=0.5): 
        super().__init__()
        nx = config.n_embd
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.dropout = nn.Dropout(dropout_prob)

        conv_layers = []
        for width in filter_widths:
            conv_layer = nn.Conv1d(nx, conv_out_channel, width, stride=1, padding=0)
            conv_layers.append(conv_layer)
        self.conv_layers = nn.ModuleList(conv_layers)

        self.maxpool = torch.nn.AdaptiveMaxPool1d(1) # maxpool input [N,C,L] to [N,C,1]
        self.fc = nn.Linear(conv_out_channel*len(filter_widths), 1, bias=False)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.config = config
        self.init_weights()

    def forward(self, x, labels=None):
        # x.shape: [N,L,C]
        conv_input = x.permute(0,2,1) # [N,C,L]
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_output = conv_layer(conv_input)
            conv_output = self.maxpool(conv_output) # [N,C,1]
            conv_output = self.relu(conv_output)
            conv_outputs.append(conv_output.reshape(-1, conv_output.shape[1]))
        
        fc_input = torch.cat(conv_outputs, dim=1)
        fc_input = self.dropout(fc_input)
        logits = self.fc(fc_input)
                            
        if labels is not None:
            loss = self.loss_fn(logits, labels) # logits, labels both have to be of shape [N,]
            output = (loss, logits)
        else:
            output = logits

        return output

    def init_weights(self):
        """ Initialize weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) and module.bias is not None:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)