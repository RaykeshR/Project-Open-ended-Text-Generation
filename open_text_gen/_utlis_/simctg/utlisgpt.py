import sys
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import argparse
import random

# ========== batch version ========= #
def ranking_fast(context_hidden, next_hidden, next_top_k_probs, alpha, beam_width):
    '''
        context_hidden: bsz*beam x seqlen x embed_dim
        next_hidden: bsz*beam x 1 x embed_dim
        next_top_k_probs: bsz x beam
    '''
    _, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)    # [B*K, S]
    scores, _ = torch.max(cosine_matrix, dim=-1)    # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)    # [B*K]
    scores = (1.0 - alpha) * next_top_k_probs - alpha * scores 
    scores = torch.stack(torch.split(scores, beam_width))    # [B, K]
    selected_idx = scores.max(dim=-1)[1]    # [B]
    return selected_idx

def ContrastiveDecodingOneStepFast(
    model, 
    ids, 
    beam_width, 
    alpha, 
    past_key_values,
    last_hidden_states,
    vocab,
    logit_for_next_step,
    first_step=False,
    ):
    # input_ids: [B, S]
    if first_step:
        output = model(
            input_ids=ids, 
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = output.past_key_values
        last_hidden_states = output.hidden_states[-1]    # [B, S, E]
        logit_for_next_step = output.logits[:, -1, :]    # [B, V]
    bsz, seqlen, embed_dim = last_hidden_states.size()
    p = random.uniform(0, 1)

    next_probs = F.softmax(logit_for_next_step, dim=-1)
    _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)    # [B, K]
    top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)    # [B, K]
    # compute new hidden
    past_key_values = enlarge_past_key_values(past_key_values, beam_width)
    output = model(
        input_ids=top_k_ids.view(-1, 1), 
        attention_mask=torch.ones_like(top_k_ids.view(-1, 1)),
        past_key_values=past_key_values,
        output_hidden_states=True,
        use_cache=True,
    )
    past_key_values = output.past_key_values
    logits = output.logits[:, -1, :]    # [B*K, V]
    next_hidden = output.hidden_states[-1]    # [B*K, 1, E]
    context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz*beam_width, seqlen, embed_dim)    # [B*K, S, E]

    selected_idx = ranking_fast(
        context_hidden, 
        next_hidden, 
        top_k_probs,    # [B, K] 
        alpha,
        beam_width,
    )     # [B]
    # prepare for the next step
    next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)    # [B, 1]
    next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))    # [B, K, E]
    next_hidden = next_hidden[range(bsz), selected_idx, :]    # [B, E]
    last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)    # [B, S, E]
    past_key_values = select_past_key_values(past_key_values, beam_width, selected_idx)
    logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]    # [B, V]
    # next_id: [B, 1]
    return next_id, past_key_values, last_hidden_states, logits 

def enlarge_past_key_values(past_key_values, beam_width):
    # from [B, num_head, seq_len, esz] to [B*K, num_head, seq_len, esz]
    new_key_values = []
    for layer in past_key_values: # layer is a tuple (key_tensor, value_tensor)
        items = []
        for item_tensor in layer: # item_tensor is either key or value tensor
            bsz, num_head, seq_len, esz = item_tensor.size()
            item_tensor = item_tensor.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(bsz*beam_width, num_head, seq_len, esz)
            items.append(item_tensor)
        new_key_values.append(tuple(items)) # Make it a tuple
    return tuple(new_key_values) # Make it a tuple

def select_past_key_values(past_key_values, beam_width, selected_idx):
    '''select_idx: [B]'''
    new_key_values = []
    for layer in past_key_values: # layer is a tuple (key_tensor, value_tensor)
        items = []
        for item_tensor in layer: # item_tensor is either key or value tensor
            bsz_and_beam, num_head, seq_len, esz = item_tensor.size()
            bsz = int(bsz_and_beam//beam_width)
            item_tensor = torch.stack(torch.split(item_tensor, beam_width, dim=0))    # [B, K, num_head, seq_len, esz] 
            item_tensor = item_tensor[range(bsz), selected_idx, :, :, :]   # [B, num_head, seq_len, esz]
            items.append(item_tensor)
        new_key_values.append(tuple(items)) # Make it a tuple
    return tuple(new_key_values) # Make it a tuple
