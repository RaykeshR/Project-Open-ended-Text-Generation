import json
import torch
import mauve 
import argparse
import numpy as np

def decode(tokens, tokenizer):
    token_id_list = tokenizer.convert_tokens_to_ids(tokens)
    text = tokenizer.decode(token_id_list)
    return text

def parse_text(reference_text, prediction_text, tokenizer):
    reference_tokens = tokenizer.tokenize(reference_text)
    prediction_tokens = tokenizer.tokenize(prediction_text)
    # min_len = min(len(reference_tokens), len(prediction_tokens)) # Non utilisé
    # On tronque à 128 maximum, mais on accepte les textes plus courts (ex: 50)
    reference_tokens = reference_tokens[:128]
    prediction_tokens = prediction_tokens[:128]
    
    reference_text = decode(reference_tokens, tokenizer)
    prediction_text = decode(prediction_tokens, tokenizer)
    
    # On accepte tout texte ayant au moins 10 tokens
    flag = True if min(len(reference_tokens), len(prediction_tokens)) > 10 else False
    return reference_text, prediction_text, flag

def load_result(in_f, tokenizer=None):
    reference_list = []
    all_prediction_list = [[]]
    
    with open(in_f, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            # Utilisation des nouvelles clés de generate.py
            ref = item.get('gold_ref') or item.get('reference_text') or ""
            gen = item.get('gen_text') or item.get('generated') or ""
            reference_list.append(ref)
            all_prediction_list[0].append(gen)
    return reference_list, all_prediction_list

def evaluate_one_instance(reference_list, prediction_list, tokenizer):
    ref_list, pred_list = [], []
    data_num = len(reference_list)
    for idx in range(data_num):
        one_ref, one_pred = reference_list[idx], prediction_list[idx]
        one_ref, one_pred, flag = parse_text(one_ref, one_pred, tokenizer)
        if flag:
            pass
        else:
            continue
        if len(one_pred.strip()) > 0: # ignore predictions with zero length
            ref_list.append(one_ref)
            pred_list.append(one_pred)
            
    if not ref_list or not pred_list:
        return 0.0

    out = mauve.compute_mauve(p_text=ref_list, q_text=pred_list, device_id=0, verbose=False,
        featurize_model_name='gpt2')
    mauve_score = out.mauve
    return mauve_score*100

def measure_mauve(in_f):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    reference_list, all_prediction_list = load_result(in_f, tokenizer)

    mauve_score_list = []
    for idx in range(len(all_prediction_list)):
        one_prediction_list = all_prediction_list[idx]
        one_mauve_score = evaluate_one_instance(reference_list, one_prediction_list, tokenizer)
        mauve_score_list.append(one_mauve_score)

    mean, std = round(np.mean(mauve_score_list),2), round(np.std(mauve_score_list),2)
    result_dict = {
        "mauve_score_list": [str(num) for num in mauve_score_list],
        'mauve_mean': str(mean),
        'mauve_std': str(std)
    }
    return result_dict