import json
import torch
import mauve 
import argparse
import numpy as np
from transformers import AutoTokenizer

def decode(tokens, tokenizer):
    token_id_list = tokenizer.convert_tokens_to_ids(tokens)
    text = tokenizer.decode(token_id_list)
    return text

def parse_text(reference_text, prediction_text, tokenizer):
    reference_tokens = tokenizer.tokenize(reference_text)
    prediction_tokens = tokenizer.tokenize(prediction_text)
    # Augmentation de la fenêtre de lecture
    # On passe à 256 pour être sûr d'inclure le prompt + la génération
    MAX_LEN = 256 
    reference_tokens = reference_tokens[:MAX_LEN]
    prediction_tokens = prediction_tokens[:MAX_LEN]
    
    reference_text = decode(reference_tokens, tokenizer)
    prediction_text = decode(prediction_tokens, tokenizer)
    
    # On garde le filtre > 10 tokens (le texte fera maintenant ~60 tokens avec le prompt)
    flag = True if min(len(reference_tokens), len(prediction_tokens)) > 10 else False
    return reference_text, prediction_text, flag

def load_result(in_f, tokenizer=None):
    reference_list = []
    all_prediction_list = [[]]
    
    print(f"Chargement des résultats depuis {in_f}...")
    with open(in_f, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            # Récupération des champs
            # Le script generate.py sauvegarde le prompt séparément, on le récupère ici
            prompt = item.get('prompt') or item.get('prefix_text') or ""
            ref_only = item.get('gold_ref') or item.get('reference_text') or ""
            gen_only = item.get('gen_text') or item.get('generated') or ""
            
            # CONCATÉNATION DU PROMPT
            # On recolle le prompt devant la référence et la génération
            if prompt:
                # On ajoute un espace pour éviter de coller les mots
                full_ref = (prompt + " " + ref_only).strip()
                full_gen = (prompt + " " + gen_only).strip()
            else:
                full_ref = ref_only
                full_gen = gen_only
            
            reference_list.append(full_ref)
            all_prediction_list[0].append(full_gen)
            
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

    # On spécifie max_text_length=256 pour que le modèle prenne tout en compte
    out = mauve.compute_mauve(p_text=ref_list, q_text=pred_list, device_id=0, verbose=False,
        featurize_model_name='gpt2',max_text_length=256)
    mauve_score = out.mauve
    return mauve_score*100

def measure_mauve(in_f):
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