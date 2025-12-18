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
    min_len = min(len(reference_tokens), len(prediction_tokens))
    reference_tokens = reference_tokens[:128]#reference_tokens[:min_len]
    reference_text = decode(reference_tokens, tokenizer)
    prediction_tokens = prediction_tokens[:128]#prediction_tokens[:min_len]
    prediction_text = decode(prediction_tokens, tokenizer)
    '''
        Truncating the text to maximum token of 128 accoridng to the author's implementation:
        https://github.com/XiangLi1999/ContrastiveDecoding/blob/98cad19349fb08ee95b0f25a661179866f8e2c84/text-generation/eval_script.py#L228

        Only evaluate instances with exact 128 tokens length based on the author's implementation:
        https://github.com/XiangLi1999/ContrastiveDecoding/blob/98cad19349fb08ee95b0f25a661179866f8e2c84/text-generation/eval_script.py#L235
    '''
    if min(len(reference_tokens), len(prediction_tokens)) == 128:
        flag = True
    else:
        flag = False
    return reference_text, prediction_text, flag

def load_result(in_f, tokenizer):
    reference_list = []
    all_prediction_list = [[]] # Match the expected format for a single prediction run
    with open(in_f, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                # Ensure the line has the required keys and the generated text is not empty
                if 'generated' not in item or 'prefix' not in item or not item['generated'].strip():
                    continue
                
                # The 'prefix' is the reference text, 'generated' is the prediction
                reference_list.append(item['prefix'])
                all_prediction_list[0].append(item['generated'])
            except json.JSONDecodeError:
                # Skip malformed lines
                print(f"Avertissement: Ligne JSON malformée ignorée dans {in_f}")
                continue
    
    print(f'Number of predictions per instance is {len(all_prediction_list)}')
    return reference_list, all_prediction_list

def evaluate_one_instance(reference_list, prediction_list, tokenizer):
    ref_list, pred_list = [], []
    data_num = len(reference_list)
    for idx in range(data_num):
        one_ref, one_pred = reference_list[idx], prediction_list[idx]
        # We still call parse_text to get the tokenized & truncated versions, but ignore the flag
        one_ref, one_pred, _ = parse_text(one_ref, one_pred, tokenizer)
        
        # We ignore the flag and just check if both reference and prediction are non-empty
        if len(one_ref.strip()) > 0 and len(one_pred.strip()) > 0:
            ref_list.append(one_ref)
            pred_list.append(one_pred)
            
    if not ref_list or not pred_list:
        print("Avertissement : Aucune paire de texte valide trouvée pour l'évaluation MAUVE après filtrage. Retour de 0.0.")
        return 0.0

    # use gpt2 model as the based model based on the author's implementation:
    # https://github.com/XiangLi1999/ContrastiveDecoding/blob/98cad19349fb08ee95b0f25a661179866f8e2c84/text-generation/eval_script.py#L248
    out =  mauve.compute_mauve(p_text=ref_list, q_text=pred_list, device_id=0, verbose=False,
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
