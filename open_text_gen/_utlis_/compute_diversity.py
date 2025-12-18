import json
import torch
import argparse
import numpy as np

def load_result(in_f):
    reference_list = []
    # On initialise une liste de listes pour correspondre à la structure attendue
    # (cas où il y aurait plusieurs prédictions par entrée, ici il n'y en a qu'une)
    all_prediction_list = [[]] 
    
    with open(in_f, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            # 'prefix' est utilisé comme référence (texte humain)
            if 'prefix' in item:
                reference_list.append(item['prefix'])
            
            # 'generated' est la prédiction
            if 'generated' in item:
                all_prediction_list[0].append(item['generated'])
                
    print(f'Number of predictions per instance is {len(all_prediction_list)}')
    return reference_list, all_prediction_list

###########################################################################################################
def eval_text(text, ngram):
    token_list = text.strip().split()
    start_idx, end_idx = 0, ngram
    total_num = 0
    ngram_set = set()
    while end_idx < len(token_list):
        one_ngram_list = token_list[start_idx:end_idx]
        assert len(one_ngram_list) == ngram
        one_ngram = ' '.join(one_ngram_list)
        total_num += 1
        ngram_set.add(one_ngram)
        start_idx += 1
        end_idx += 1
    return len(ngram_set), total_num

def eval_one_instance(text, ngram_list):
    res_dict = {}
    for n in ngram_list:
        n_unique, n_total = eval_text(text, n)
        res_dict[n] = {'unique':n_unique, 'total':n_total}
    unique_token_set = set(text.strip().split())
    return res_dict, unique_token_set

def measure_repetition_and_diversity(text_list):
    '''
        text_list: the list of text
    '''
    ngram_list = [2,3,4]
    pred_res_dict = {}
    for n in ngram_list:
        pred_res_dict[n] = {}
        pred_res_dict[n]['unique'] = 0
        pred_res_dict[n]['total'] = 0
    
    pred_unique_token_set = set()
    for text in text_list:
        text = text.strip('\n').strip()
        one_pred_res_dict, one_pred_uni_token_set = eval_one_instance(text, ngram_list)

        # unique token set
        pred_unique_token_set = pred_unique_token_set.union(one_pred_uni_token_set)
        # ngram statistic
        for n in ngram_list:
            pred_res_dict[n]['unique'] += one_pred_res_dict[n]['unique']
            pred_res_dict[n]['total'] += one_pred_res_dict[n]['total']

    # prediction result
    pred_seq_2 = 1 - (pred_res_dict[2]['unique']/pred_res_dict[2]['total'])
    pred_seq_2 = round(pred_seq_2 * 100, 2)
    pred_seq_3 = 1 - (pred_res_dict[3]['unique']/pred_res_dict[3]['total'])
    pred_seq_3 = round(pred_seq_3 * 100, 2)
    pred_seq_4 = 1 - (pred_res_dict[4]['unique']/pred_res_dict[4]['total'])
    pred_seq_4 = round(pred_seq_4 * 100, 2)
    pred_div = (1 - pred_seq_2/100) * (1 - pred_seq_3/100) * (1 - pred_seq_4/100)
    return pred_seq_2, pred_seq_3, pred_seq_4, pred_div
###########################################################################################################

def measure_diversity(in_f):
    reference_list, all_prediction_list = load_result(in_f)
    #from simctg.evaluation import measure_repetition_and_diversity
    if len(reference_list) > 0:
        _, _, _, reference_diversity = measure_repetition_and_diversity(reference_list)
        reference_diversity = round(reference_diversity*100, 2)
    else:
        reference_diversity = 0.0

    prediction_diversity_list = []
    for idx in range(len(all_prediction_list)):
        _, _, _, one_prediction_diversity = measure_repetition_and_diversity(all_prediction_list[idx])
        one_prediction_diversity = round(one_prediction_diversity*100, 2)
        prediction_diversity_list.append(one_prediction_diversity)

    pred_div_mean = np.mean(prediction_diversity_list)
    pred_div_std = np.std(prediction_diversity_list)

    result_dict = {
        'reference_div': str(reference_diversity),
        'prediction_diversity_list': [str(num) for num in prediction_diversity_list],
        'prediction_div_mean': str(pred_div_mean),
        'prediction_div_std': str(pred_div_std),
    }
    return result_dict