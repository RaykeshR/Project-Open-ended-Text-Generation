import json
import numpy as np

def load_result(in_f):
    all_prediction_list = [[]]
    with open(in_f, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            if 'generated' in item:
                all_prediction_list[0].append(item['generated'])
                
    print(f'Number of predictions per instance is {len(all_prediction_list)}')
    return all_prediction_list
    
def compute_one_gen_len(text_list):
    all_len = 0.
    if len(text_list) == 0:
        return 0.
    for text in text_list:
        all_len += len(text.strip().split())
    return all_len / len(text_list)
    
def measure_gen_length(in_f):
    all_prediction_list = load_result(in_f)
    len_list = []
    for one_prediction_list in all_prediction_list:
        len_list.append(compute_one_gen_len(one_prediction_list))
    result_dict = {
        'gen_len_list': [str(num) for num in len_list],
        'gen_len_mean': str(round(np.mean(len_list),2)),
        'gen_len_std': str(round(np.std(len_list),2))
    }
    return result_dict