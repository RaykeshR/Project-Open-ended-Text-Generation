import json
import numpy as np
def load_result(in_f):
    all_prediction_list = []
    with open(in_f, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            if not all_prediction_list:
                # Initialize based on the number of generated results in the first item
                num_predictions = len(item['generated_result'])
                all_prediction_list = [[] for _ in range(num_predictions)]
            
            for idx in range(len(all_prediction_list)):
                all_prediction_list[idx].append(item['generated_result'][str(idx)])
    
    print(f'Number of prediction sets is {len(all_prediction_list)}')
    return all_prediction_list
    
def compute_one_gen_len(text_list):
    all_len = 0.
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