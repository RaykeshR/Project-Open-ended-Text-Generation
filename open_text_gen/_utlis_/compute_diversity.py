import json
import argparse
import numpy as np

def load_result(in_f):
    """
    Charge un fichier JSONL (une ligne par objet JSON).
    Compatible avec les fichiers de génération ligne par ligne.
    """
    reference_list = []
    all_prediction_list = [[]] # Structure pour une seule liste de prédictions (greedy/nucleus standard)
    
    with open(in_f, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.strip()
                if not line: continue
                
                item = json.loads(line)
                
                # Vérification basique des clés
                if 'generated' in item and item['generated'].strip():
                    # 'prefix' est généralement le texte de référence
                    if 'prefix' in item:
                        reference_list.append(item['prefix'])
                    else:
                        # Fallback si pas de préfixe (rare)
                        reference_list.append("") 
                        
                    all_prediction_list[0].append(item['generated'])
            except json.JSONDecodeError:
                print(f"AVERTISSEMENT: Ligne ignorée dans {in_f} (JSON invalide)")
                continue
    
    print(f'Number of predictions per instance is {len(all_prediction_list)}')
    return reference_list, all_prediction_list

###########################################################################################################
def measure_repetition_and_diversity(text_list):
    '''
        diversity: 1.0 - repetition_rate
    '''
    dist_dict = {1:0, 2:0, 3:0, 4:0}
    pred_res_dict = {1:{'unique':0, 'total':0}, 2:{'unique':0, 'total':0}, 3:{'unique':0, 'total':0}, 4:{'unique':0, 'total':0}}
    for text in text_list:
        token_list = text.strip().split()
        n = len(token_list)
        for i in range(n):
            for j in range(1, 5):
                if i+j <= n:
                    ngram = tuple(token_list[i:i+j])
                    if ngram in dist_dict:
                        dist_dict[ngram] += 1
                    else:
                        dist_dict[ngram] = 1
                        
    for ngram, count in dist_dict.items():
        n = len(ngram)
        pred_res_dict[n]['total'] += count
        pred_res_dict[n]['unique'] += 1

    # Calcul des scores (sans condition if > 0 pour éviter de masquer des erreurs, ou avec protection si vide)
    def safe_div(unique, total):
        return 1.0 - (unique / total) if total > 0 else 0.0

    pred_seq_2 = safe_div(pred_res_dict[2]['unique'], pred_res_dict[2]['total'])
    pred_seq_2 = round(pred_seq_2 * 100, 2)
    
    pred_seq_3 = safe_div(pred_res_dict[3]['unique'], pred_res_dict[3]['total'])
    pred_seq_3 = round(pred_seq_3 * 100, 2)
    
    pred_seq_4 = safe_div(pred_res_dict[4]['unique'], pred_res_dict[4]['total'])
    pred_seq_4 = round(pred_seq_4 * 100, 2)
    
    pred_div = (1 - pred_seq_2/100) * (1 - pred_seq_3/100) * (1 - pred_seq_4/100)
    return pred_seq_2, pred_seq_3, pred_seq_4, pred_div

def measure_diversity(in_f):
    reference_list, all_prediction_list = load_result(in_f)
    
    if not reference_list:
        print("Erreur: Aucune donnée chargée.")
        return {}

    # 1. Diversité de la référence (si disponible)
    reference_diversity = 0.0
    if len(reference_list) > 0 and any(reference_list):
        _, _, _, reference_diversity = measure_repetition_and_diversity(reference_list)
        reference_diversity = round(reference_diversity*100, 2)

    # 2. Diversité des prédictions
    prediction_diversity_list = []
    rep2_list, rep3_list, rep4_list = [], [], []

    for idx in range(len(all_prediction_list)):
        # Récupération des 4 valeurs
        rep2, rep3, rep4, one_prediction_diversity = measure_repetition_and_diversity(all_prediction_list[idx])
        
        one_prediction_diversity = round(one_prediction_diversity*100, 2)
        prediction_diversity_list.append(one_prediction_diversity)
        
        # Stockage pour les moyennes
        rep2_list.append(rep2)
        rep3_list.append(rep3)
        rep4_list.append(rep4)

    pred_div_mean = np.mean(prediction_diversity_list)
    pred_div_std = np.std(prediction_diversity_list)
    
    # Calcul des moyennes des répétitions
    rep2_mean = np.mean(rep2_list) if rep2_list else 0.0
    rep3_mean = np.mean(rep3_list) if rep3_list else 0.0
    rep4_mean = np.mean(rep4_list) if rep4_list else 0.0

    print(f"Diversity: {pred_div_mean}, Rep-2: {rep2_mean}")

    result_dict = {
        'reference_div': str(reference_diversity),
        'prediction_diversity_list': [str(num) for num in prediction_diversity_list],
        'prediction_div_mean': str(pred_div_mean),
        'prediction_div_std': str(pred_div_std),
        'rep_2': str(round(rep2_mean, 2)),
        'rep_3': str(round(rep3_mean, 2)),
        'rep_4': str(round(rep4_mean, 2)),
    }
    return result_dict

if __name__ == '__main__':
    # Test simple si exécuté directement
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, help="Path to input file")
    args = parser.parse_args()
    if args.test_file:
        print(measure_diversity(args.test_file))