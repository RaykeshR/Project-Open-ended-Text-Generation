import json
import os
import glob
import pandas as pd
import re
import numpy as np

# --- 1. FONCTIONS DE CALCUL (DIVERSITÉ) ---
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
    return len(ngram_set), total_num if total_num > 0 else 1

def eval_one_instance(text, ngram_list):
    res_dict = {}
    for n in ngram_list:
        n_unique, n_total = eval_text(text, n)
        res_dict[n] = {'unique':n_unique, 'total':n_total}
    unique_token_set = set(text.strip().split())
    return res_dict, unique_token_set

def calculate_diversity_metrics(text_list):
    ngram_list = [2, 3, 4]
    pred_res_dict = {n: {'unique': 0, 'total': 0} for n in ngram_list}
    
    for text in text_list:
        text = text.strip('\n').strip()
        one_pred_res_dict, _ = eval_one_instance(text, ngram_list)
        for n in ngram_list:
            pred_res_dict[n]['unique'] += one_pred_res_dict[n]['unique']
            pred_res_dict[n]['total'] += one_pred_res_dict[n]['total']

    rep_2 = 1 - (pred_res_dict[2]['unique'] / pred_res_dict[2]['total']) if pred_res_dict[2]['total'] > 0 else 0
    rep_3 = 1 - (pred_res_dict[3]['unique'] / pred_res_dict[3]['total']) if pred_res_dict[3]['total'] > 0 else 0
    rep_4 = 1 - (pred_res_dict[4]['unique'] / pred_res_dict[4]['total']) if pred_res_dict[4]['total'] > 0 else 0
    
    return {
        'rep-2': round(rep_2, 4),
        'rep-3': round(rep_3, 4),
        'rep-4': round(rep_4, 4)
    }

def load_predictions(filepath):
    predictions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                predictions.append(item.get('gen_text') or item.get('generated') or "")
            except: pass
    return predictions

# --- 2. FONCTION D'AFFICHAGE (COULEURS LATEX/MARKDOWN) ---
def print_colored_table(df):
    if df.empty:
        print("Aucune donnée à afficher.")
        return

    # Définition des critères (Min ou Max)
    col_criteria = {
        'MAUVE': 'max',
        'Gen_Length': 'max',
        'Perplexity': 'min', # Plus bas = Meilleur
        'rep-2': 'min', 'rep-3': 'min', 'rep-4': 'min'
    }
    # Les scores de cohérence sont toujours à maximiser
    for col in df.columns:
        if 'Coh_' in col: col_criteria[col] = 'max'

    # Calcul des meilleures valeurs
    best_values = {}
    for col in df.columns:
        if col in col_criteria:
            if col_criteria[col] == 'max':
                best_values[col] = df[col].max()
            else:
                best_values[col] = df[col].min()

    # Meilleure ligne globale (basée sur MAUVE)
    best_overall_idx = df['MAUVE'].idxmax() if 'MAUVE' in df.columns else -1

    headers = df.columns.tolist()
    
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for idx, row in df.iterrows():
        row_str = "|"
        
        # Couleur de la ligne (Vert si meilleur MAUVE global)
        row_color = "green" if idx == best_overall_idx else None

        for col in headers:
            val = row[col]
            val_str = f"{val:.4f}" if isinstance(val, (float, int)) else str(val)

            # Est-ce la meilleure valeur de la colonne ?
            is_best_col = False
            if col in best_values and pd.notna(val):
                if np.isclose(val, best_values[col]):
                    is_best_col = True

            final_str = val_str
            
            # Priorité formatage : Rouge (Best Cell) > Couleur Ligne > Normal
            if is_best_col:
                final_str = f"$\\color{{red}}{{\\textsf{{{val_str}}}}}$"
            elif row_color:
                final_str = f"$\\color{{{row_color}}}{{\\textsf{{{val_str}}}}}$"
            else:
                final_str = f"$\\textsf{{{val_str}}}$"

            row_str += f" {final_str} |"

        print(row_str)

# --- 3. PARSING ET MAIN ---
def parse_epsilon_filename(filename):
    # Pattern: wikitext_epsilon_k10_alpha0.2_gpt2-xl.jsonl
    # Regex: k(\d+) ... alpha([\d.]+) ... ([^_]+)\.jsonl
    match = re.search(r'epsilon_k(\d+)_alpha([\d.]+)_([^\.]+)', filename)
    if match:
        k, alpha, model = match.groups()
        return int(k), float(alpha), model
    return None, None, None

def main():
    results_dir = 'open_text_gen/wikitext_epsilon_grid_search'
    print(f"Analyse des résultats Epsilon dans : {results_dir}")

    jsonl_files = glob.glob(os.path.join(results_dir, '*.jsonl'))
    rows = []

    for f_path in jsonl_files:
        basename = os.path.basename(f_path)
        base_no_ext = os.path.splitext(f_path)[0]
        
        k, alpha, model = parse_epsilon_filename(basename)
        if k is None: continue

        row_data = {
            'Model': model,
            'k': k,
            'alpha': alpha
        }

        # 1. Diversité (calculée depuis le jsonl)
        preds = load_predictions(f_path)
        if preds:
            row_data.update(calculate_diversity_metrics(preds))

        # 2. MAUVE / Gen Length
        div_file = base_no_ext + '_diversity_mauve_gen_length_result.json'
        if os.path.exists(div_file):
            try:
                with open(div_file, 'r') as f: d = json.load(f)[0]
                row_data['MAUVE'] = float(d.get('mauve_dict', {}).get('mauve_mean', 0))
                row_data['Gen_Length'] = float(d.get('gen_length_dict', {}).get('gen_len_mean', 0))
            except: pass

        # 3. Perplexité (IMPORTANT)
        ppl_file = base_no_ext + '_perplexity_result.json'
        if os.path.exists(ppl_file):
            try:
                with open(ppl_file, 'r') as f: d = json.load(f)
                if isinstance(d, list): d = d[0]
                val = d.get('mean_perplexity') or d.get('ppl') or d.get('perplexity')
                if val: row_data['Perplexity'] = float(val)
            except: pass

        # 4. Cohérence (tous les fichiers correspondants)
        coh_files = glob.glob(base_no_ext + '*_coherence_result.json')
        for c_file in coh_files:
            try:
                # Extraire nom du modèle de cohérence (ex: _opt-125m)
                suffix = c_file.replace(base_no_ext, '').replace('_coherence_result.json', '').strip('._')
                with open(c_file, 'r') as f: d = json.load(f)[0]
                row_data[f'Coh_{suffix}'] = float(d.get('coherence_mean', 0))
            except: pass

        rows.append(row_data)

    if not rows:
        print("Aucun fichier trouvé.")
        return

    # Création et Tri du DataFrame
    df = pd.DataFrame(rows)
    
    # Colonnes à afficher en priorité
    cols_order = ['k', 'alpha', 'MAUVE', 'Perplexity', 'Gen_Length']
    # Ajout des autres colonnes dynamiquement
    cols_order += [c for c in df.columns if c not in cols_order and c != 'Model']
    
    # On garde Model s'il y a plusieurs modèles, sinon on peut l'enlever pour épurer
    if len(df['Model'].unique()) > 1:
        cols_order.insert(0, 'Model')

    # Filtrer pour ne garder que les colonnes existantes
    final_cols = [c for c in cols_order if c in df.columns]
    df = df[final_cols]

    # Tri par k puis alpha
    df = df.sort_values(by=['k', 'alpha']).reset_index(drop=True)

    print("\n" + "="*120)
    print(" RÉSULTATS : EPSILON SAMPLING (Avec Perplexité)")
    print(" Légende : Rouge = Meilleur de la colonne | Vert = Meilleure configuration (selon MAUVE)")
    print("="*120)
    
    print_colored_table(df)
    print("="*120)

if __name__ == '__main__':
    main()