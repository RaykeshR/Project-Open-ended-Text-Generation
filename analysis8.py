import json
import os
import glob
import pandas as pd
import re
import numpy as np

def parse_epsilon_filename(filename):
    # Récupère k et alpha depuis le nom du fichier
    match = re.search(r'epsilon_k(\d+)_alpha([\d.]+)', filename)
    if match:
        k, alpha = match.groups()
        return int(k), float(alpha)
    return None, None

def load_metrics(base_path):
    metrics = {}
    
    # 1. MAUVE, Diversity, Gen Length
    div_file = base_path + '_diversity_mauve_gen_length_result.json'
    if os.path.exists(div_file):
        try:
            with open(div_file, 'r') as f:
                data = json.load(f)[0]
                metrics['mauve'] = float(data.get('mauve_dict', {}).get('mauve_mean', 0))
                metrics['gen_length'] = float(data.get('gen_length_dict', {}).get('gen_len_mean', 0))
                metrics['diversity'] = float(data.get('diversity_dict', {}).get('prediction_div_mean', 0))
        except: pass

    # 2. Perplexité
    ppl_file = base_path + '_perplexity_result.json'
    if os.path.exists(ppl_file):
        try:
            with open(ppl_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list): data = data[0]
                metrics['perplexity'] = float(data.get('mean_perplexity') or data.get('ppl', 0))
        except: pass

    # 3. Cohérence (Prend le premier fichier dispo, priorité à opt-2.7b)
    coh_files = glob.glob(base_path + '*_coherence_result.json')
    if coh_files:
        try:
            target_file = next((f for f in coh_files if 'opt-2.7b' in f), coh_files[0])
            with open(target_file, 'r') as f:
                data = json.load(f)[0]
                metrics['coherence_score'] = float(data.get('coherence_mean', -999))
        except: pass
            
    return metrics

def print_markdown_table(df):
    if df.empty:
        print("Aucune donnée.")
        return

    # Ordre des colonnes demandé
    cols = ['k', 'alpha', 'gen_length', 'coherence_score', 'diversity', 'mauve', 'perplexity']
    
    # Critères "Meilleur" (Max ou Min)
    criteria = {
        'coherence_score': 'max',
        'diversity': 'max',
        'mauve': 'max',
        'perplexity': 'min'
        # gen_length n'est généralement pas "noté" par max/min dans ce tableau
    }

    # Calcul des meilleures valeurs
    best_values = {}
    for c in criteria:
        if c in df.columns:
            if criteria[c] == 'max':
                best_values[c] = df[c].max()
            else:
                best_values[c] = df[c].min()

    # Affichage En-tête Markdown
    header = "| " + " | ".join(cols) + " |"
    separator = "|" + "|".join([" :---: " for _ in cols]) + "|"
    
    print(header)
    print(separator)

    # Affichage Lignes
    for _, row in df.iterrows():
        line = "|"
        for col in cols:
            val = row.get(col, None)
            
            if pd.isna(val):
                line += " - |"
                continue

            # Formatage nombre
            if col in ['k']:
                val_str = f"{val}" # Entier pour k
            else:
                val_str = f"{val:.2f}" # 2 décimales pour le reste

            # Mise en gras si c'est la meilleure valeur
            if col in best_values and np.isclose(val, best_values[col], atol=1e-3):
                val_str = f"**{val_str}**"
            
            line += f" {val_str} |"
        print(line)

def main():
    results_dir = 'open_text_gen/wikitext_epsilon_grid_search'
    files = glob.glob(os.path.join(results_dir, '*.jsonl'))
    
    data_rows = []
    for f_path in files:
        base_path = os.path.splitext(f_path)[0]
        k, alpha = parse_epsilon_filename(os.path.basename(f_path))
        
        if k is None: continue
        
        metrics = load_metrics(base_path)
        metrics['k'] = k
        metrics['alpha'] = alpha
        data_rows.append(metrics)
        
    df = pd.DataFrame(data_rows)
    
    if not df.empty:
        # Tri par k puis alpha
        df = df.sort_values(by=['k', 'alpha'])
        print_markdown_table(df)
    else:
        print("Aucun fichier trouvé.")

if __name__ == "__main__":
    main()