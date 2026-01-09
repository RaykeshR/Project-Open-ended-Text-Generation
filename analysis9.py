import json
import os
import glob
import pandas as pd
import re

def load_metrics(diversity_file):
    metrics = {}
    directory = os.path.dirname(diversity_file)
    filename = os.path.basename(diversity_file)
    
    # Extraction du nom du modèle
    # Ex: wikitext_llama3.2_ollama_opt-125m... -> llama3.2
    # Ex: wikitext_mistral_ollama... -> mistral
    match = re.search(r'wikitext_(.+?)_ollama', filename)
    model_name = match.group(1) if match else filename
    
    # 1. Charger Diversity / MAUVE / Gen Length
    try:
        with open(diversity_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0: 
                data = data[0]
            
            # Gestion safe des types (float/str)
            def get_val(d, k1, k2):
                val = d.get(k1, {}).get(k2, 0)
                try:
                    return float(val)
                except:
                    return 0.0

            metrics['MAUVE'] = get_val(data, 'mauve_dict', 'mauve_mean')
            metrics['Gen_Length'] = get_val(data, 'gen_length_dict', 'gen_len_mean')
            metrics['Diversity'] = get_val(data, 'diversity_dict', 'prediction_div_mean')
    except Exception as e:
        metrics['MAUVE'] = 0.0
        metrics['Gen_Length'] = 0.0
        metrics['Diversity'] = 0.0

    # 2. Chercher le fichier de Cohérence
    # Pattern: wikitext_<model>_ollama.*coherence_result.json
    prefix = f"wikitext_{model_name}_ollama"
    # On cherche large pour trouver le fichier coherence associé
    possible_coh_files = glob.glob(os.path.join(directory, f"*{model_name}*coherence_result.json"))
    
    metrics['SimCSE'] = 0.0 # Default
    if possible_coh_files:
        target_coh = possible_coh_files[0]
        try:
            with open(target_coh, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0: 
                    data = data[0]
                val = data.get('coherence_mean', 0)
                metrics['SimCSE'] = float(val)
        except:
            pass

    metrics['Model'] = f"`{model_name}`"
    metrics['Perplexity'] = "N/A" # Pas de PPL pour Ollama dans ces fichiers
    return metrics

def format_value(val, is_best=False, color_mode=False):
    if isinstance(val, str):
        return val
    
    formatted = f"{val:.3f}"
    
    if color_mode and is_best:
        return f"$\\color{{red}}{{\\textsf{{{formatted}}}}}$"
    return formatted

def generate_table(df, colored=False):
    # Colonnes
    cols = ['Model', 'SimCSE', 'Diversity', 'MAUVE', 'Gen_Length', 'Perplexity']
    
    # Identifier les meilleurs scores (Max pour tout sauf PPL, ici PPL est N/A)
    best_indices = {}
    if not df.empty:
        for col in ['SimCSE', 'Diversity', 'MAUVE', 'Gen_Length']:
            best_indices[col] = df[col].idxmax()
            
        # Trouver le modèle avec le meilleur MAUVE global pour le mettre en gras
        best_mauve_idx = df['MAUVE'].idxmax()
    
    # Header
    markdown = "| " + " | ".join(cols) + " |\n"
    markdown += "|" + "|".join([" :--- " if i==0 else " :---: " for i in range(len(cols))]) + "|\n"
    
    # Rows
    for idx, row in df.iterrows():
        line = "|"
        
        # Nom du modèle (Gras si meilleur MAUVE et mode couleur)
        model_str = row['Model']
        if colored and idx == best_mauve_idx:
            # Enlever les backticks existants pour mettre en gras proprement
            clean_name = model_str.replace('`', '')
            model_str = f"**`{clean_name}`**"
        line += f" {model_str} |"
        
        # Autres colonnes
        for col in cols[1:]:
            val = row[col]
            is_best = (idx == best_indices.get(col)) if col in best_indices else False
            
            formatted_val = format_value(val, is_best, colored)
            line += f" {formatted_val} |"
        markdown += line + "\n"
        
    return markdown

def main():
    results_dir = 'open_text_gen/ollama_results'
    
    if not os.path.exists(results_dir):
        print(f"Le dossier {results_dir} n'existe pas.")
        return

    # Fichiers sources
    files = glob.glob(os.path.join(results_dir, '*_diversity_mauve_gen_length_result.json'))
    
    if not files:
        print("Aucun fichier de résultats trouvé.")
        return

    data_rows = []
    for f_path in files:
        metrics = load_metrics(f_path)
        data_rows.append(metrics)
        
    df = pd.DataFrame(data_rows)
    df = df.sort_values(by='Model')
    df = df.reset_index(drop=True)

    print("\n### Résultats Ollama\n")
    
    print("(Version avec couleur)\n")
    print(generate_table(df, colored=True))
    
    print("\n> **Légende :** $\color{red}{\\textsf{Rouge}}$ = Meilleur score. **`Nom_en_Gras`** = Meilleur Modèle (MAUVE).\n")
    
    print("(Version sans couleur)\n")
    print(generate_table(df, colored=False))

if __name__ == "__main__":
    main()