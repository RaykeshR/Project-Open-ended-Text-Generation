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
    match = re.search(r'wikitext_(.+?)_ollama', filename)
    model_name = match.group(1) if match else "Unknown"
    
    # 1. Charger Diversity / MAUVE / Gen Length
    try:
        with open(diversity_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0: 
                data = data[0]
            
            def get_val(d, k1, k2):
                val = d.get(k1, {}).get(k2, 0)
                try: return float(val)
                except: return 0.0

            metrics['MAUVE'] = get_val(data, 'mauve_dict', 'mauve_mean')
            metrics['Gen_Length'] = get_val(data, 'gen_length_dict', 'gen_len_mean')
            metrics['Diversity'] = get_val(data, 'diversity_dict', 'prediction_div_mean')
    except Exception as e:
        metrics['MAUVE'] = 0.0
        metrics['Gen_Length'] = 0.0
        metrics['Diversity'] = 0.0

    # 2. Charger Coherence (Likelihood / OPT) - Ancien fichier
    # Pattern: *coherence_result.json (excluant simcse)
    metrics['Coh_Likelihood'] = 0.0
    # On cherche tous les fichiers coherence pour ce modèle
    coh_files = glob.glob(os.path.join(directory, f"*{model_name}*coherence_result.json"))
    # On s'assure de ne pas prendre un fichier qui contiendrait 'simcse' par erreur, bien que le nommage soit différent
    coh_files = [f for f in coh_files if 'simcse' not in f]
    
    if coh_files:
        try:
            with open(coh_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list): data = data[0]
                metrics['Coh_Likelihood'] = float(data.get('coherence_mean', 0))
        except: pass

    # 3. Charger Coherence (SimCSE) - Nouveau fichier
    metrics['Coh_SimCSE'] = 0.0
    simcse_files = glob.glob(os.path.join(directory, f"*{model_name}*simcse_result.json"))
    
    if simcse_files:
        try:
            with open(simcse_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Le script simcse sauvegarde un dict direct, pas une liste
                metrics['Coh_SimCSE'] = float(data.get('coherence_mean', 0))
        except: pass

    metrics['Model'] = f"`{model_name}`"
    metrics['Perplexity'] = "N/A" # Pas de PPL pour Ollama ici
    return metrics

def format_value(val, is_best=False, color_mode=False):
    if isinstance(val, str):
        return val
    
    formatted = f"{val:.3f}"
    
    if color_mode and is_best:
        return f"$\\color{{red}}{{\\textsf{{{formatted}}}}}$"
    return formatted

def generate_table(df, colored=False):
    if df.empty:
        return "Aucune donnée disponible."

    # Colonnes à afficher
    cols = ['Model', 'Coh_Likelihood', 'Coh_SimCSE', 'Diversity', 'MAUVE', 'Gen_Length', 'Perplexity']
    
    # Identifier les meilleurs scores
    best_indices = {}
    best_mauve_idx = -1
    
    if not df.empty:
        # On suppose que plus c'est haut mieux c'est pour ces métriques
        for col in ['Coh_Likelihood', 'Coh_SimCSE', 'Diversity', 'MAUVE', 'Gen_Length']:
            best_indices[col] = df[col].idxmax()
            
        best_mauve_idx = df['MAUVE'].idxmax()
    
    # Header Markdown
    markdown = "| " + " | ".join(cols) + " |\n"
    markdown += "|" + "|".join([" :--- " if i==0 else " :---: " for i in range(len(cols))]) + "|\n"
    
    # Lignes
    for idx, row in df.iterrows():
        line = "|"
        
        # Nom du modèle
        model_str = row['Model']
        if colored and idx == best_mauve_idx:
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

    files = glob.glob(os.path.join(results_dir, '*_diversity_mauve_gen_length_result.json'))
    
    if not files:
        print("Aucun fichier de résultats trouvé.")
        return

    data_rows = []
    for f_path in files:
        metrics = load_metrics(f_path)
        data_rows.append(metrics)
        
    df = pd.DataFrame(data_rows)
    
    if not df.empty:
        df = df.sort_values(by='Model')
        df = df.reset_index(drop=True)

        print("\n### Résultats Ollama (Double Cohérence)\n")
        
        print("(Version avec couleur)\n")
        print(generate_table(df, colored=True))
        
        print("\n> **Légende :** $\color{red}{\\textsf{Rouge}}$ = Meilleur score. **`Nom_en_Gras`** = Meilleur Modèle (MAUVE).")
        print("> **Coh_Likelihood** : Log-vraisemblance moyenne (OPT). **Coh_SimCSE** : Similarité sémantique (SimCSE).\n")
        
        print("(Version sans couleur)\n")
        print(generate_table(df, colored=False))
    else:
        print("Erreur : Impossible de créer le dataframe.")

if __name__ == "__main__":
    main()