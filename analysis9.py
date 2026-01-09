import json
import os
import glob
import pandas as pd
import re

# --- Codes couleurs ANSI pour le terminal ---
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[96m"  # Pour les headers

def load_metrics(diversity_file):
    metrics = {}
    directory = os.path.dirname(diversity_file)
    filename = os.path.basename(diversity_file)
    
    # Extraction du nom du modèle
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

    # 2. Charger Coherence (Likelihood)
    metrics['Coh_Likelihood'] = 0.0
    coh_files = glob.glob(os.path.join(directory, f"*{model_name}*coherence_result.json"))
    coh_files = [f for f in coh_files if 'simcse' not in f]
    
    if coh_files:
        try:
            with open(coh_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list): data = data[0]
                metrics['Coh_Likelihood'] = float(data.get('coherence_mean', 0))
        except: pass

    # 3. Charger Coherence (SimCSE)
    metrics['Coh_SimCSE'] = 0.0
    simcse_files = glob.glob(os.path.join(directory, f"*{model_name}*simcse_result.json"))
    
    if simcse_files:
        try:
            with open(simcse_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                metrics['Coh_SimCSE'] = float(data.get('coherence_mean', 0))
        except: pass

    metrics['Model'] = model_name 
    metrics['Perplexity'] = "N/A"
    return metrics

def format_value_markdown(val, is_best=False, color_mode=False):
    """Formate une valeur pour le Markdown (avec syntaxe LaTeX pour couleur)"""
    if isinstance(val, str): return val
    formatted = f"{val:.3f}"
    if color_mode and is_best:
        return f"$\\color{{red}}{{\\textsf{{{formatted}}}}}$"
    return formatted

def generate_markdown_table(df, colored=False):
    """Génère le code Markdown pour le rapport"""
    if df.empty: return "Aucune donnée."

    cols = ['Model', 'Coh_Likelihood', 'Coh_SimCSE', 'Diversity', 'MAUVE', 'Gen_Length', 'Perplexity']
    
    best_indices = {}
    best_mauve_idx = -1
    
    if not df.empty:
        for col in ['Coh_Likelihood', 'Coh_SimCSE', 'Diversity', 'MAUVE', 'Gen_Length']:
            if col in df.columns:
                best_indices[col] = df[col].idxmax()
        best_mauve_idx = df['MAUVE'].idxmax()
    
    markdown = "| " + " | ".join(cols) + " |\n"
    markdown += "|" + "|".join([" :--- " if i==0 else " :---: " for i in range(len(cols))]) + "|\n"
    
    for idx, row in df.iterrows():
        line = "|"
        
        # Modèle
        model_str = f"`{row['Model']}`"
        if colored and idx == best_mauve_idx:
            model_str = f"**`{row['Model']}`**"
        line += f" {model_str} |"
        
        # Métriques
        for col in cols[1:]:
            val = row[col]
            is_best = (idx == best_indices.get(col))
            formatted_val = format_value_markdown(val, is_best, colored)
            line += f" {formatted_val} |"
        markdown += line + "\n"
        
    return markdown

def print_aligned_terminal_table(df):
    """Affiche un tableau parfaitement aligné dans le terminal avec couleurs"""
    if df.empty: return

    # Colonnes à afficher
    cols_order = ['Model', 'Coh_Likelihood', 'Coh_SimCSE', 'Diversity', 'MAUVE', 'Gen_Length']
    
    # 1. Calculer les meilleurs indices
    best_indices = {}
    for col in cols_order[1:]: # Skip Model
        if col in df.columns:
            best_indices[col] = df[col].idxmax()
    best_mauve_idx = df['MAUVE'].idxmax()

    # 2. Calculer la largeur nécessaire pour chaque colonne (basé sur le texte brut sans couleur)
    col_widths = {col: len(col) for col in cols_order}
    formatted_data = [] # Stocke les données pré-formattées (texte brut, text coloré)

    for idx, row in df.iterrows():
        row_data = {}
        for col in cols_order:
            val = row[col]
            # Texte brut pour calculer la largeur
            if isinstance(val, float): raw_text = f"{val:.3f}"
            else: raw_text = str(val)
            
            # Mise à jour largeur max
            col_widths[col] = max(col_widths[col], len(raw_text))
            
            # Préparer le texte coloré
            is_best = (idx == best_indices.get(col))
            is_best_model = (col == 'Model' and idx == best_mauve_idx)
            
            if is_best:
                styled_text = f"{RED}{raw_text}{RESET}"
            elif is_best_model:
                styled_text = f"{BOLD}{raw_text}{RESET}"
            else:
                styled_text = raw_text
                
            row_data[col] = (raw_text, styled_text)
        formatted_data.append(row_data)

    # Ajouter un peu de marge (padding)
    for col in col_widths:
        col_widths[col] += 3 # 3 espaces de marge

    # 3. Affichage Header
    header = ""
    separator = ""
    for col in cols_order:
        # Alignement gauche pour Model, centré/droit pour chiffres (ici tout à gauche avec padding pour simplifier)
        header += f"{CYAN}{col:<{col_widths[col]}}{RESET}"
        separator += "-" * (col_widths[col] - 1) + " "
    
    print(separator)
    print(header)
    print(separator)

    # 4. Affichage Lignes
    for row in formatted_data:
        line_str = ""
        for col in cols_order:
            raw, styled = row[col]
            # L'astuce : on utilise la longueur du RAW pour calculer le padding, 
            # mais on affiche le STYLED.
            padding = " " * (col_widths[col] - len(raw))
            line_str += styled + padding
        print(line_str)
    print(separator)

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
        df = df.sort_values(by='Model').reset_index(drop=True)

        print("\n" + "="*70)
        print(f"{BOLD} APERÇU TERMINAL (Aligné){RESET}")
        print("="*70 + "\n")
        
        # Appel de la nouvelle fonction d'affichage
        print_aligned_terminal_table(df)
        
        print("\n" + "="*70)
        print(f"{BOLD} CODE MARKDOWN (Avec Couleurs - Pour Rapport){RESET}")
        print("="*70 + "\n")
        print(generate_markdown_table(df, colored=True))
        
        print("\n> **Légende :** $\color{red}{\\textsf{Rouge}}$ = Meilleur score. **`Nom_en_Gras`** = Meilleur Modèle (MAUVE).\n")
        
        print("="*70)
        print(f"{BOLD} CODE MARKDOWN (Simple - Pour Github){RESET}")
        print("="*70 + "\n")
        print(generate_markdown_table(df, colored=False))
    else:
        print("Erreur : Impossible de créer le dataframe.")

if __name__ == "__main__":
    main()