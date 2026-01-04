import os
import json
import pandas as pd
import glob
import numpy as np

# --- 1. FONCTION D'AFFICHAGE COLORÉ ---
def print_highlighted_table(df):
    """
    Affiche le DataFrame avec des couleurs ANSI :
    - Rouge : Meilleur résultat de la colonne.
    - Vert : Meilleure ligne globale (Basé sur le score MAUVE).
    """
    if df.empty:
        print("Tableau vide.")
        return

    # Codes ANSI
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # 1. Définir la direction du "Meilleur" pour chaque métrique
    col_criteria = {
        'MAUVE': 'max',
        'mauve': 'max',
        'diversity': 'max',
        'coherence_score': 'max',
        'gen_length': 'max' # Discutable, mais on va dire que plus long = mieux pour l'instant
    }

    # 2. Calculer les meilleures valeurs par colonne
    best_values = {}
    for col in df.columns:
        if col in col_criteria:
            if pd.api.types.is_numeric_dtype(df[col]):
                if col_criteria[col] == 'max':
                    best_values[col] = df[col].max()
                else:
                    best_values[col] = df[col].min()

    # 3. Identifier la meilleure ligne (Gagnant selon MAUVE)
    best_overall_idx = -1
    sort_metric = 'mauve' if 'mauve' in df.columns else ('MAUVE' if 'MAUVE' in df.columns else None)
    
    if sort_metric and sort_metric in df.columns and pd.api.types.is_numeric_dtype(df[sort_metric]):
        best_overall_idx = df[sort_metric].idxmax()

    # 4. Formatage et Affichage
    headers = df.columns.tolist()
    
    # Prépare les formateurs de texte
    formatters = {}
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            formatters[col] = "{:.2f}"
        else:
            formatters[col] = "{}"

    # Calcul des largeurs de colonnes
    col_widths = {col: len(col) for col in headers}
    for idx, row in df.iterrows():
        for col in headers:
            val = row[col]
            if pd.notnull(val):
                val_str = formatters[col].format(val)
                col_widths[col] = max(col_widths[col], len(val_str))
    
    # Ajout d'un peu de marge
    for col in col_widths:
        col_widths[col] += 2

    # Affichage En-tête
    header_str = "".join([h.ljust(col_widths[h]) for h in headers])
    print("-" * len(header_str))
    print(BOLD + header_str + RESET)
    print("-" * len(header_str))

    # Affichage Lignes
    for idx, row in df.iterrows():
        row_str = ""
        is_green = (idx == best_overall_idx)
        
        # Si c'est la meilleure ligne, on la prépare en vert, sinon normal
        row_color_start = GREEN if is_green else ""
        row_color_end = RESET if is_green else ""

        for col in headers:
            val = row[col]
            val_str = "N/A"
            if pd.notnull(val):
                val_str = formatters[col].format(val)
            
            # Vérifier si c'est la meilleure valeur de la colonne (Rouge)
            is_best_val = False
            if col in best_values and pd.notnull(val):
                # Utilise isclose pour comparer les floats
                if np.isclose(val, best_values[col]):
                    is_best_val = True
            
            padded_str = val_str.ljust(col_widths[col])
            
            if is_best_val:
                # Rouge écrase le vert de la ligne pour cette cellule spécifique
                row_str += f"{RED}{padded_str}{RESET}{row_color_start}" 
            else:
                row_str += padded_str
        
        print(row_color_start + row_str + row_color_end)
    print("-" * len(header_str))


# --- 2. LOGIQUE D'ANALYSE PRINCIPALE ---
def analyze_epsilon_results(results_dir='open_text_gen/wikitext_epsilon_grid_search'):
    print(f"🔍 Recherche des fichiers dans : {results_dir}")
    all_json_files = glob.glob(os.path.join(results_dir, '*.json'))
    
    div_files = [f for f in all_json_files if ('diversity' in f or 'resuult' in f) and 'coherence' not in f]
    coh_files = [f for f in all_json_files if 'coherence' in f]
    
    if not div_files:
        print("⚠️ Aucun fichier de métriques (diversity/mauve) trouvé.")
        return

    data_store = {}

    # Parser les noms de fichiers
    def get_params(filename):
        basename = os.path.basename(filename)
        k = 5 
        alpha = None
        parts = basename.replace('-', '_').split('_')
        for part in parts:
            if part.startswith('k') and part[1:].isdigit():
                try: k = int(part[1:])
                except: pass
            if part.startswith('alpha'):
                try: alpha = float(part.replace('alpha', ''))
                except: pass
        return k, alpha

    # Chargement métriques
    for f in div_files:
        k, alpha = get_params(f)
        if alpha is not None:
            key = (k, alpha)
            if key not in data_store: data_store[key] = {'k': k, 'alpha': alpha}
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    content = json.load(file)
                    if isinstance(content, list): content = content[0] if len(content) > 0 else {}
                    
                    if 'gen_length_dict' in content and 'gen_len_mean' in content['gen_length_dict']:
                        data_store[key]['gen_length'] = float(content['gen_length_dict']['gen_len_mean'])
                    elif 'gen_length' in content:
                        data_store[key]['gen_length'] = float(content['gen_length'])
                    
                    if 'diversity_dict' in content and 'prediction_div_mean' in content['diversity_dict']:
                        data_store[key]['diversity'] = float(content['diversity_dict']['prediction_div_mean'])
                    elif 'diversity' in content:
                        data_store[key]['diversity'] = float(content['diversity'])
                        
                    if 'mauve_dict' in content and 'mauve_mean' in content['mauve_dict']:
                        data_store[key]['mauve'] = float(content['mauve_dict']['mauve_mean'])
                    elif 'mauve' in content:
                        data_store[key]['mauve'] = float(content['mauve'])

            except Exception as e:
                print(f"Erreur lecture {f}: {e}")

    # Chargement Cohérence
    for f in coh_files:
        k, alpha = get_params(f)
        if alpha is not None:
            key = (k, alpha)
            if key in data_store:
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        content = json.load(file)
                        if isinstance(content, list): content = content[0] if len(content) > 0 else {}
                        
                        if 'coherence_score' in content:
                            data_store[key]['coherence_score'] = float(content['coherence_score'])
                        elif 'mean_score' in content:
                            data_store[key]['coherence_score'] = float(content['mean_score'])
                except Exception as e:
                    print(f"Erreur lecture {f}: {e}")

    # Création DataFrame
    df = pd.DataFrame(list(data_store.values()))
    
    target_cols = ['k', 'alpha', 'gen_length', 'coherence_score', 'diversity', 'mauve']
    final_cols = [c for c in target_cols if c in df.columns]
    
    if not df.empty:
        # Tri
        df = df[final_cols].sort_values(by=['k', 'alpha']).reset_index(drop=True)
        
        print("\n" + "="*80)
        print("📊 RÉSULTATS : EPSILON GREEDY SEARCH")
        print("Légende : Rouge = Meilleur de la colonne | Vert = Meilleur Score MAUVE Global")
        print("="*80)
        
        # Appel de la fonction de coloration
        print_highlighted_table(df)
        print("="*80)
    else:
        print("Aucune donnée extraite.")

if __name__ == "__main__":
    analyze_epsilon_results()