import os
import json
import glob
import pandas as pd
import numpy as np
from tabulate import tabulate

# =============================================================================
# 1. CHARGEMENT ROBUSTE DES DONNÉES
# =============================================================================

def load_json(path):
    if not os.path.exists(path): return {}
    with open(path, 'r', encoding='utf-8') as f:
        try: return json.load(f)
        except: return {}

def get_val(data, *keys):
    curr = data
    if isinstance(curr, list) and curr: curr = curr[0]
    for k in keys:
        if isinstance(curr, dict):
            curr = curr.get(k, {})
        else:
            return None
    if not isinstance(curr, (dict, list)):
        try: return float(curr)
        except: return None
    return None

def collect_data():
    root_dirs = [
        'open_text_gen/wikitext_grid_search',
        'open_text_gen/wikitext_epsilon_grid_search',
        'open_text_gen/wikitext',
    ]
    data_map = {} 
    for d in root_dirs:
        if not os.path.exists(d): continue
        files = glob.glob(os.path.join(d, '*.json'))
        for f in files:
            fname = os.path.basename(f)
            key = None
            m_type = None
            if '_simcse_result.json' in fname:
                key = fname.replace('_simcse_result.json', '')
                m_type = 'simcse'
            elif '_diversity_mauve_gen_length_result.json' in fname:
                key = fname.replace('_diversity_mauve_gen_length_result.json', '')
                m_type = 'div_mauve'
            elif '_perplexity_result.json' in fname:
                key = fname.replace('_perplexity_result.json', '')
                m_type = 'ppl'
            if key and m_type:
                if key not in data_map: data_map[key] = {}
                data_map[key][m_type] = load_json(f)
    
    rows = []
    for key, metrics in data_map.items():
        # NETTOYAGE CRITIQUE : On remplace les underscores par des tirets
        # pour éviter que LaTeX ne plante sur GitHub.
        clean_name = key.replace('wikitext_', '').replace('_', '-')
        
        dm = metrics.get('div_mauve', {})
        row = {
            'Model/Config': clean_name,
            'SimCSE': get_val(metrics.get('simcse', {}), 'coherence_mean'),
            'Diversity': get_val(dm, 'diversity_dict', 'prediction_div_mean'),
            'MAUVE': get_val(dm, 'mauve_dict', 'mauve_mean'),
            'Gen_Length': get_val(dm, 'gen_length_dict', 'gen_len_mean'),
            'Perplexity': get_val(metrics.get('ppl', {}), 'mean_perplexity') or get_val(metrics.get('ppl', {}), 'ppl')
        }
        rows.append(row)
    return pd.DataFrame(rows)

# =============================================================================
# 2. AFFICHAGE TERMINAL (ANSI)
# =============================================================================

def print_terminal(df):
    if df.empty: return
    RED = '\033[91m'; GREEN = '\033[92m'; RESET = '\033[0m'; BOLD = '\033[1m'
    
    criteria = {'SimCSE': 'max', 'Diversity': 'max', 'MAUVE': 'max', 'Gen_Length': 'max', 'Perplexity': 'min'}
    best_vals = {}
    for col, crit in criteria.items():
        if col in df.columns:
            best_vals[col] = df[col].max() if crit == 'max' else df[col].min()

    best_idx = df['MAUVE'].idxmax() if 'MAUVE' in df.columns else -1
    cols = [c for c in df.columns if c != 'Model/Config']
    
    print(f"\n{BOLD}RÉSULTATS (TERMINAL){RESET}")
    print("-" * 120)
    print(f"{'Model/Config':<40} " + " ".join([f"{c:<12}" for c in cols]))
    print("-" * 120)

    for idx, row in df.iterrows():
        line = ""
        is_winner = (idx == best_idx)
        color_line = GREEN if is_winner else ""
        
        line += f"{color_line}{str(row['Model/Config']):<40}{RESET} "
        
        for col in cols:
            val = row[col]
            txt = f"{val:.3f}" if pd.notna(val) else "N/A"
            is_best = False
            if pd.notna(val) and col in best_vals and np.isclose(val, best_vals[col]): is_best = True
            
            if is_best: line += f"{RED}{txt:<12}{RESET}"
            else: line += f"{color_line}{txt:<12}{RESET}"
        print(line)
    print("-" * 120)

# =============================================================================
# 3. GÉNÉRATION MARKDOWN (FULL COLOR LATEX - SAFE)
# =============================================================================

def generate_markdown(df):
    if df.empty: return

    criteria = {'SimCSE': 'max', 'Diversity': 'max', 'MAUVE': 'max', 'Gen_Length': 'max', 'Perplexity': 'min'}
    best_vals = {}
    for col, crit in criteria.items():
        if col in df.columns:
            best_vals[col] = df[col].max() if crit == 'max' else df[col].min()

    best_idx = df['MAUVE'].idxmax() if 'MAUVE' in df.columns else -1
    headers = list(df.columns)
    
    md = []
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join([":---"] + [":---:"]*(len(headers)-1)) + " |")

    for idx, row in df.iterrows():
        cells = []
        is_winner = (idx == best_idx)
        
        for col in headers:
            val = row[col]
            # Conversion en string propre
            if isinstance(val, float):
                txt = f"{val:.3f}"
            else:
                txt = str(val).replace('_', '-') # Double sécurité
            
            if pd.isna(val): txt = "N/A"

            # 1. NOM DU MODÈLE
            if col == 'Model/Config':
                if is_winner:
                    # Maintenant que c'est "dash-separated", la couleur passe !
                    cells.append(f"$\\color{{green}}{{\\textsf{{{txt}}}}}$")
                else:
                    cells.append(f"$\\textsf{{{txt}}}$")
                continue

            # 2. CHIFFRES
            is_best_val = False
            if pd.notna(val) and col in best_vals and np.isclose(val, best_vals[col]):
                is_best_val = True
            
            if is_best_val:
                cells.append(f"$\\color{{red}}{{\\textsf{{{txt}}}}}$")
            elif is_winner:
                # Toute la ligne en vert si c'est le gagnant (sauf si record rouge)
                cells.append(f"$\\color{{green}}{{\\textsf{{{txt}}}}}$")
            else:
                cells.append(f"$\\textsf{{{txt}}}$")
        
        md.append("| " + " | ".join(cells) + " |")

    print("\n### CODE MARKDOWN FINAL (AVEC COULEURS)\n")
    print("\n".join(md))
    print("\n> **Légende :** $\\color{red}{\\textsf{Rouge}}$ = Meilleur score. $\\color{green}{\\textsf{Vert}}$ = Modèle avec le meilleur score MAUVE global.")

# =============================================================================
# MAIN
# =============================================================================

def main():
    df = collect_data()
    if not df.empty:
        df = df.sort_values('Model/Config').reset_index(drop=True)
    
    print_terminal(df)
    generate_markdown(df)

if __name__ == "__main__":
    main()