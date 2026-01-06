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
        dm = metrics.get('div_mauve', {})
        row = {
            'Model/Config': key.replace('wikitext_', ''),
            'SimCSE': get_val(metrics.get('simcse', {}), 'coherence_mean'),
            'Diversity': get_val(dm, 'diversity_dict', 'prediction_div_mean'),
            'MAUVE': get_val(dm, 'mauve_dict', 'mauve_mean'),
            'Gen_Length': get_val(dm, 'gen_length_dict', 'gen_len_mean'),
            'Perplexity': get_val(metrics.get('ppl', {}), 'mean_perplexity') or get_val(metrics.get('ppl', {}), 'ppl')
        }
        rows.append(row)
    return pd.DataFrame(rows)

# =============================================================================
# 2. AFFICHAGE TERMINAL (ANSI COLORS)
# =============================================================================

def print_terminal_colored(df):
    if df.empty: return
    RED = '\033[91m'; GREEN = '\033[92m'; RESET = '\033[0m'; BOLD = '\033[1m'
    criteria = {'SimCSE': 'max', 'Diversity': 'max', 'MAUVE': 'max', 'Gen_Length': 'max', 'Perplexity': 'min'}
    
    best_vals = {}
    for col, crit in criteria.items():
        if col in df.columns:
            best_vals[col] = df[col].max() if crit == 'max' else df[col].min()

    best_idx = df['MAUVE'].idxmax() if 'MAUVE' in df.columns and not df['MAUVE'].isna().all() else -1
    cols = [c for c in df.columns if c != 'Model/Config']
    headers = ['Model/Config'] + cols
    
    widths = {h: len(h) + 2 for h in headers}
    max_name_len = df['Model/Config'].astype(str).map(len).max()
    widths['Model/Config'] = max(widths['Model/Config'], max_name_len + 2)

    header_str = "".join([h.ljust(widths[h]) for h in headers])
    print("\n" + "="*len(header_str))
    print(BOLD + " RÉSULTATS COMPARATIFS (TERMINAL)" + RESET)
    print("-" * len(header_str))
    print(BOLD + header_str + RESET)
    print("-" * len(header_str))

    for idx, row in df.iterrows():
        line_str = ""
        is_winner_row = (idx == best_idx)
        line_color = GREEN if is_winner_row else ""
        end_color = RESET if is_winner_row else ""
        
        line_str += line_color + str(row['Model/Config']).ljust(widths['Model/Config']) + end_color
        for col in cols:
            val = row[col]
            cell_str = f"{val:.3f}" if pd.notna(val) else "N/A"
            is_best_col = False
            if pd.notna(val) and col in best_vals and np.isclose(val, best_vals[col]): is_best_col = True
            
            padded = cell_str.ljust(widths[col])
            if is_best_col: line_str += f"{RED}{padded}{RESET}{line_color}" 
            else: line_str += f"{line_color}{padded}{end_color}"
        print(line_str)
    print("=" * len(header_str) + "\n")

# =============================================================================
# 3. GÉNÉRATION MARKDOWN (LATEX COLORS + FIX UNDERSCORES)
# =============================================================================

def generate_markdown_latex_colors(df):
    if df.empty: return

    criteria = {'SimCSE': 'max', 'Diversity': 'max', 'MAUVE': 'max', 'Gen_Length': 'max', 'Perplexity': 'min'}
    best_vals = {}
    for col, crit in criteria.items():
        if col in df.columns:
            best_vals[col] = df[col].max() if crit == 'max' else df[col].min()

    best_idx = df['MAUVE'].idxmax() if 'MAUVE' in df.columns and not df['MAUVE'].isna().all() else -1
    headers = list(df.columns)
    md_lines = []
    
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join([":---"] + [":---:"]*(len(headers)-1)) + " |")

    for idx, row in df.iterrows():
        row_cells = []
        is_winner_row = (idx == best_idx)
        
        for col in headers:
            val = row[col]
            val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
            if pd.isna(val): val_str = "N/A"

            if col == 'Model/Config':
                safe_name = str(val_str).replace('_', '\\_') 
                if is_winner_row:
                    row_cells.append(f"$\\color{{green}}{{\\textsf{{{safe_name}}}}}$")
                else:
                    row_cells.append(f"`{val_str}`") # On garde le code block simple pour les non-gagnants
                continue

            is_best = False
            if pd.notna(val) and col in best_vals and np.isclose(val, best_vals[col]):
                is_best = True
            
            if is_best:
                row_cells.append(f"$\\color{{red}}{{\\textsf{{{val_str}}}}}$")
            else:
                row_cells.append(f"$\\textsf{{{val_str}}}$")
        
        md_lines.append("| " + " | ".join(row_cells) + " |")

    print("\n### CODE MARKDOWN (AVEC COULEURS) À COPIER DANS LE README\n")
    print("\n".join(md_lines))
    print("\n> **Légende :** $\\color{red}{\\textsf{Rouge}}$ = Meilleur score de la colonne. $\\color{green}{\\textsf{Vert}}$ = Modèle avec le meilleur score MAUVE global.")

# =============================================================================
# MAIN
# =============================================================================

def main():
    df = collect_data()
    if not df.empty:
        df = df.sort_values('Model/Config').reset_index(drop=True)
    
    print_terminal_colored(df)
    generate_markdown_latex_colors(df)

if __name__ == "__main__":
    main()