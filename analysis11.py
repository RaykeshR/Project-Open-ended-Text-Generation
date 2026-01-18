import json
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
BASE_DIR = "open_text_gen"
DATASETS = ["wikitext", "bookcorpus", "cc_news"]
METRICS_COLS = ['MAUVE', 'Diversity', 'Coh_Likelihood', 'Coh_SimCSE', 'Perplexity', 'Gen_Length']

# --- PARAMÈTRES OPTIMAUX PAR DATASET ---
# Modifiez ici les valeurs si nécessaire (ex: k=50 pour cc_news)
DATASET_PARAMS = {
    "wikitext":   {"k": 10, "alpha": 0.6, "p": 0.95},
    "bookcorpus": {"k": 10, "alpha": 0.6, "p": 0.95},
    "cc_news":    {"k": 50, "alpha": 0.2, "p": 0.95} 
}

MODEL_NAME_FILTER = "gpt2-xl" # Pour éviter de mélanger avec gpt2-large/medium

# --- Codes couleurs ANSI ---
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[96m"

def get_json_val(data, keys, default=0.0):
    val = data
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, default)
        else:
            return default
    try:
        return float(val)
    except:
        return default

def detect_method_name(filename, dataset):
    """
    Déduit le nom de la méthode.
    Filtre STRICTEMENT selon le modèle et les paramètres du dataset.
    """
    base = os.path.basename(filename)
    
    # 1. Filtrage Modèle (Crucial pour Wikitext qui a plein de modèles)
    if MODEL_NAME_FILTER not in base:
        return "Skip"

    params = DATASET_PARAMS[dataset]
    best_k = params["k"]
    best_alpha = params["alpha"]
    best_p = params["p"]

    # 2. Identification Méthode
    
    # Greedy
    if "greedy" in base:
        return "Greedy Search"
    
    # Nucleus
    elif f"p-{best_p}" in base or (f"nucleus" in base and str(best_p) in base):
        return f"Nucleus Sampling (p={best_p})"
    
    # Typical
    elif "typical" in base and str(best_p) in base:
        return f"Typical Sampling (p={best_p})"
    
    # Epsilon Sampling
    elif "epsilon" in base:
        if f"k{best_k}" in base and f"alpha{best_alpha}" in base:
            return f"Epsilon Sampling (k={best_k}, α={best_alpha})"
        return "Skip" # On ignore les autres configs

    # Contrastive Search
    elif "contrastive" in base:
        # Souvent contrastive a le format 'contrastive-alpha-0.6'
        if f"alpha-{best_alpha}" in base or f"alpha{best_alpha}" in base:
            return f"Contrastive Search (k={best_k}, α={best_alpha})"
        return "Skip"
        
    # Format alternatif Contrastive (généré par generate.py) : k10_a0.6_e0.0
    elif f"k{best_k}_a{best_alpha}_e0.0" in base:
        return f"Contrastive Search (k={best_k}, α={best_alpha})"

    return "Skip"

def load_all_methods():
    all_data = []

    for dataset in DATASETS:
        # Dossiers à scanner
        dirs_to_scan = [
            os.path.join(BASE_DIR, f"{dataset}_grid_search"),
            # os.path.join(BASE_DIR, "ollama_results") 
        ]
        
        # Ajout dossier Epsilon
        if dataset == "wikitext":
            dirs_to_scan.append(os.path.join(BASE_DIR, f"{dataset}_epsilon_grid_search_256"))
        else:
            dirs_to_scan.append(os.path.join(BASE_DIR, f"{dataset}_epsilon_grid_search"))

        for folder in dirs_to_scan:
            if not os.path.exists(folder):
                continue

            # On cherche les fichiers de résultats
            result_files = glob.glob(os.path.join(folder, "*diversity_mauve_gen_length_result.json"))

            for res_file in result_files:
                basename = os.path.basename(res_file)
                
                # Vérif dataset
                if dataset not in basename:
                    continue

                # Détection méthode avec paramètres spécifiques au dataset
                method_name = detect_method_name(basename, dataset)
                if method_name == "Skip": 
                    continue

                file_prefix = res_file.replace("_diversity_mauve_gen_length_result.json", "")
                
                row = {
                    'Dataset': dataset,
                    'Method': method_name,
                }

                # --- Chargement Métriques ---
                
                # 1. Diversity / MAUVE
                try:
                    with open(res_file, 'r', encoding='utf-8') as f:
                        d = json.load(f)
                        if isinstance(d, list): d = d[0]
                        row['MAUVE'] = get_json_val(d, ['mauve_dict', 'mauve_mean'])
                        row['Diversity'] = get_json_val(d, ['diversity_dict', 'prediction_div_mean'])
                        row['Gen_Length'] = get_json_val(d, ['gen_length_dict', 'gen_len_mean'])
                except: pass

                # 2. Perplexity
                ppl_file = f"{file_prefix}_perplexity_result.json"
                if os.path.exists(ppl_file):
                    try:
                        with open(ppl_file, 'r', encoding='utf-8') as f:
                            d = json.load(f)
                            val = d.get('mean_perplexity')
                            if val is None: val = d.get('ppl')
                            if val is not None:
                                row['Perplexity'] = float(val)
                            else:
                                row['Perplexity'] = float('nan')
                    except: row['Perplexity'] = float('nan')
                else: row['Perplexity'] = float('nan')

                # 3. Coherence (Likelihood)
                coh_files = glob.glob(f"{file_prefix}*_coherence_result.json")
                coh_files = [x for x in coh_files if 'simcse' not in x]
                if coh_files:
                    # On préfère le plus gros modèle juge
                    coh_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
                    try:
                        with open(coh_files[0], 'r', encoding='utf-8') as f:
                            d = json.load(f)
                            if isinstance(d, list): d = d[0]
                            row['Coh_Likelihood'] = float(d.get('coherence_mean', 0.0))
                    except: row['Coh_Likelihood'] = 0.0
                else: row['Coh_Likelihood'] = 0.0

                # 4. SimCSE
                simcse_file = f"{file_prefix}_simcse_result.json"
                if os.path.exists(simcse_file):
                    try:
                        with open(simcse_file, 'r', encoding='utf-8') as f:
                            d = json.load(f)
                            row['Coh_SimCSE'] = float(d.get('coherence_mean', 0.0))
                    except: row['Coh_SimCSE'] = 0.0
                else: row['Coh_SimCSE'] = 0.0

                all_data.append(row)

    return pd.DataFrame(all_data)

def format_val(val, metric, is_best, color_mode="none"):
    if pd.isna(val): return "N/A"
    
    fmt = "{:.3f}"
    if metric in ['Perplexity', 'MAUVE', 'Gen_Length']: fmt = "{:.2f}"
    
    txt = fmt.format(val)
    
    if is_best:
        if color_mode == "terminal":
            return f"{GREEN}{BOLD}{txt}{RESET}"
        elif color_mode == "markdown_color":
            return f"$\\color{{green}}{{\\mathbf{{{txt}}}}}$"
        elif color_mode == "markdown":
            return f"**{txt}**"
    return txt

def generate_comparison_table(df, dataset_name, output_format="terminal"):
    sub_df = df[df['Dataset'] == dataset_name].copy()
    if sub_df.empty: return "Pas de données."

    # Récupération params pour affichage
    params = DATASET_PARAMS[dataset_name]
    
    # Ordre
    method_order = [
        "Greedy Search", 
        f"Nucleus Sampling (p={params['p']})", 
        f"Typical Sampling (p={params['p']})", 
        f"Contrastive Search (k={params['k']}, α={params['alpha']})",
        f"Epsilon Sampling (k={params['k']}, α={params['alpha']})"
    ]
    
    # Tri
    sub_df['Method'] = pd.Categorical(sub_df['Method'], categories=method_order, ordered=True)
    sub_df = sub_df.sort_values('Method')

    # Meilleurs scores
    best_idxs = {}
    for col in METRICS_COLS:
        if col in sub_df.columns and not sub_df[col].isna().all():
            if col == 'Perplexity':
                best_idxs[col] = sub_df[col].idxmin()
            elif col == 'Gen_Length':
                best_idxs[col] = None 
            else:
                best_idxs[col] = sub_df[col].idxmax()

    # Affichage
    lines = []
    col_width = 16
    method_width = 45
    
    if output_format == "terminal":
        header = f"{CYAN}{'Method':<{method_width}} " + " ".join([f"{c:<{col_width}}" for c in METRICS_COLS]) + f"{RESET}"
        lines.append(header)
        lines.append("-" * len(re.sub(r'\033\[[0-9;]*m', '', header)))
    else:
        # Markdown Header
        header = "| Method | " + " | ".join(METRICS_COLS) + " |"
        lines.append(header)
        lines.append("|:---| " + " | ".join([":---:" for _ in METRICS_COLS]) + " |")

    for idx, row in sub_df.iterrows():
        m_name = str(row['Method'])
        row_str = ""
        
        if output_format == "terminal":
            row_str += f"{m_name:<{method_width}} "
        else:
            row_str += f"| {m_name} | "
            
        for col in METRICS_COLS:
            val = row.get(col, float('nan'))
            is_best = (idx == best_idxs.get(col))
            fmt_val = format_val(val, col, is_best, color_mode=output_format)
            
            if output_format == "terminal":
                plain_len = len(re.sub(r'\033\[[0-9;]*m', '', fmt_val))
                padding = " " * (col_width - plain_len)
                row_str += f"{fmt_val}{padding} "
            else:
                row_str += f"{fmt_val} | "
        lines.append(row_str)
        
    return "\n".join(lines)

def plot_comparative_graphs(df):
    if df.empty: return
    sns.set_theme(style="whitegrid")
    output_dir = "plots_final_comparison"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    metrics_to_plot = ['MAUVE', 'Diversity', 'Coh_Likelihood', 'Perplexity']
    
    for metric in metrics_to_plot:
        if metric not in df.columns: continue
        valid_df = df.dropna(subset=[metric])
        if valid_df.empty: continue

        plt.figure(figsize=(12, 6))
        g = sns.catplot(
            data=valid_df, kind="bar",
            x="Dataset", y=metric, hue="Method",
            palette="viridis", height=6, aspect=1.5
        )
        g.fig.suptitle(f"Comparaison : {metric}", y=1.02, fontweight='bold')
        g.set_axis_labels("Dataset", metric)
        
        filename = os.path.join(output_dir, f"compare_{metric}.png")
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        print(f"Graphique sauvegardé : {filename}")
        plt.close()

def main():
    print(f"{BOLD}Analyse Comparative Finale{RESET}")
    print(f"Filtre Modèle : {MODEL_NAME_FILTER}")
    print(f"Params CC_NEWS : k={DATASET_PARAMS['cc_news']['k']}, alpha={DATASET_PARAMS['cc_news']['alpha']}")
    
    df = load_all_methods()
    
    if df.empty:
        print(f"{RED}Aucune donnée trouvée.{RESET}")
        return

    # 1. Terminal Output
    print("\n" + "="*130)
    print(f"{BOLD} TABLEAUX COMPARATIFS (TERMINAL){RESET}")
    print("="*130)
    for ds in DATASETS:
        print(f"\n{BOLD}>>> DATASET: {ds.upper()}{RESET}")
        print(generate_comparison_table(df, ds, "terminal"))

    # 2. Markdown Simple
    print("\n" + "="*130)
    print(f"{BOLD} MARKDOWN GITHUB{RESET}")
    print("="*130)
    for ds in DATASETS:
        print(f"\n### Résultats : {ds.upper()}")
        print(generate_comparison_table(df, ds, "markdown"))

    # 3. Markdown Latex
    print("\n" + "="*130)
    print(f"{BOLD} MARKDOWN COLORÉ (LATEX){RESET}")
    print("="*130)
    for ds in DATASETS:
        print(f"\n### {ds.upper()}")
        print(generate_comparison_table(df, ds, "markdown_color"))

    # 4. Graphs
    print("\n" + "="*130)
    print(f"{BOLD} GÉNÉRATION DES GRAPHIQUES{RESET}")
    plot_comparative_graphs(df)

if __name__ == "__main__":
    main()