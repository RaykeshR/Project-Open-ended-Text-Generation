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

# Meilleurs hyperparamètres pour Contrastive Search
BEST_K = 10
BEST_ALPHA = 0.6

# --- Codes couleurs ANSI ---
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[96m"

def get_json_val(data, keys, default=0.0):
    """Récupère une valeur imbriquée dans un dict JSON."""
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

def detect_method_name(filename):
    """Déduit le nom propre de la méthode depuis le nom de fichier."""
    base = os.path.basename(filename)
    if "greedy" in base:
        return "Greedy Search"
    elif "beam" in base:
        return "Beam Search"
    elif "contrastive" in base or "epsilon" in base:
        # On vérifie si c'est la config "Best"
        if f"k{BEST_K}" in base and f"alpha{BEST_ALPHA}" in base:
            return f"Contrastive (k={BEST_K}, α={BEST_ALPHA})"
        return "Contrastive (Other)"
    elif "typical" in base:
        return "Typical Sampling (p=0.95)"
    elif "p-0.95" in base or "nucleus" in base:
        return "Nucleus Sampling (p=0.95)"
    elif "ollama" in base:
        if "llama3.2" in base: return "Llama 3.2 (Ollama)"
        if "mistral" in base: return "Mistral (Ollama)"
        if "gpt2-xl" in base: return "GPT2-XL (Ollama)"
        return "Ollama Model"
    return "Unknown"

def load_all_methods():
    all_data = []

    for dataset in DATASETS:
        # 1. Définir les dossiers où chercher
        dirs_to_scan = [
            os.path.join(BASE_DIR, dataset),  # Baselines (Greedy, Nucleus...)
            os.path.join(BASE_DIR, f"{dataset}_epsilon_grid_search"), # Contrastive
            os.path.join(BASE_DIR, "ollama_results") # Modèles externes
        ]

        for folder in dirs_to_scan:
            if not os.path.exists(folder):
                continue

            # On cherche tous les fichiers résultats principaux (Diversity/MAUVE)
            result_files = glob.glob(os.path.join(folder, "*diversity_mauve_gen_length_result.json"))

            for res_file in result_files:
                basename = os.path.basename(res_file)
                
                # Filtrage : On ne veut QUE les méthodes finales
                # Si c'est Contrastive, on ne prend que k=10 et alpha=0.6
                if "epsilon" in basename or "contrastive" in basename:
                    if f"k{BEST_K}" not in basename or f"alpha{BEST_ALPHA}" not in basename:
                        continue # On saute les hyperparamètres non optimaux
                
                # Si c'est Ollama, on vérifie que ça correspond au dataset
                if "ollama" in basename and dataset not in basename:
                    continue

                method_name = detect_method_name(basename)
                if method_name == "Unknown": continue

                # Préfixe pour trouver les fichiers frères (Perplexity, Coherence...)
                file_prefix = res_file.replace("_diversity_mauve_gen_length_result.json", "")
                
                row = {
                    'Dataset': dataset,
                    'Method': method_name,
                }

                # --- Chargement des Métriques ---
                
                # 1. MAUVE / Diversity
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
                            row['Perplexity'] = float(d.get('mean_perplexity', d.get('ppl', 0.0)))
                    except: row['Perplexity'] = float('nan')
                else: row['Perplexity'] = float('nan')

                # 3. Coherence (Likelihood)
                # Astuce: le nom du fichier coherence peut varier (opt-2.7b vs opt-125m)
                # On cherche n'importe quel fichier coherence commençant par le préfixe
                coh_files = glob.glob(f"{file_prefix}*_coherence_result.json")
                # Exclure SimCSE des fichiers coherence
                coh_files = [x for x in coh_files if 'simcse' not in x]
                
                if coh_files:
                    try:
                        with open(coh_files[0], 'r', encoding='utf-8') as f:
                            d = json.load(f)
                            if isinstance(d, list): d = d[0]
                            row['Coh_Likelihood'] = float(d.get('coherence_mean', 0.0))
                    except: row['Coh_Likelihood'] = 0.0
                else: row['Coh_Likelihood'] = 0.0

                # 4. Coherence (SimCSE)
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

    # Ordre d'affichage des méthodes
    method_order = ["Greedy Search", "Nucleus Sampling (p=0.95)", "Typical Sampling (p=0.95)", f"Contrastive (k={BEST_K}, α={BEST_ALPHA})"]
    # Ajouter les méthodes Ollama trouvées à la fin
    others = [m for m in sub_df['Method'].unique() if m not in method_order]
    final_order = method_order + others
    
    # Rendre catégorique pour le tri
    sub_df['Method'] = pd.Categorical(sub_df['Method'], categories=final_order, ordered=True)
    sub_df = sub_df.sort_values('Method')

    # Identifier les meilleurs scores
    best_idxs = {}
    for col in METRICS_COLS:
        if col in sub_df.columns and not sub_df[col].isna().all():
            if col == 'Perplexity':
                best_idxs[col] = sub_df[col].idxmin()
            elif col == 'Gen_Length':
                best_idxs[col] = None 
            else:
                best_idxs[col] = sub_df[col].idxmax()

    # Header
    lines = []
    col_width = 14
    if output_format == "terminal":
        header = f"{CYAN}{'Method':<30} " + " ".join([f"{c:<{col_width}}" for c in METRICS_COLS]) + f"{RESET}"
        lines.append(header)
        lines.append("-" * len(re.sub(r'\033\[[0-9;]*m', '', header)))
    else:
        header = "| Method | " + " | ".join(METRICS_COLS) + " |"
        lines.append(header)
        lines.append("|:---| " + " | ".join([":---:" for _ in METRICS_COLS]) + " |")

    # Rows
    for idx, row in sub_df.iterrows():
        m_name = str(row['Method'])
        
        row_str = ""
        if output_format == "terminal":
            row_str += f"{m_name:<30} "
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
    """Génère des barplots comparant les méthodes."""
    if df.empty: return
    sns.set_theme(style="whitegrid")
    
    output_dir = "plots_final_comparison"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    metrics_to_plot = ['MAUVE', 'Diversity', 'Coh_Likelihood', 'Perplexity']
    
    for metric in metrics_to_plot:
        if metric not in df.columns: continue
        
        # Filtrer les NaN
        valid_df = df.dropna(subset=[metric])
        if valid_df.empty: continue

        g = sns.catplot(
            data=valid_df, 
            kind="bar",
            x="Dataset", 
            y=metric, 
            hue="Method",
            palette="viridis",
            height=5, 
            aspect=1.5
        )
        
        g.fig.suptitle(f"Comparaison : {metric}", y=1.02)
        g.set_axis_labels("Dataset", metric)
        
        filename = os.path.join(output_dir, f"compare_{metric}.png")
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        print(f"Graphique sauvegardé : {filename}")
        plt.close()

def main():
    print(f"{BOLD}Analyse Comparative Finale{RESET}")
    print(f"Recherche des résultats pour : Greedy, Nucleus, Typical, et Contrastive (k={BEST_K}, α={BEST_ALPHA})...")
    
    df = load_all_methods()
    
    if df.empty:
        print(f"{RED}Aucune donnée trouvée.{RESET}")
        return

    # 1. Terminal
    print("\n" + "="*100)
    print(f"{BOLD} TABLEAUX COMPARATIFS (TERMINAL){RESET}")
    print("="*100)
    for ds in DATASETS:
        print(f"\n{BOLD}>>> DATASET: {ds.upper()}{RESET}")
        print(generate_comparison_table(df, ds, "terminal"))

    # 2. Markdown Simple
    print("\n" + "="*100)
    print(f"{BOLD} MARKDOWN GITHUB{RESET}")
    print("="*100)
    for ds in DATASETS:
        print(f"\n### Résultats : {ds.upper()}")
        print(generate_comparison_table(df, ds, "markdown"))

    # 3. Markdown Latex
    print("\n" + "="*100)
    print(f"{BOLD} MARKDOWN COLORÉ (LATEX){RESET}")
    print("="*100)
    for ds in DATASETS:
        print(f"\n### {ds.upper()}")
        print(generate_comparison_table(df, ds, "markdown_color"))

    # 4. Graphs
    print("\n" + "="*100)
    print(f"{BOLD} GÉNÉRATION DES GRAPHIQUES COMPARATIFS{RESET}")
    plot_comparative_graphs(df)

if __name__ == "__main__":
    main()