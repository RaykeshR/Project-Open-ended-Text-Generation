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

# --- PARAMÈTRES OPTIMAUX PAR DATASET ---
# Modifiez ces valeurs pour qu'elles correspondent EXACTEMENT à ce que vous avez généré.
DATASET_PARAMS = {
    "wikitext":   {"k": 10, "alpha": 0.6, "p": 0.95},
    "bookcorpus": {"k": 10, "alpha": 0.6, "p": 0.95},
    "cc_news":    {"k": 50, "alpha": 0.2, "p": 0.95} 
}

MODEL_NAME_FILTER = "gpt2-xl" 

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
    Déduit le nom de la méthode et filtre selon les paramètres DATASET_PARAMS.
    Retourne (Nom_Methode, Raison_Si_Skip)
    """
    base = os.path.basename(filename)
    
    if MODEL_NAME_FILTER not in base:
        return "Skip", "Mauvais modèle"

    params = DATASET_PARAMS[dataset]
    best_k = params["k"]
    best_alpha = params["alpha"]
    best_p = params["p"]

    # Greedy
    if "greedy" in base:
        return "Greedy Search", None
    
    # Nucleus
    elif f"p-{best_p}" in base or (f"nucleus" in base and str(best_p) in base):
        return f"Nucleus Sampling (p={best_p})", None
    
    # Typical
    elif "typical" in base and str(best_p) in base:
        return f"Typical Sampling (p={best_p})", None
    
    # Epsilon Sampling
    elif "epsilon" in base:
        if f"k{best_k}" in base and f"alpha{best_alpha}" in base:
            return f"Epsilon Sampling (k={best_k}, α={best_alpha})", None
        return "Skip", f"Params Epsilon mismatch (found {base}, want k{best_k} a{best_alpha})"

    # Contrastive Search
    # Pattern 1: contrastive-alpha-0.6
    # Pattern 2: k10_a0.6_e0.0 (généré par generate.py standard)
    elif "contrastive" in base or (f"k{best_k}" in base and f"a{best_alpha}" in base and "e0.0" in base):
        # On vérifie k et alpha
        if (f"alpha-{best_alpha}" in base) or (f"a{best_alpha}" in base and f"k{best_k}" in base):
            return f"Contrastive Search (k={best_k}, α={best_alpha})", None
        return "Skip", f"Params Contrastive mismatch (found {base}, want k{best_k} a{best_alpha})"

    return "Skip", "Méthode inconnue ou params incorrects"

def load_all_methods():
    all_data = []
    print(f"{YELLOW}Chargement des fichiers...{RESET}")

    for dataset in DATASETS:
        # Dossiers à scanner
        dirs_to_scan = [
            os.path.join(BASE_DIR, f"{dataset}_grid_search"),
        ]
        
        # Ajout dossier Epsilon
        if dataset == "wikitext":
            dirs_to_scan.append(os.path.join(BASE_DIR, f"{dataset}_epsilon_grid_search_256"))
        else:
            dirs_to_scan.append(os.path.join(BASE_DIR, f"{dataset}_epsilon_grid_search"))

        for folder in dirs_to_scan:
            if not os.path.exists(folder):
                print(f"{RED}[WARN] Dossier introuvable : {folder}{RESET}")
                continue

            result_files = glob.glob(os.path.join(folder, "*diversity_mauve_gen_length_result.json"))

            for res_file in result_files:
                basename = os.path.basename(res_file)
                
                # Vérif dataset dans le nom (sécurité)
                if dataset not in basename:
                    continue

                method_name, skip_reason = detect_method_name(basename, dataset)
                if method_name == "Skip": 
                    # Décommenter pour voir pourquoi un fichier est ignoré
                    # print(f"  [SKIP] {basename} -> {skip_reason}")
                    continue

                print(f"  [OK] Found: {method_name} in {basename}")

                file_prefix = res_file.replace("_diversity_mauve_gen_length_result.json", "")
                
                row = {
                    'Dataset': dataset,
                    'Method': method_name,
                }

                # 1. Diversity / MAUVE
                try:
                    with open(res_file, 'r', encoding='utf-8') as f:
                        d = json.load(f)
                        if isinstance(d, list): d = d[0]
                        row['MAUVE'] = get_json_val(d, ['mauve_dict', 'mauve_mean'])
                        row['Diversity'] = get_json_val(d, ['diversity_dict', 'prediction_div_mean'])
                        row['Gen_Length'] = get_json_val(d, ['gen_length_dict', 'gen_len_mean'])
                except Exception as e: 
                    print(f"{RED}Error reading DIV/MAUVE for {basename}: {e}{RESET}")

                # 2. Perplexity
                ppl_file = f"{file_prefix}_perplexity_result.json"
                if os.path.exists(ppl_file):
                    try:
                        with open(ppl_file, 'r', encoding='utf-8') as f:
                            d = json.load(f)
                            val = d.get('mean_perplexity') or d.get('ppl')
                            row['Perplexity'] = float(val) if val else float('nan')
                    except: row['Perplexity'] = float('nan')
                else: 
                    # print(f"  [WARN] Missing PPL file for {method_name}")
                    row['Perplexity'] = float('nan')

                # 3. Coherence (Likelihood) - DYNAMIQUE PAR MODELE JUGE
                # Pattern: fichier se termine par _coherence_result.json
                # Mais attention, simcse se termine aussi par là dans certains noms, d'où le filtrage
                coh_files = glob.glob(f"{file_prefix}*_coherence_result.json")
                
                for cf in coh_files:
                    cf_base = os.path.basename(cf)
                    if "simcse" in cf_base: continue 
                    
                    # Extraction du nom du modèle juge
                    # Ex: wikinews_greedy..._opt-2.7b_coherence_result.json
                    # On cherche la partie entre le préfixe et _coherence_result
                    # C'est un peu tricky, on va essayer de parser "opt-..."
                    try:
                        with open(cf, 'r', encoding='utf-8') as f:
                            d = json.load(f)
                            if isinstance(d, list): d = d[0]
                            val = float(d.get('coherence_mean', 0.0))
                            
                            # Trouver le nom du juge
                            if "opt-2.7b" in cf_base: judge = "OPT-2.7B"
                            elif "opt-1.3b" in cf_base: judge = "OPT-1.3B"
                            elif "opt-125m" in cf_base: judge = "OPT-125m"
                            elif "gpt2-xl" in cf_base and "gpt2-xl_" not in cf_base: judge = "GPT2-XL" # Attention auto-reference
                            else: judge = "Unknown_Judge"
                            
                            row[f'Coh_Like_{judge}'] = val
                    except: pass

                # 4. SimCSE
                simcse_file = f"{file_prefix}_simcse_result.json"
                if os.path.exists(simcse_file):
                    try:
                        with open(simcse_file, 'r', encoding='utf-8') as f:
                            d = json.load(f)
                            val = float(d.get('coherence_mean', 0.0))
                            if val == 0.0:
                                print(f"{RED}  [WARN] SimCSE is 0.0 for {method_name}{RESET}")
                            row['Coh_Sem_SimCSE'] = val
                    except: row['Coh_Sem_SimCSE'] = 0.0
                else: 
                    # print(f"  [WARN] Missing SimCSE for {method_name}")
                    row['Coh_Sem_SimCSE'] = float('nan')

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

    # Colonnes Dynamiques
    # On prend les colonnes standard + toutes les colonnes Coh_ trouvées
    std_cols = ['MAUVE', 'Diversity', 'Perplexity', 'Gen_Length']
    coh_cols = sorted([c for c in sub_df.columns if c.startswith('Coh_')])
    
    # On met SimCSE à la fin des Coh
    sim_cols = [c for c in coh_cols if "SimCSE" in c]
    like_cols = [c for c in coh_cols if "SimCSE" not in c]
    
    current_metrics = std_cols + like_cols + sim_cols
    
    # Vérifier que les colonnes existent bien dans ce sous-df (sinon drop)
    current_metrics = [c for c in current_metrics if c in sub_df.columns]

    params = DATASET_PARAMS[dataset_name]
    
    # Ordre des méthodes
    method_order = [
        "Greedy Search", 
        f"Nucleus Sampling (p={params['p']})", 
        f"Typical Sampling (p={params['p']})", 
        f"Contrastive Search (k={params['k']}, α={params['alpha']})",
        f"Epsilon Sampling (k={params['k']}, α={params['alpha']})"
    ]
    
    # Catégorisation
    sub_df['Method'] = pd.Categorical(sub_df['Method'], categories=method_order, ordered=True)
    sub_df = sub_df.sort_values('Method')

    # Meilleurs scores
    best_idxs = {}
    for col in current_metrics:
        if not sub_df[col].isna().all():
            if col == 'Perplexity':
                best_idxs[col] = sub_df[col].idxmin()
            elif col == 'Gen_Length':
                best_idxs[col] = None 
            else:
                best_idxs[col] = sub_df[col].idxmax()

    # Construction du tableau
    lines = []
    
    # Headers
    if output_format == "terminal":
        col_width = 14
        method_width = 45
        header = f"{CYAN}{'Method':<{method_width}} " + " ".join([f"{c:<{col_width}}" for c in current_metrics]) + f"{RESET}"
        lines.append(header)
        lines.append("-" * len(re.sub(r'\033\[[0-9;]*m', '', header)))
    else:
        # Markdown
        header = "| Method | " + " | ".join(current_metrics) + " |"
        lines.append(header)
        lines.append("|:---| " + " | ".join([":---:" for _ in current_metrics]) + " |")

    # Rows
    for idx, row in sub_df.iterrows():
        m_name = str(row['Method'])
        row_str = ""
        
        if output_format == "terminal":
            row_str += f"{m_name:<{method_width}} "
        else:
            row_str += f"| {m_name} | "
            
        for col in current_metrics:
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

    # On plotte les métriques standard + une moyenne de cohérence si dispo
    metrics_to_plot = ['MAUVE', 'Diversity', 'Perplexity']
    
    # Ajouter une colonne cohérence principale (ex: OPT-2.7B si dispo)
    cols = df.columns
    opt_cols = [c for c in cols if "OPT-2.7B" in c]
    if opt_cols: metrics_to_plot.append(opt_cols[0])
    
    sim_cols = [c for c in cols if "SimCSE" in c]
    if sim_cols: metrics_to_plot.append(sim_cols[0])

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
    
    df = load_all_methods()
    
    if df.empty:
        print(f"{RED}Aucune donnée trouvée.{RESET}")
        return

    # 1. Terminal Output
    print("\n" + "="*150)
    print(f"{BOLD} TABLEAUX COMPARATIFS (TERMINAL){RESET}")
    print("="*150)
    for ds in DATASETS:
        print(f"\n{BOLD}>>> DATASET: {ds.upper()}{RESET}")
        print(generate_comparison_table(df, ds, "terminal"))

    # Texte Introductif Markdown
    intro_md = """
## Détails de l'Expérience

Les résultats ci-dessous comparent différentes stratégies de décodage sur le modèle **GPT2-XL**.

* **Wikitext-103** : 100 exemples générés, longueur de décodage = 256 tokens.
* **CC-News & BookCorpus** : 50 exemples générés, longueur de décodage = 32 tokens.
* **Paramètres Contrastive/Epsilon** :
    * Wikitext/BookCorpus : k=10, alpha=0.6
    * CC-News : k=50, alpha=0.2 (Optimisé via Grid Search)
"""

    # 2. Markdown Simple
    print("\n" + "="*150)
    print(f"{BOLD} MARKDOWN GITHUB{RESET}")
    print("="*150)
    print(intro_md)
    for ds in DATASETS:
        print(f"\n### Résultats : {ds.upper()}")
        print(generate_comparison_table(df, ds, "markdown"))

    # 3. Markdown Latex
    print("\n" + "="*150)
    print(f"{BOLD} MARKDOWN COLORÉ (LATEX){RESET}")
    print("="*150)
    print(intro_md)
    for ds in DATASETS:
        print(f"\n### {ds.upper()}")
        print(generate_comparison_table(df, ds, "markdown_color"))

    # 4. Graphs
    print("\n" + "="*150)
    print(f"{BOLD} GÉNÉRATION DES GRAPHIQUES{RESET}")
    plot_comparative_graphs(df)

if __name__ == "__main__":
    main()