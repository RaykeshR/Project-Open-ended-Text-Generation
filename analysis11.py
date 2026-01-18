import json
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. CONFIGURATION DES HYPERPARAMÈTRES OPTIMAUX (Le Cœur du Script)
# =============================================================================
# Ici, vous définissez la configuration "Gagnante" pour chaque méthode et chaque dataset.
# Le script ne cherchera QUE les fichiers correspondant à ces valeurs.

OPTIMAL_PARAMS = {
    "wikitext": {
        "nucleus":     0.95,
        "typical":     0.95,
        "contrastive": {"k": 10, "alpha": 0.6}, # Standard
        "epsilon":     {"k": 10, "alpha": 0.6}  # Standard
    },
    "bookcorpus": {
        "nucleus":     0.95,
        "typical":     0.95,
        "contrastive": {"k": 10, "alpha": 0.6},
        "epsilon":     {"k": 10, "alpha": 0.6}
    },
    "cc_news": {
        "nucleus":     0.95,
        "typical":     0.95,
        "contrastive": {"k": 10, "alpha": 0.6}, # On garde le standard pour Contrastive
        "epsilon":     {"k": 5, "alpha": 0.6}  # On prend l'optimisé pour Epsilon
    }
}

# =============================================================================
# 2. CONFIGURATION GÉNÉRALE & AFFICHAGE
# =============================================================================

BASE_DIR = "open_text_gen"
DATASETS = ["wikitext", "bookcorpus", "cc_news"]
MODEL_NAME_FILTER = "gpt2-xl" 

# Affichage dans le terminal : Voulez-vous une colonne "Model" ?
SHOW_MODEL_COL_TERMINAL = False 

# Quels juges de cohérence (Likelihood) afficher ?
# Commentez ceux que vous ne voulez pas voir.
SELECTED_COH_JUDGES = [
    # "opt-125m", 
    # "opt-1.3b",
    "opt-2.7b",     # Le plus précis
    # "gpt2-xl"
]

# =============================================================================
# CODES & UTILITAIRES
# =============================================================================
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
    Identifie la méthode et vérifie si elle correspond aux OPTIMAL_PARAMS du dataset.
    """
    base = os.path.basename(filename)
    
    if MODEL_NAME_FILTER not in base:
        return "Skip", "Mauvais modèle"

    # Récupération des params optimaux pour ce dataset
    opt = OPTIMAL_PARAMS[dataset]

    # --- 1. GREEDY ---
    if "greedy" in base:
        return "Greedy Search", None
    
    # --- 2. NUCLEUS ---
    target_p = opt["nucleus"]
    if f"p-{target_p}" in base or (f"nucleus" in base and str(target_p) in base):
        return f"Nucleus Sampling (p={target_p})", None
    
    # --- 3. TYPICAL ---
    target_p = opt["typical"]
    if "typical" in base and str(target_p) in base:
        return f"Typical Sampling (p={target_p})", None
    
    # --- 4. EPSILON ---
    if "epsilon" in base:
        k = opt["epsilon"]["k"]
        a = opt["epsilon"]["alpha"]
        # On vérifie si le fichier contient kX et alphaY
        if f"k{k}_" in base and f"alpha{a}" in base:
            return f"Epsilon Sampling (k={k}, α={a})", None
        return "Skip", f"Epsilon mismatch (fichier: {base} | attendu: k{k} a{a})"

    # --- 5. CONTRASTIVE ---
    # Peut être nommé "contrastive-alpha-X" (k souvent implicite 10 ou dans args) 
    # OU "k10_a0.6_e0.0" (généré par generate.py)
    if "contrastive" in base or ("e0.0" in base and "k" in base and "a" in base):
        k = opt["contrastive"]["k"]
        a = opt["contrastive"]["alpha"]
        
        # Vérification patterns
        match_k = (f"k{k}" in base) or ("contrastive" in base and k==10) # Souvent implicite k=10 pour contrastive script
        match_a = (f"alpha-{a}" in base) or (f"a{a}" in base) or (f"alpha{a}" in base)
        
        if match_k and match_a:
            return f"Contrastive Search (k={k}, α={a})", None
            
        return "Skip", f"Contrastive mismatch (fichier: {base} | attendu: k{k} a{a})"

    return "Skip", "Méthode inconnue"

def load_all_methods():
    all_data = []
    print(f"{YELLOW}Chargement et filtrage des fichiers...{RESET}")

    for dataset in DATASETS:
        # Dossiers à scanner
        dirs_to_scan = [
            os.path.join(BASE_DIR, f"{dataset}_grid_search"),
        ]
        # Ajout des dossiers Epsilon spécifiques
        if dataset == "wikitext":
            dirs_to_scan.append(os.path.join(BASE_DIR, f"{dataset}_epsilon_grid_search_256"))
        else:
            dirs_to_scan.append(os.path.join(BASE_DIR, f"{dataset}_epsilon_grid_search"))

        for folder in dirs_to_scan:
            if not os.path.exists(folder):
                continue

            # On utilise les fichiers de résultats diversity comme ancre
            result_files = glob.glob(os.path.join(folder, "*diversity_mauve_gen_length_result.json"))

            for res_file in result_files:
                basename = os.path.basename(res_file)
                
                if dataset not in basename: continue

                method_name, skip_reason = detect_method_name(basename, dataset)
                
                if method_name == "Skip":
                    # print(f"  [SKIP] {basename} -> {skip_reason}")
                    continue

                print(f"  [OK] {dataset:<12} | {method_name:<35} | {res_file}")

                file_prefix = res_file.replace("_diversity_mauve_gen_length_result.json", "")
                
                row = {
                    'Dataset': dataset,
                    'Method': method_name,
                    'Model': MODEL_NAME_FILTER
                }

                # --- 1. Metrics Standard ---
                try:
                    with open(res_file, 'r', encoding='utf-8') as f:
                        d = json.load(f)
                        if isinstance(d, list): d = d[0]
                        row['MAUVE'] = get_json_val(d, ['mauve_dict', 'mauve_mean'])
                        row['Diversity'] = get_json_val(d, ['diversity_dict', 'prediction_div_mean'])
                        row['Gen_Length'] = get_json_val(d, ['gen_length_dict', 'gen_len_mean'])
                except: pass

                # --- 2. Perplexity ---
                ppl_file = f"{file_prefix}_perplexity_result.json"
                if os.path.exists(ppl_file):
                    try:
                        with open(ppl_file, 'r', encoding='utf-8') as f:
                            d = json.load(f)
                            val = d.get('mean_perplexity') or d.get('ppl')
                            row['Perplexity'] = float(val) if val else float('nan')
                    except: row['Perplexity'] = float('nan')
                else: 
                    row['Perplexity'] = float('nan')

                # --- 3. Coherence (Likelihood) - Filtrage par Juge ---
                coh_files = glob.glob(f"{file_prefix}*_coherence_result.json")
                
                for cf in coh_files:
                    if "simcse" in os.path.basename(cf): continue
                    
                    # On ne traite le fichier que si le juge est dans SELECTED_COH_JUDGES
                    judge_name = None
                    for allowed in SELECTED_COH_JUDGES:
                        # On check si "opt-2.7b" est dans le nom du fichier
                        if allowed.lower() in os.path.basename(cf).lower():
                            judge_name = allowed
                            break
                    
                    if judge_name:
                        try:
                            with open(cf, 'r', encoding='utf-8') as f:
                                d = json.load(f)
                                if isinstance(d, list): d = d[0]
                                val = float(d.get('coherence_mean', 0.0))
                                # On formate le nom de la colonne proprement
                                row[f'Coh_Like_{judge_name}'] = val
                        except: pass

                # --- 4. SimCSE ---
                simcse_file = f"{file_prefix}_simcse_result.json"
                if os.path.exists(simcse_file):
                    try:
                        with open(simcse_file, 'r', encoding='utf-8') as f:
                            d = json.load(f)
                            row['Coh_Sem_SimCSE'] = float(d.get('coherence_mean', 0.0))
                    except: row['Coh_Sem_SimCSE'] = 0.0
                else: 
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

    # --- Sélection des Colonnes ---
    # Base
    cols_to_show = ['MAUVE', 'Diversity', 'Perplexity', 'Gen_Length']
    
    # Ajout dynamique des cohérences trouvées (filtrées par les juges sélectionnés)
    coh_like_cols = sorted([c for c in df.columns if c.startswith('Coh_Like_')])
    coh_sem_cols = ['Coh_Sem_SimCSE'] if 'Coh_Sem_SimCSE' in df.columns else []
    
    final_cols = cols_to_show + coh_like_cols + coh_sem_cols
    # On ne garde que celles qui existent vraiment dans le DF
    final_cols = [c for c in final_cols if c in sub_df.columns]

    # --- Ordre des Méthodes ---
    opt = OPTIMAL_PARAMS[dataset_name]
    
    # On construit les noms exacts attendus pour le tri
    method_order = [
        "Greedy Search", 
        f"Nucleus Sampling (p={opt['nucleus']})", 
        f"Typical Sampling (p={opt['typical']})", 
        f"Contrastive Search (k={opt['contrastive']['k']}, α={opt['contrastive']['alpha']})",
        f"Epsilon Sampling (k={opt['epsilon']['k']}, α={opt['epsilon']['alpha']})"
    ]
    
    sub_df['Method'] = pd.Categorical(sub_df['Method'], categories=method_order, ordered=True)
    sub_df = sub_df.sort_values('Method')

    # --- Meilleurs Scores ---
    best_idxs = {}
    for col in final_cols:
        if not sub_df[col].isna().all():
            if col == 'Perplexity':
                best_idxs[col] = sub_df[col].idxmin()
            elif col == 'Gen_Length':
                best_idxs[col] = None 
            else:
                best_idxs[col] = sub_df[col].idxmax()

    lines = []
    
    # --- En-têtes ---
    col_width = 15
    method_width = 42
    
    show_model_col = (output_format == "terminal" and SHOW_MODEL_COL_TERMINAL)

    if output_format == "terminal":
        header = f"{CYAN}{'Method':<{method_width}} "
        if show_model_col: header += f"{'Model':<12} "
        header += " ".join([f"{c:<{col_width}}" for c in final_cols]) + f"{RESET}"
        lines.append(header)
        lines.append("-" * len(re.sub(r'\033\[[0-9;]*m', '', header)))
    else:
        # Markdown
        header = "| Method | " + " | ".join(final_cols) + " |"
        lines.append(header)
        lines.append("|:---| " + " | ".join([":---:" for _ in final_cols]) + " |")

    # --- Lignes ---
    for idx, row in sub_df.iterrows():
        m_name = str(row['Method'])
        row_str = ""
        
        if output_format == "terminal":
            row_str += f"{m_name:<{method_width}} "
            if show_model_col:
                row_str += f"{str(row.get('Model','')):<12} "
        else:
            row_str += f"| {m_name} | "
            
        for col in final_cols:
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

    metrics = ['MAUVE', 'Diversity', 'Perplexity']
    # Ajout auto des cohérences
    metrics += [c for c in df.columns if c.startswith('Coh_')]

    for metric in metrics:
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
    print(f"{BOLD}Analyse Comparative Finale (Multi-Hyperparamètres){RESET}")
    print(f"Modèle filtré : {MODEL_NAME_FILTER}")
    print(f"Juges sélectionnés : {SELECTED_COH_JUDGES}")
    
    df = load_all_methods()
    
    if df.empty:
        print(f"{RED}Aucune donnée trouvée correspondante aux critères.{RESET}")
        return

    # 1. Terminal Output
    print("\n" + "="*160)
    print(f"{BOLD} TABLEAUX COMPARATIFS (TERMINAL){RESET}")
    print("="*160)
    for ds in DATASETS:
        print(f"\n{BOLD}>>> DATASET: {ds.upper()}{RESET}")
        print(generate_comparison_table(df, ds, "terminal"))

    # Texte Introductif Markdown
    intro_md = """
## Détails de l'Expérience

Les résultats ci-dessous comparent différentes stratégies de décodage sur le modèle **GPT2-XL**.

* **Wikitext-103** : 100 exemples générés, longueur de décodage = 256 tokens.
* **CC-News & BookCorpus** : 50 exemples générés, longueur de décodage = 32 tokens.
* **Choix des Hyperparamètres** :
    * Les paramètres (k, alpha, p) ont été sélectionnés individuellement pour chaque dataset afin de maximiser le compromis MAUVE/Diversité.
"""

    # 2. Markdown Simple
    print("\n" + "="*160)
    print(f"{BOLD} MARKDOWN GITHUB{RESET}")
    print("="*160)
    print(intro_md)
    for ds in DATASETS:
        print(f"\n### Résultats : {ds.upper()}")
        print(generate_comparison_table(df, ds, "markdown"))

    # 3. Markdown Latex
    print("\n" + "="*160)
    print(f"{BOLD} MARKDOWN COLORÉ (LATEX){RESET}")
    print("="*160)
    print(intro_md)
    for ds in DATASETS:
        print(f"\n### {ds.upper()}")
        print(generate_comparison_table(df, ds, "markdown_color"))

    # 4. Graphs
    print("\n" + "="*160)
    print(f"{BOLD} GÉNÉRATION DES GRAPHIQUES{RESET}")
    plot_comparative_graphs(df)

if __name__ == "__main__":
    main()