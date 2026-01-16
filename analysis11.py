import json
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
BASE_DIR = "open_text_gen"  # Dossier racine contenant les sous-dossiers
DATASETS = ["wikitext", "bookcorpus", "cc_news"]
METRICS_COLS = ['MAUVE', 'Diversity', 'Coh_Likelihood', 'Coh_SimCSE', 'Perplexity', 'Gen_Length']

# --- Codes couleurs ANSI (Terminal) ---
RED = "\033[91m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[96m"

def get_json_val(data, keys, default=0.0):
    """Récupère une valeur imbriquée dans un dict JSON de manière sécurisée."""
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

def load_data():
    """Charge et agrège les données JSON des dossiers de grid search."""
    all_data = []

    # Regex pour capturer k et alpha du nom de fichier
    # Ex: wikitext_epsilon_k10_alpha0.2_gpt2-xl_diversity...
    filename_pattern = re.compile(r'epsilon_k(\d+)_alpha([\d\.]+)_gpt2-xl')

    for dataset in DATASETS:
        search_dir = os.path.join(BASE_DIR, f"{dataset}_epsilon_grid_search")
        if not os.path.exists(search_dir):
            print(f"{RED}Attention: Dossier introuvable {search_dir}{RESET}")
            continue
            
        # On utilise les fichiers diversity comme "ancre" pour trouver les paires (k, alpha)
        diversity_files = glob.glob(os.path.join(search_dir, "*diversity_mauve_gen_length_result.json"))

        for div_file in diversity_files:
            basename = os.path.basename(div_file)
            match = filename_pattern.search(basename)
            
            if not match:
                continue

            k = int(match.group(1))
            alpha = float(match.group(2))
            
            # Préfixe commun pour récupérer les autres fichiers frères
            # Ex: wikitext_epsilon_k10_alpha0.2_gpt2-xl
            file_prefix = basename.split("_diversity")[0]
            
            row = {
                'Dataset': dataset,
                'K': k,
                'Alpha': alpha,
                'Model_Ref': 'gpt2-xl'
            }

            # 1. Load Diversity / MAUVE
            try:
                with open(div_file, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                    if isinstance(d, list): d = d[0]
                    row['MAUVE'] = get_json_val(d, ['mauve_dict', 'mauve_mean'])
                    row['Diversity'] = get_json_val(d, ['diversity_dict', 'prediction_div_mean'])
                    row['Gen_Length'] = get_json_val(d, ['gen_length_dict', 'gen_len_mean'])
            except Exception as e:
                pass

            # 2. Load Perplexity
            ppl_file = os.path.join(search_dir, f"{file_prefix}_perplexity_result.json")
            if os.path.exists(ppl_file):
                try:
                    with open(ppl_file, 'r', encoding='utf-8') as f:
                        d = json.load(f)
                        # Parfois c'est 'mean_perplexity', parfois 'ppl'
                        val = d.get('mean_perplexity', d.get('ppl', 0.0))
                        row['Perplexity'] = float(val)
                except: 
                    row['Perplexity'] = float('nan')
            else:
                row['Perplexity'] = float('nan')

            # 3. Load Coherence (Likelihood - OPT)
            # Cherche n'importe quel fichier coherence qui n'est pas simcse
            coh_files = glob.glob(os.path.join(search_dir, f"{file_prefix}*_coherence_result.json"))
            coh_files = [x for x in coh_files if 'simcse' not in x]
            if coh_files:
                try:
                    with open(coh_files[0], 'r', encoding='utf-8') as f:
                        d = json.load(f)
                        if isinstance(d, list): d = d[0]
                        row['Coh_Likelihood'] = float(d.get('coherence_mean', 0.0))
                except: 
                    row['Coh_Likelihood'] = 0.0
            else:
                row['Coh_Likelihood'] = 0.0

            # 4. Load SimCSE
            simcse_file = os.path.join(search_dir, f"{file_prefix}_simcse_result.json")
            if os.path.exists(simcse_file):
                try:
                    with open(simcse_file, 'r', encoding='utf-8') as f:
                        d = json.load(f)
                        row['Coh_SimCSE'] = float(d.get('coherence_mean', 0.0))
                except: 
                    row['Coh_SimCSE'] = 0.0
            else:
                row['Coh_SimCSE'] = 0.0

            all_data.append(row)

    return pd.DataFrame(all_data)

def format_val(val, metric, is_best, color_mode="none"):
    """Formate les valeurs pour les tableaux avec coloration conditionnelle."""
    if pd.isna(val): return "N/A"
    
    fmt = "{:.3f}"
    if metric == 'Perplexity': fmt = "{:.2f}"
    if metric == 'MAUVE': fmt = "{:.2f}"
    if metric == 'Gen_Length': fmt = "{:.1f}"
    
    txt = fmt.format(val)
    
    if is_best:
        if color_mode == "terminal":
            return f"{RED}{BOLD}{txt}{RESET}"
        elif color_mode == "markdown_color":
            # Syntaxe LaTeX pour Jupyter/Markdown supportant le rendu math
            return f"$\\color{{red}}{{\\mathbf{{{txt}}}}}$"
        elif color_mode == "markdown":
            return f"**{txt}**"
    
    return txt

def generate_table(df, dataset_name, output_format="terminal"):
    """Génère un tableau formaté pour un dataset donné."""
    # Trier par K puis Alpha
    sub_df = df[df['Dataset'] == dataset_name].sort_values(by=['K', 'Alpha'])
    
    if sub_df.empty:
        return "Pas de données pour ce dataset."

    # Identifier les meilleurs scores pour ce dataset
    best_idxs = {}
    for col in METRICS_COLS:
        if col in sub_df.columns and not sub_df[col].isna().all():
            if col == 'Perplexity':
                best_idxs[col] = sub_df[col].idxmin()
            elif col == 'Gen_Length':
                # Pour la longueur, "meilleur" est subjectif, on ne colore généralement pas, 
                # ou on vise une cible. Ici on ne colore pas.
                best_idxs[col] = None 
            else:
                best_idxs[col] = sub_df[col].idxmax()

    # Construction Header
    lines = []
    if output_format == "terminal":
        header = f"{CYAN}{'K':<4} {'Alpha':<6} " + " ".join([f"{c:<14}" for c in METRICS_COLS]) + f"{RESET}"
        lines.append(header)
        lines.append("-" * len(re.sub(r'\033\[[0-9;]*m', '', header))) # Longueur sans codes couleurs
    else:
        header = "| K | Alpha | " + " | ".join(METRICS_COLS) + " |"
        lines.append(header)
        lines.append("|:---:|:---:| " + " | ".join([":---:" for _ in METRICS_COLS]) + " |")

    # Construction Rows
    for idx, row in sub_df.iterrows():
        k_str = str(row['K'])
        a_str = str(row['Alpha'])
        
        row_str = ""
        if output_format == "terminal":
            row_str += f"{k_str:<4} {a_str:<6} "
        else:
            row_str += f"| {k_str} | {a_str} | "
            
        for col in METRICS_COLS:
            val = row.get(col, float('nan'))
            is_best = (idx == best_idxs.get(col))
            
            fmt_val = format_val(val, col, is_best, color_mode=output_format)
            
            if output_format == "terminal":
                # Calcul padding pour alignement terminal (approx) en ignorant les codes ANSI
                plain_len = len(re.sub(r'\033\[[0-9;]*m', '', fmt_val))
                padding = " " * (14 - plain_len)
                row_str += f"{fmt_val}{padding} "
            else:
                row_str += f"{fmt_val} | "
        
        lines.append(row_str)
        
    return "\n".join(lines)

def plot_graphs(df):
    """Génère des graphiques comparatifs Seaborn."""
    sns.set_theme(style="whitegrid")
    
    # On veut un graphe par Métrique
    metrics_to_plot = ['MAUVE', 'Coh_Likelihood', 'Perplexity', 'Diversity']
    
    # Créer un dossier pour les graphes
    output_dir = "plots_epsilon_grid_search"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for metric in metrics_to_plot:
        if metric not in df.columns: continue
        
        # Créer une figure avec 1 ligne et 3 colonnes (pour les 3 datasets)
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
        fig.suptitle(f"Evolution de {metric} selon Alpha et K (Contrastive Search)", fontsize=16)
        
        for i, dataset in enumerate(DATASETS):
            ax = axes[i]
            data = df[df['Dataset'] == dataset].sort_values(by='Alpha')
            
            if data.empty:
                ax.set_title(f"{dataset} (Aucune donnée)")
                continue

            # Lineplot: X=Alpha, Y=Metric, Hue=K
            sns.lineplot(
                data=data, 
                x='Alpha', 
                y=metric, 
                hue='K', 
                palette="tab10", 
                marker="o",
                linewidth=2.5,
                ax=ax
            )
            
            ax.set_title(dataset.upper(), fontsize=14)
            ax.set_xlabel("Alpha", fontsize=12)
            if i == 0: 
                ax.set_ylabel(metric, fontsize=12)
            else: 
                ax.set_ylabel("")
            
            ax.legend(title='K')
        
        plt.tight_layout()
        filename = os.path.join(output_dir, f"analysis_{metric}.png")
        plt.savefig(filename, dpi=150)
        print(f"Graphique sauvegardé : {filename}")
        plt.close()

def main():
    print(f"{BOLD}Chargement des données...{RESET}")
    df = load_data()
    
    if df.empty:
        print(f"{RED}Aucune donnée trouvée. Vérifiez que les dossiers '{BASE_DIR}/[dataset]_epsilon_grid_search' existent et contiennent des résultats.{RESET}")
        return

    # 1. Version Terminal
    print("\n" + "="*90)
    print(f"{BOLD} RAPPORT TERMINAL (Aligné & Coloré){RESET}")
    print("="*90)
    for ds in DATASETS:
        print(f"\n{BOLD}--- {ds.upper()} ---{RESET}")
        print(generate_table(df, ds, "terminal"))

    # 2. Markdown Simple
    print("\n" + "="*90)
    print(f"{BOLD} MARKDOWN GITHUB (Simple){RESET}")
    print("="*90)
    for ds in DATASETS:
        print(f"\n### {ds.upper()}")
        print(generate_table(df, ds, "markdown"))

    # 3. Markdown Latex
    print("\n" + "="*90)
    print(f"{BOLD} MARKDOWN RAPPORT (LaTeX Colors - Pour Notebook/PDF){RESET}")
    print("="*90)
    for ds in DATASETS:
        print(f"\n### {ds.upper()}")
        print(generate_table(df, ds, "markdown_color"))

    # 4. Graphs
    print("\n" + "="*90)
    print(f"{BOLD} GÉNÉRATION DES GRAPHIQUES{RESET}")
    print("="*90)
    plot_graphs(df)

if __name__ == "__main__":
    main()