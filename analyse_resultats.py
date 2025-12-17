import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration de l'affichage de pandas pour une meilleure lisibilité dans le terminal
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

def parse_filename(filepath):
    """Analyse le nom du fichier pour extraire le dataset et la stratégie."""
    basename = os.path.basename(filepath)
    
    # Isole la partie principale du nom de fichier, ex: 'book_greedy_gpt2-xl_256'
    if '_gpt2-xl_256' in basename:
        core_name = basename.split('_gpt2-xl_256')[0]
    else:
        core_name = basename.rsplit('_result.json', 1)[0]  # Fallback

    parts = core_name.split('_')
    dataset = parts[0]
    strategy = '_'.join(parts[1:])

    # Nettoyage des noms de stratégie pour l'affichage
    strategy_map = {
        'p-0.95': 'Nucleus (p=0.95)',
        'typical-0.95': 'Typical (p=0.95)',
        'greedy': 'Greedy Search'
    }
    return dataset, strategy_map.get(strategy, strategy)

def load_all_results(base_dir='open_text_gen'):
    """Charge tous les résultats d'évaluation et les fusionne."""
    all_results_map = {}

    # Trouve tous les fichiers de résultats de diversité/mauve/longueur
    result_files = glob.glob(os.path.join(base_dir, '**/*_diversity_mauve_gen_length_result.json'), recursive=True)

    for f_path in result_files:
        try:
            dataset, strategy = parse_filename(f_path)
            with open(f_path, 'r') as f:
                data = json.load(f)[0]
            
            key = (dataset, strategy)
            all_results_map[key] = {
                'Dataset': dataset,
                'Strategy': strategy,
                'MAUVE': data['mauve_dict']['mauve'],
                'Gen_Length': data['gen_length_dict']['gen_length_mean'],
                'Diversity_rep2': data['diversity_dict']['rep_2'],
                'Diversity_rep3': data['diversity_dict']['rep_3'],
                'Diversity_rep4': data['diversity_dict']['rep_4'],
            }
        except Exception as e:
            print(f"AVERTISSEMENT: Impossible de traiter {f_path}: {e}")

    # Trouve les fichiers de cohérence et les fusionne
    coherence_files = glob.glob(os.path.join(base_dir, '**/*_coherence_result.json'), recursive=True)

    for c_path in coherence_files:
        try:
            dataset, strategy = parse_filename(c_path)
            key = (dataset, strategy)
            if key in all_results_map:
                with open(c_path, 'r') as f:
                    c_data = json.load(f)[0]
                all_results_map[key]['Coherence'] = float(c_data['coherence_mean'])
        except Exception as e:
            print(f"AVERTISSEMENT: Impossible de traiter le fichier de cohérence {c_path}: {e}")

    # Finalisation du DataFrame
    final_list = list(all_results_map.values())
    for res in final_list:
        if 'Coherence' not in res:
            res['Coherence'] = np.nan
            
    return pd.DataFrame(final_list)

def plot_grouped_bar(df, metric, title, higher_is_better=True, output_filename=None):
    """Crée un graphique à barres groupées et le sauvegarde dans un fichier."""
    # S'assurer que la colonne de la métrique existe et n'est pas vide
    if metric not in df.columns or df[metric].isnull().all():
        print(f"AVERTISSEMENT: Métrique '{metric}' non trouvée ou vide. Graphique non généré.")
        return
        
    pivot_df = df.pivot(index='Dataset', columns='Strategy', values=metric)
    
    ax = pivot_df.plot(kind='bar', figsize=(12, 7), rot=0, colormap='viridis')
    
    # Ajout des valeurs sur les barres
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', label_type='edge', padding=2)
        
    plt.title(title, fontsize=16)
    direction = "(plus haut = mieux)" if higher_is_better else "(plus bas = mieux)"
    plt.ylabel(f"{metric} {direction}", fontsize=12)
    plt.xlabel('Dataset', fontsize=12)
    plt.legend(title='Stratégie', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajustement de la mise en page pour inclure la légende
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    if output_filename:
        plt.savefig(output_filename)
        print(f"Graphique sauvegardé : {output_filename}")
    else:
        plt.show()
    
    plt.close() # Libère la mémoire

def main():
    """Fonction principale du script."""
    print("Lancement de l'analyse des résultats...")
    
    # Charger les données
    results_df = load_all_results()
    
    if results_df.empty:
        print("ERREUR: Aucun fichier de résultat trouvé. Vérifiez les chemins et les noms des fichiers.")
        return

    # Afficher le tableau
    print("\n" + "="*80)
    print("TABLEAU COMPARATIF DES RÉSULTATS")
    print("="*80)
    full_table = results_df.set_index(['Dataset', 'Strategy']).sort_index()
    print(full_table)
    print("="*80 + "\n")
    
    # Générer et sauvegarder les graphiques
    print("Génération des graphiques comparatifs...")
    
    plot_grouped_bar(results_df, 'Coherence', 'Comparaison de la Cohérence', 
                     higher_is_better=True, output_filename='comparaison_coherence.png')
                     
    plot_grouped_bar(results_df, 'MAUVE', 'Comparaison du score MAUVE', 
                     higher_is_better=True, output_filename='comparaison_mauve.png')
                     
    plot_grouped_bar(results_df, 'Diversity_rep2', 'Taux de Répétition des Bi-grams (rep-2)', 
                     higher_is_better=False, output_filename='comparaison_rep2.png')
                     
    plot_grouped_bar(results_df, 'Diversity_rep3', 'Taux de Répétition des Tri-grams (rep-3)', 
                     higher_is_better=False, output_filename='comparaison_rep3.png')
                     
    plot_grouped_bar(results_df, 'Diversity_rep4', 'Taux de Répétition des Quad-grams (rep-4)', 
                     higher_is_better=False, output_filename='comparaison_rep4.png')
                     
    print("\nAnalyse terminée. Les graphiques ont été sauvegardés sous forme de fichiers .png dans le répertoire.")


if __name__ == "__main__":
    main()
