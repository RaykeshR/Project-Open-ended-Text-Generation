import os
import subprocess
import sys
import time
import datetime

def format_timedelta(seconds):
    """Formate les secondes en H:M:S"""
    return str(datetime.timedelta(seconds=int(seconds)))

def main():
    # =========================================================================
    # 1. CONFIGURATION
    # =========================================================================
    
    # Liste des modèles à tester (doit correspondre à ceux du Grid Search pour comparer)
    gen_models = [
        'gpt2',          # Rapide pour tester
        #'gpt2-xl',       # Votre modèle principal
        # Ajoutez vos autres modèles ici...
    ]

    # Liste des juges pour la cohérence
    coherence_models = [
        'facebook/opt-125m',
        # 'facebook/opt-2.7b' 
    ]

    # Paramètres globaux
    dataset_name = 'wikitext' 
    dataset_config = 'wikitext-103-raw-v1'
    dataset_split = 'test' 
    num_prefixes = 100         
    decoding_len = 256        
    
    # DÉFINITION DES STRATÉGIES "BASELINE"
    # Ce sont les méthodes classiques auxquelles on se compare
    baselines = [
        {
            'name': 'Greedy Search',
            'strategy_flag': 'greedy',
            'extra_args': [],
            'file_suffix': 'greedy'
        },
        {
            'name': 'Nucleus Sampling (p=0.95)',
            'strategy_flag': 'nucleus',
            'extra_args': ['--probs', '0.95'],
            'file_suffix': 'p-0.95'
        },
        {
            'name': 'Typical Sampling (p=0.95)',
            'strategy_flag': 'typical',
            'extra_args': ['--probs', '0.95'],
            'file_suffix': 'typical-0.95'
        }
    ]

    # =========================================================================
    # 2. EXÉCUTION
    # =========================================================================
    
    python_exe = sys.executable
    # On utilise le MÊME dossier que le grid search pour faciliter l'analyse groupée
    output_dir = f'open_text_gen/{dataset_name}_grid_search'
    os.makedirs(output_dir, exist_ok=True)

    total_runs = len(gen_models) * len(baselines)
    print(f" Démarrage des Baselines : {total_runs} configurations à lancer.")
    print(f" Dossier de sortie (output_dir) : {output_dir}")

    global_start_time = time.time()
    iteration_times = []
    run_count = 0

    for model_name in gen_models:
        safe_model_name = model_name.replace('/', '-')
        
        for strat in baselines:
            run_count += 1
            iter_start_time = time.time()
            
            print(f"\n{'='*80}")
            print(f"Configuration {run_count}/{total_runs} : Modèle={model_name} | Stratégie={strat['name']}")
            print(f"{'='*80}")

            # --- A. GÉNÉRATION ---
            # Construction du nom de fichier attendu (format standard du projet)
            # Ex: wikitext_greedy_gpt2-xl_256.jsonl
            # Note: Le script generate.py construit souvent le nom lui-même, mais on doit le deviner pour l'étape suivante.
            filename_base = f'{dataset_name}_{strat["file_suffix"]}_{safe_model_name}_{decoding_len}'
            jsonl_output_path = f'{output_dir}/{filename_base}.jsonl'

            print(f" 1. Génération du texte ({strat['name']})...")
            
            gen_cmd = [
                python_exe, 'open_text_gen/generate.py',
                '--model_name', model_name,
                '--dataset_name', dataset_name,
                '--dataset_config', dataset_config,
                '--dataset_split', dataset_split,
                '--output_dir', output_dir,
                '--decoding_strategy', strat['strategy_flag'],
                '--decoding_len', str(decoding_len),
                '--num_prefixes', str(num_prefixes)
            ] + strat['extra_args']

            try:
                # On capture la sortie pour éviter de polluer, sauf en cas d'erreur
                subprocess.run(gen_cmd, check=True) #, capture_output=False)
            except subprocess.CalledProcessError as e:
                print(f"\n\033[31m[ERREUR CRITIQUE] La génération a échoué pour {strat['name']}.\033[0m")
                print(f"\033[91mCode retour : {e.returncode}\033[0m")
                continue

            # Vérification que le fichier a bien été créé (parfois generate.py change légèrement le nom)
            # Si le fichier exact n'existe pas, on essaie de le trouver avec glob ou on avertit.
            if not os.path.exists(jsonl_output_path):
                # Fallback : certains scripts nomment différemment (ex: sans le _256 à la fin)
                fallback_path = f'{output_dir}/{dataset_name}_{strat["file_suffix"]}_{safe_model_name}.jsonl'
                if os.path.exists(fallback_path):
                    jsonl_output_path = fallback_path
                else:
                    print(f"\033[33m[ATTENTION] Impossible de trouver le fichier généré attendu :\n{jsonl_output_path}\nPassage aux évaluations suivantes impossible pour cette config.\033[0m")
                    continue

            # --- B. ÉVALUATION (COHÉRENCE) ---
            for coh_model in coherence_models:
                print(f" 2. Cohérence (Juge: {coh_model})...")
                coh_cmd = [
                    python_exe, 'open_text_gen/compute_coherence.py',
                    '--opt_model_name', coh_model,
                    '--test_path', jsonl_output_path
                ]
                try:
                    subprocess.run(coh_cmd, check=True)
                except subprocess.CalledProcessError:
                    print(f"\033[91m -> Erreur lors du calcul de cohérence ({coh_model})\033[0m")

            # --- C. ÉVALUATION (DIVERSITÉ & MAUVE) ---
            print(" 3. Diversité, MAUVE & Longueur...")
            div_output_path = f'{output_dir}/{filename_base}_diversity_mauve_gen_length_result.json'
            
            div_cmd = [
                python_exe, 'open_text_gen/measure_diversity_mauve_gen_length.py',
                '--test_path', jsonl_output_path,
                '--result_path', div_output_path
            ]
            try:
                subprocess.run(div_cmd, check=True)
            except subprocess.CalledProcessError:
                 print(f"\033[91m -> Erreur lors du calcul Diversité/MAUVE\033[0m")

            # --- CALCULS DE TEMPS ---
            iter_end_time = time.time()
            iter_duration = iter_end_time - iter_start_time
            iteration_times.append(iter_duration)
            
            avg_duration = sum(iteration_times) / len(iteration_times)
            total_elapsed = iter_end_time - global_start_time
            remaining_iters = total_runs - run_count
            estimated_remaining = avg_duration * remaining_iters
            
            print(f"\n Fin de l'étape {run_count}/{total_runs}")
            print(f"  \033[32m ➤ \033[0m Durée : {format_timedelta(iter_duration)}")
            print(f"  \033[32m ➤ \033[0m Moyenne : {format_timedelta(avg_duration)}")
            print(f"  \033[34m ➤ \033[0m RESTANT ESTIMÉ : {format_timedelta(estimated_remaining)}")

    print(f"\n\033[42m[SUCCÈS] Tous les baselines sont terminés en {format_timedelta(time.time() - global_start_time)}.\033[0m")

if __name__ == '__main__':
    main()