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
    
    # Liste des modèles de génération
    gen_models = [
        # --- FAMILLE GPT-2 ---
        # 'gpt2',          # ~124M params (Très rapide)
        # 'gpt2-medium',   # ~355M params
        # 'gpt2-large',    # ~774M params
        'gpt2-xl',       # ~1.5B params (Celui demandé)

        # --- FAMILLE OPT ---
        # 'facebook/opt-125m',
        # 'facebook/opt-1.3b',
    ]

    # Liste des juges pour la cohérence
    coherence_models = [
        # 'facebook/opt-125m',  # Très rapide, idéal pour tester
        # # # ###########'facebook/opt-350m',  # Bon compromis
        # 'facebook/opt-1.3b', # Plus précis mais demande ~6Go VRAM/RAM
        'facebook/opt-2.7b',  # Le plus précis (celui qui a fait planter mon PC avant)

        # # # --- FAMILLE GPT-2 (Les classiques) ---
        # 'gpt2',          # ~124M params (Très rapide)
        # 'gpt2-medium',   # ~355M params
        # 'gpt2-large',    # ~774M params
        # 'gpt2-xl',        # ~1.5B params (Lourd)
    ]

    # Paramètres globaux
    dataset_name = 'wikitext' # wikitext | cc_news | bookcorpus
    dataset_config = 'wikitext-103-raw-v1' # wikitext-103-raw-v1 | plain_text | plain_text
    dataset_split = 'test' # test | train | train
    num_prefixes = 100         # Nombre d'exemples à générer (100 est standard utilisé pour 'gpt2...'~124M-~1.5B params sinon 5 )
    decoding_len = 256        # Longueur du texte généré (256 est standard utilisé pour 'gpt2...'~124M-~1.5B  params  sinon 16)
    
    # GRILLE DE RECHERCHE EPSILON GREEDY
    # On va tester toutes les combinaisons de ces listes
    alphas = [0.2, 0.4, 0.6, 0.8]
    ks = [5, 10, 50]

    # =========================================================================
    # 2. EXÉCUTION
    # =========================================================================
    
    python_exe = sys.executable
    # Dossier spécifique pour ne pas mélanger avec les baselines
    output_dir = f'open_text_gen/{dataset_name}_epsilon_grid_search'
    os.makedirs(output_dir, exist_ok=True)

    total_runs = len(gen_models) * len(alphas) * len(ks)
    print(f" Démarrage de la Grid Search Epsilon : {total_runs} configurations à lancer.")
    print(f" Dossier de sortie (output_dir) : {output_dir}")

    global_start_time = time.time()
    iteration_times = []
    run_count = 0
    
    # --- LOG DES ERREURS ---
    errors_log = [] 

    for model_name in gen_models:
        safe_model_name = model_name.replace('/', '-')              
        
        for k in ks:
            for alpha in alphas:
                run_count += 1
                iter_start_time = time.time()
                config_name = f"{model_name} | k={k} | alpha={alpha}"
                
                print(f"\n{'='*80}")
                print(f"Configuration {run_count}/{total_runs} : {config_name}")
                print(f"{'='*80}")

                # --- A. GÉNÉRATION ---
                # Construction du nom de fichier attendu (format défini dans generate_epsilon.py)
                # output_filename = f'{args.dataset_name}_epsilon_k{args.k}_alpha{args.alpha}_{safe_model_name}.jsonl'
                filename_base = f'{dataset_name}_epsilon_k{k}_alpha{alpha}_{safe_model_name}'
                jsonl_output_path = f'{output_dir}/{filename_base}.jsonl'

                if os.path.exists(jsonl_output_path):
                    print(f" Fichier existant trouvé, on passe la génération : {jsonl_output_path}")
                else:
                    print(f" 1. Génération Epsilon Greedy...")
                    gen_cmd = [
                        python_exe, 'open_text_gen/generate_epsilon.py',
                        '--model_name', model_name,
                        '--dataset_name', dataset_name,
                        '--dataset_config', dataset_config,
                        '--dataset_split', dataset_split,
                        '--output_dir', output_dir,
                        '--alpha', str(alpha),
                        '--k', str(k),
                        '--decoding_len', str(decoding_len),
                        '--num_prefixes', str(num_prefixes)
                    ]

                    try:
                        subprocess.run(gen_cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"\n\033[31m[ERREUR CRITIQUE] La génération a échoué.\033[0m")
                        errors_log.append({'config': config_name, 'step': 'GÉNÉRATION', 'details': f'Code erreur: {e.returncode}'})
                        continue

                # Vérification fichier
                if not os.path.exists(jsonl_output_path):
                    print(f"\033[33m[ATTENTION] Fichier généré introuvable : {jsonl_output_path}\033[0m")
                    errors_log.append({'config': config_name, 'step': 'FICHIER', 'details': 'JSONL introuvable'})
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
                        print(f"\033[91m -> Erreur cohérence ({coh_model})\033[0m")
                        errors_log.append({'config': config_name, 'step': 'COHÉRENCE', 'details': f'Juge {coh_model}'})

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
                    print(f"\033[91m -> Erreur Diversité/MAUVE\033[0m")
                    errors_log.append({'config': config_name, 'step': 'MAUVE/DIV', 'details': 'Script mesure échoué'})

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

    # =========================================================================
    # 3. RAPPORT FINAL
    # =========================================================================
    total_time = time.time() - global_start_time
    print("\n" + "="*80)
    print(f"RAPPORT FINAL ({format_timedelta(total_time)})")
    print("="*80)
    
    if len(errors_log) == 0:
        print(f"\033[42m SUCCÈS TOTAL \033[0m : 0 erreurs.")
    else:
        print(f"\033[41m {len(errors_log)} ERREURS DÉTECTÉES : \033[0m")
        print("-" * 80)
        print(f"{'CONFIGURATION':<40} | {'ÉTAPE':<12} | {'DÉTAIL'}")
        print("-" * 80)
        for err in errors_log:
            print(f"{err['config']:<40} | \033[31m{err['step']:<12}\033[0m | {err['details']}")
    print("="*80)

if __name__ == '__main__':
    main()
