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
    
    # Liste des modèles de génération (du plus léger au plus lourd)
    # Vous pouvez commenter ceux que vous ne voulez pas tester (ex: gpt2-xl si manque de RAM)
    gen_models = [
        # --- FAMILLE GPT-2 (Les classiques) ---
        'gpt2',          # ~124M params (Très rapide)
        'gpt2-medium',   # ~355M params
        'gpt2-large',    # ~774M params
        'gpt2-xl'        # ~1.5B params (Lourd)

        # # --- FAMILLE QWEN (Le top actuel en "petits" modèles) ---
        # # Très performants, souvent meilleurs que des modèles 10x plus gros d'il y a 2 ans.
        # 'Qwen/Qwen1.5-0.5B',            # 0.5B : Incroyablement léger et surprenant
        # # 'Qwen/Qwen1.5-1.8B',          # 1.8B : Le test ultime pour votre RAM (limite haute)

        # # --- FAMILLE LLAMA (Architecture moderne) ---
        # # Le standard actuel, très efficace.
        # 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', # 1.1B : Excellent compromis taille/performance
        # # 'meta-llama/Meta-Llama-3-8B', # 8B : Le nouveau roi (trop gros ici)

        # # --- FAMILLE DEEPSEEK (Coding/Math specialists) ---
        # # Leurs modèles "Coder" existent en petite taille et sont très logiques
        # 'deepseek-ai/deepseek-coder-1.3b-base', # 1.3B : Excellent raisonnement
        # # 'deepseek-ai/deepseek-llm-7b-base', # 7B (trop gros ici)

        # # --- FAMILLE OPT (Facebook/Meta) ---
        # # Architecture similaire à GPT-2 mais entraînement différent
        # 'facebook/opt-125m',  # Très rapide, idéal pour tester
        # 'facebook/opt-350m',            # 350M : Pour comparer avec gpt2-medium
        # 'facebook/opt-1.3b',            # 1.3B : Concurrent direct de gpt2-xl et TinyLlama
        # # 'facebook/opt-2.7b',  # Le plus précis (celui qui a fait planter mon PC avant)

        # # --- FAMILLE MICROSOFT (Phi) ---
        # # Entraînés sur des données de très haute qualité (livres, code)
        # 'microsoft/phi-1_5',            # 1.3B : Très logique et cohérent
        # # 'microsoft/phi-2',            # 2.7B : Risque de crash (comme OPT-2.7B)

        # # --- FAMILLE ELEUTHER AI (Open Science) ---
        # # Modèles très utilisés dans la recherche
        # 'EleutherAI/gpt-neo-1.3B',      # 1.3B : L'alternative open-source historique à GPT-3
        # 'EleutherAI/pythia-1.4b',       # 1.4B : Très clean, bon pour l'analyse scientifique
        
        # # --- FAMILLE BLOOM (Multilingue) ---
        # # Si jamais vous voulez tester un peu de français plus tard
        # 'bigscience/bloom-560m',        # 560M : Petit mais polyvalent

        # # --- FAMILLE MISTRAL (fr) ---
        # 'mistralai/Mistral-7B-v0.1',  # 7B : Le standard actuel (trop gros ici)
    ]

    # Liste des juges pour la cohérence
    coherence_models = [
        'facebook/opt-125m',  # Très rapide, idéal pour tester
        # # # ###########'facebook/opt-350m',  # Bon compromis
        'facebook/opt-1.3b', # Plus précis mais demande ~6Go VRAM/RAM
        'facebook/opt-2.7b',  # Le plus précis (celui qui a fait planter mon PC avant)

        # # --- FAMILLE GPT-2 (Les classiques) ---
        'gpt2',          # ~124M params (Très rapide)
        'gpt2-medium',   # ~355M params
        'gpt2-large',    # ~774M params
        'gpt2-xl',        # ~1.5B params (Lourd)
    ]

    # Paramètres globaux
    dataset_name = 'wikitext' # wikitext | cc_news | bookcorpus
    dataset_config = 'wikitext-103-raw-v1' # wikitext-103-raw-v1 | plain_text | plain_text
    dataset_split = 'test' # test | train | train
    num_prefixes = 100         # Nombre d'exemples à générer (100 est standard utilisé pour 'gpt2...'~124M-~1.5B params sinon 5 )
    decoding_len = 256        # Longueur du texte généré (256 est standard utilisé pour 'gpt2...'~124M-~1.5B  params  sinon 16)
    
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
    
    # --- LOG DES ERREURS ---
    # Liste pour stocker le détail précis des échecs
    errors_log = [] 

    for model_name in gen_models:
        safe_model_name = model_name.replace('/', '-')              
        
        for strat in baselines:
            run_count += 1
            iter_start_time = time.time()
            config_name = f"{model_name} | {strat['name']}"
            
            print(f"\n{'='*80}")
            print(f"Configuration {run_count}/{total_runs} : {config_name}")
            print(f"{'='*80}")

            # --- A. GÉNÉRATION ---
            # Construction du nom de fichier attendu (format standard du projet)
            # Ex: wikitext_greedy_gpt2-xl_256.jsonl
            # Note: Le script generate.py construit souvent le nom lui-même, mais on doit le deviner pour l'étape suivante.
            filename_base = f'{dataset_name}_{strat["file_suffix"]}_{safe_model_name}_{decoding_len}'
            jsonl_output_path = f'{output_dir}/{filename_base}.jsonl'

            # Si le fichier existe déjà, on peut sauter l'étape (pratique si le script plante et qu'on relance)
            if os.path.exists(jsonl_output_path):
                print(f" Fichier existant trouvé, on passe la génération : {jsonl_output_path}")
            else:
                print(f" 1. Génération du texte ({strat['name']})...")
                gen_cmd = [
                    python_exe, 'open_text_gen/generate_baselines.py',
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
                    # On enregistre l'erreur et on passe
                    errors_log.append({'config': config_name, 'step': 'GÉNÉRATION', 'details': f'Code erreur: {e.returncode}'})
                    continue

            # Vérification que le fichier a bien été créé (parfois generate.py change légèrement le nom)
            # Si le fichier exact n'existe pas, on essaie de le trouver avec glob ou on avertit.
            if not os.path.exists(jsonl_output_path):
                print(f"\033[33m[ATTENTION] Impossible de trouver le fichier généré attendu :\n{jsonl_output_path}\nPassage aux évaluations suivantes impossible pour cette config.\033[0m")
                errors_log.append({'config': config_name, 'step': 'FICHIER', 'details': 'Fichier JSONL introuvable'})
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
                    errors_log.append({'config': config_name, 'step': 'COHÉRENCE', 'details': f'Juge {coh_model} échoué'})

            # --- C. ÉVALUATION (DIVERSITÉ & MAUVE) ---
            print(" 3. Diversité, MAUVE & Longueur...")
            # MODIFICATION ICI : Ajout de la longueur dans le nom du fichier de résultat
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
    # 3. RAPPORT FINAL (DICO D'ERREURS)
    # =========================================================================
    total_time = time.time() - global_start_time
    print("\n" + "="*80)
    print(f"RAPPORT FINAL ({format_timedelta(total_time)})")
    print("="*80)
    
    if len(errors_log) == 0:
        print(f"\033[42m SUCCÈS TOTAL \033[0m : 0 erreurs détectées.")
    else:
        print(f"\033[41m {len(errors_log)} ERREURS DÉTECTÉES : \033[0m")
        print("-" * 80)
        # Affichage tabulaire propre
        print(f"{'CONFIGURATION':<40} | {'ÉTAPE':<12} | {'DÉTAIL'}")
        print("-" * 80)
        for err in errors_log:
            # Coloration rouge pour l'étape
            print(f"{err['config']:<40} | \033[31m{err['step']:<12}\033[0m | {err['details']}")
    print("="*80)

if __name__ == '__main__':
    main()