import os
import subprocess
import sys
import itertools
import time
import datetime

def format_timedelta(seconds):
    """Formate les secondes en H:M:S"""
    return str(datetime.timedelta(seconds=int(seconds)))

def main():
    # =========================================================================
    # 1. CONFIGURATION DE LA GRID SEARCH
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

    # Liste des modèles "juges" pour la cohérence (du plus léger au plus lourd)
    # Utiliser plusieurs juges permet de vérifier si le score est robuste.
    coherence_models = [
        'facebook/opt-125m',  # Très rapide, idéal pour tester
        # # # ###########'facebook/opt-350m',  # Bon compromis
        # 'facebook/opt-1.3b', # Plus précis mais demande ~6Go VRAM/RAM
        # 'facebook/opt-2.7b',  # Le plus précis (celui qui a fait planter mon PC avant)

        # # # --- FAMILLE GPT-2 (Les classiques) ---
        # 'gpt2',          # ~124M params (Très rapide)
        # 'gpt2-medium',   # ~355M params
        # 'gpt2-large',    # ~774M params
        # 'gpt2-xl',        # ~1.5B params (Lourd)
    ]

    # Hyperparamètres à tester
    ks = [5, 10]              # Taille du beam (beam_width)              |[5, 10][5    ]
    alphas = [0.4, 0.6, 0.8]  # Pénalité de dégénérescence               |[0.4, 0.6, 0.8][     0.6     ]
    epsilons = [0.0]          # Seuil de probabilité (0.0 = désactivé)   |[0.0]

    # Paramètres globaux
    dataset_name = 'wikitext' # wikitext | cc_news | bookcorpus
    dataset_config = 'wikitext-103-raw-v1' # wikitext-103-raw-v1 | plain_text | plain_text
    dataset_split = 'test' # test | train | train
    num_prefixes = 100         # Nombre d'exemples à générer (100 est standard utilisé pour 'gpt2...'~124M-~1.5B params sinon 5 )
    decoding_len = 256        # Longueur du texte généré (256 est standard utilisé pour 'gpt2...'~124M-~1.5B  params  sinon 16)

    # =========================================================================
    # 2. EXÉCUTION
    # =========================================================================
    
    python_exe = sys.executable
    output_dir = f'open_text_gen/{dataset_name}_grid_search'
    os.makedirs(output_dir, exist_ok=True)

    # On utilise itertools.product pour faire toutes les combinaisons possibles
    # (Modèle x K x Alpha x Epsilon)
    combinations = list(itertools.product(gen_models, ks, alphas, epsilons))
    total_combinations = len(combinations)
    
    print(f" Démarrage de la Grid Search : {total_combinations} configurations à tester.")
    print(f" Résultats sauvegardés dans : {output_dir}")

    # Initialisation des chronomètres globaux
    global_start_time = time.time()
    iteration_times = []

    for idx, (model_name, k, alpha, epsilon) in enumerate(combinations):
        iter_start_time = time.time()
        current_iter = idx + 1
        
        print(f"\n{'='*60}")
        print(f"Configuration {current_iter}/{total_combinations} : Modèle={model_name} | k={k} | alpha={alpha} | epsilon={epsilon}")
        print(f"{'='*60}")

        # --- A. GÉNÉRATION ---
        # Nom du fichier qui sera créé par generate.py
        # DOIT correspondre au format dans generate.py : {dataset}_k{k}_a{alpha}_e{epsilon}_{model}.jsonl
        safe_model_name = model_name.replace('/', '-')
        
        filename_base = f'{dataset_name}_k{k}_a{alpha}_e{epsilon}_{safe_model_name}'
        jsonl_output_path = f'{output_dir}/{filename_base}.jsonl'

        # Si le fichier existe déjà, on peut sauter l'étape (pratique si le script plante et qu'on relance)
        if os.path.exists(jsonl_output_path):
            print(f" Fichier existant trouvé, on passe la génération : {jsonl_output_path}")
        else:
            print(" Génération du texte...")
            gen_cmd = [
                python_exe, 'open_text_gen/generate.py',
                '--model_name', model_name,
                '--dataset_name', dataset_name,
                '--dataset_config', dataset_config,
                '--dataset_split', dataset_split,
                '--output_dir', output_dir,
                '--decoding_strategy', 'contrastive',
                '--decoding_len', str(decoding_len),
                '--num_prefixes', str(num_prefixes),
                # On passe des listes de taille 1 pour contrôler la boucle ici
                '--ks', str(k),
                '--alphas', str(alpha),
                '--epsilons', str(epsilon)
            ]
            try:
                subprocess.run(gen_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f" Erreur lors de la génération pour \033[31m{model_name}. On passe à la suite.")
                print(f"\n[ERREUR CRITIQUE] Le processus a crashé avec le code : {e.returncode}")
                print(f"Commande échouée : \x1b[2m{e.cmd}\033[0m")
                print(f"\x1b[31m\x1b[5m!!!\x1b[1m Erreur \x1b[101m: {e}\x1b[25m\033[0m")
                continue

        # --- B. ÉVALUATION (COHÉRENCE) ---
        # On boucle sur tous les modèles "juges"
        for coh_model in coherence_models:
            print(f" Mesure de la cohérence avec le juge : {coh_model}...")
            
            # compute_coherence.py génère son propre nom de sortie basé sur le jsonl d'entrée
            # Il ajoute généralement "_{opt_model}_coherence_result.json"
            coh_cmd = [
                python_exe, 'open_text_gen/compute_coherence.py',
                '--opt_model_name', coh_model,
                '--test_path', jsonl_output_path
            ]
            try:
                subprocess.run(coh_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f" Erreur cohérence avec \033[31m{coh_model}")
                print(f"\n[ERREUR CRITIQUE] Le processus a crashé avec le code : {e.returncode}")
                print(f"Commande échouée : {e.cmd}\033[0m")

        # --- C. ÉVALUATION (DIVERSITÉ & MAUVE) ---
        print(" Mesure de Diversité & MAUVE...")
        div_output_path = f'{output_dir}/{filename_base}_diversity_mauve_gen_length_result.json'
        
        div_cmd = [
            python_exe, 'open_text_gen/measure_diversity_mauve_gen_length.py',
            '--test_path', jsonl_output_path,
            '--result_path', div_output_path
        ]
        try:
            subprocess.run(div_cmd, check=True)
        except subprocess.CalledProcessError:
            print(" Erreur Diversité/MAUVE")

        # --- CALCULS DE TEMPS ---
        iter_end_time = time.time()
        iter_duration = iter_end_time - iter_start_time
        iteration_times.append(iter_duration)
        
        # Moyenne des itérations passées
        avg_duration = sum(iteration_times) / len(iteration_times)
        
        # Temps écoulé total
        total_elapsed = iter_end_time - global_start_time
        
        # Estimation du reste
        remaining_iters = total_combinations - current_iter
        estimated_remaining = avg_duration * remaining_iters
        
        print(f"\n Fin de l'itération {current_iter}/{total_combinations}")
        print(f"  \033[31m ➤ \033[0m Durée config actuelle : {format_timedelta(iter_duration)}")
        print(f"  \033[31m ➤ \033[0m Temps écoulé total    : {format_timedelta(total_elapsed)}")
        print(f"  \033[31m ➤ \033[0m Moyenne par config    : {format_timedelta(avg_duration)}")
        print(f"  \033[31m ➤ \033[0m TEMPS RESTANT ESTIMÉ  : {format_timedelta(estimated_remaining)}")

    print(f"\n Grid Search terminée en {format_timedelta(time.time() - global_start_time)} ! Vous pouvez analyser les fichiers JSON dans le dossier de sortie.")

if __name__ == '__main__':
    main()