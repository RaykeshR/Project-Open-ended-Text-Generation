import os
import subprocess
import sys
import itertools

def main():
    # =========================================================================
    # 1. CONFIGURATION DE LA GRID SEARCH
    # =========================================================================
    
    # Liste des modèles de génération (du plus léger au plus lourd)
    # Vous pouvez commenter ceux que vous ne voulez pas tester (ex: gpt2-xl si manque de RAM)
    gen_models = [
        'gpt2',          # ~124M params (Très rapide)
        'gpt2-medium',   # ~355M params
        'gpt2-large',    # ~774M params
        # 'gpt2-xl'        # ~1.5B params (Lourd)
    ]

    # Liste des modèles "juges" pour la cohérence (du plus léger au plus lourd)
    # Utiliser plusieurs juges permet de vérifier si le score est robuste.
    coherence_models = [
        'facebook/opt-125m',  # Très rapide, idéal pour tester
        'facebook/opt-350m',  # Bon compromis
        # 'facebook/opt-1.3b', # Plus précis mais demande ~6Go VRAM/RAM
        # 'facebook/opt-2.7b'  # Le plus précis (celui qui a fait planter mon PC avant)
    ]

    # Hyperparamètres à tester
    ks = [5, 10]              # Taille du beam (beam_width)
    alphas = [0.4, 0.6, 0.8]  # Pénalité de dégénérescence
    epsilons = [0.0]          # Seuil de probabilité (0.0 = désactivé)

    # Paramètres globaux
    dataset_name = 'wikitext'
    num_prefixes = 100        # Nombre d'exemples à générer (100 est standard)
    decoding_len = 256        # Longueur du texte généré

    # =========================================================================
    # 2. EXÉCUTION
    # =========================================================================
    
    python_exe = sys.executable
    output_dir = f'open_text_gen/{dataset_name}_grid_search'
    os.makedirs(output_dir, exist_ok=True)

    # On utilise itertools.product pour faire toutes les combinaisons possibles
    # (Modèle x K x Alpha x Epsilon)
    combinations = list(itertools.product(gen_models, ks, alphas, epsilons))
    
    print(f" Démarrage de la Grid Search : {len(combinations)} configurations à tester.")
    print(f" Résultats sauvegardés dans : {output_dir}")

    for model_name, k, alpha, epsilon in combinations:
        print(f"\n{'='*60}")
        print(f"Configuration : Modèle={model_name} | k={k} | alpha={alpha} | epsilon={epsilon}")
        print(f"{'='*60}")

        # --- A. GÉNÉRATION ---
        # Nom du fichier qui sera créé par generate.py
        # DOIT correspondre au format dans generate.py : {dataset}_k{k}_a{alpha}_e{epsilon}_{model}.jsonl
        filename_base = f'{dataset_name}_k{k}_a{alpha}_e{epsilon}_{model_name}'
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
                print(f" Erreur lors de la génération pour {model_name}. On passe à la suite.")
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
            except subprocess.CalledProcessError:
                print(f" Erreur cohérence avec {coh_model}")

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

    print("\n Grid Search terminée ! Vous pouvez analyser les fichiers JSON dans le dossier de sortie.")

if __name__ == '__main__':
    main()