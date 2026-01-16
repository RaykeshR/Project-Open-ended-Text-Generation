import os
import subprocess
import time

# --- CONFIGURATION COMMUNE ---
MODEL = "gpt2-xl"
GEN_LENGTH = 256      # Longueur de génération
NUM_SAMPLES = 100     # Nombre d'échantillons
BASE_DIR = "open_text_gen" # Dossier racine du projet

# Liste des datasets
DATASETS = ["wikitext", "cc_news", "bookcorpus"] 

commands = []

print(f"--- Préparation du Benchmark pour {MODEL} ---")

for dataset in DATASETS:
    # 1. Définition des dossiers de sortie cibles (pour matcher le script d'analyse)
    dir_baselines = os.path.join(BASE_DIR, dataset)
    dir_contrastive = os.path.join(BASE_DIR, f"{dataset}_epsilon_grid_search")
    
    # Création des dossiers si inexistants
    os.makedirs(dir_baselines, exist_ok=True)
    os.makedirs(dir_contrastive, exist_ok=True)
    
    print(f"[{dataset}] Baselines iront dans : {dir_baselines}")
    print(f"[{dataset}] Contrastive ira dans : {dir_contrastive}")

    # --- Commandes Baselines (Greedy, Nucleus, Typical) ---
    # Elles vont dans open_text_gen/dataset/
    
    # Greedy
    commands.append(
        f"python open_text_gen/generate_baselines.py --model_name {MODEL} --dataset_name {dataset} "
        f"--decoding_strategy greedy --decoding_len {GEN_LENGTH} --num_prefixes {NUM_SAMPLES} "
        f"--output_dir {dir_baselines}"
    )
    
    # Nucleus (p=0.95)
    commands.append(
        f"python open_text_gen/generate_baselines.py --model_name {MODEL} --dataset_name {dataset} "
        f"--decoding_strategy nucleus --probs 0.95 --decoding_len {GEN_LENGTH} --num_prefixes {NUM_SAMPLES} "
        f"--output_dir {dir_baselines}"
    )
    
    # Typical (p=0.95)
    commands.append(
        f"python open_text_gen/generate_baselines.py --model_name {MODEL} --dataset_name {dataset} "
        f"--decoding_strategy typical --probs 0.95 --decoding_len {GEN_LENGTH} --num_prefixes {NUM_SAMPLES} "
        f"--output_dir {dir_baselines}"
    )
    
    # --- Commande Contrastive (k=10, alpha=0.6) ---
    # Elle va dans open_text_gen/dataset_epsilon_grid_search/
    commands.append(
        f"python open_text_gen/generate_epsilon.py --model_name {MODEL} --dataset_name {dataset} "
        f"--k 10 --alpha 0.6 --decoding_len {GEN_LENGTH} --num_prefixes {NUM_SAMPLES} "
        f"--output_dir {dir_contrastive}"
    )

# --- EXÉCUTION ---
t0 = time.time()
print(f"\nLancement de {len(commands)} tâches de génération...\n")

for i, cmd in enumerate(commands):
    # Extraction infos pour affichage
    parts = cmd.split()
    ds = parts[parts.index("--dataset_name") + 1]
    
    if "--decoding_strategy" in parts:
        method = parts[parts.index("--decoding_strategy") + 1]
    else:
        method = "contrastive (k=10, α=0.6)"

    elapsed = int(time.time() - t0)
    avg_time = elapsed / i if i > 0 else 0
    eta = int(avg_time * (len(commands) - i))
    
    print(f" Étape {i+1}/{len(commands)} | {elapsed}s (Fin ~{eta}s) | {ds} -> {method}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f" Erreur critique sur l'étape {i+1}. Commande : {cmd}")