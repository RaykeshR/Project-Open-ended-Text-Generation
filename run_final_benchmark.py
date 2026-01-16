import os
import subprocess
import time

# --- CONFIGURATION COMMUNE ---
MODEL = "gpt2-xl"
GEN_LENGTH = 256      # Correspond à --decoding_len
NUM_SAMPLES = 100     # Correspond à --num_prefixes (nombre de prompts à traiter)

# Liste des datasets
DATASETS = ["wikitext", "cc_news", "bookcorpus"] 

# Commandes pour chaque méthode
commands = []

for dataset in DATASETS:
    # 1. Greedy Search
    # Note: --decoding_strategy remplace --method
    # Note: --decoding_len remplace --max_length
    # Note: --num_prefixes remplace --num_return_sequences
    commands.append(
        f"python open_text_gen/generate_baselines.py --model_name {MODEL} --dataset_name {dataset} "
        f"--decoding_strategy greedy --decoding_len {GEN_LENGTH} --num_prefixes {NUM_SAMPLES}"
    )
    
    # 2. Nucleus Sampling (p=0.95)
    # Note: L'argument pour p semble être --probs selon l'usage standard de ce repo, ou implicite via strategy
    commands.append(
        f"python open_text_gen/generate_baselines.py --model_name {MODEL} --dataset_name {dataset} "
        f"--decoding_strategy nucleus --probs 0.95 --decoding_len {GEN_LENGTH} --num_prefixes {NUM_SAMPLES}"
    )
    
    # 3. Typical Sampling (p=0.95)
    commands.append(
        f"python open_text_gen/generate_baselines.py --model_name {MODEL} --dataset_name {dataset} "
        f"--decoding_strategy typical --probs 0.95 --decoding_len {GEN_LENGTH} --num_prefixes {NUM_SAMPLES}"
    )
    
    # 4. Contrastive Search (k=10, alpha=0.6)
    # Note: generate_epsilon.py n'accepte pas --batch_size ou --num_return_sequences
    commands.append(
        f"python open_text_gen/generate_epsilon.py --model_name {MODEL} --dataset_name {dataset} "
        f"--k 10 --alpha 0.6 --decoding_len {GEN_LENGTH} --num_prefixes {NUM_SAMPLES}"
    )

# --- EXÉCUTION ---
t0 = time.time() # Top départ
print(f"Démarrage du benchmark sur {len(commands)} configurations...\n")

for i, cmd in enumerate(commands):
    # Parsing simple pour l'affichage
    parts = cmd.split()
    ds = "Unknown"
    method = "Unknown"
    
    if "--dataset_name" in parts:
        ds = parts[parts.index("--dataset_name") + 1]
    
    if "--decoding_strategy" in parts:
        method = parts[parts.index("--decoding_strategy") + 1]
    elif "generate_epsilon" in cmd:
        method = "contrastive (k=10, a=0.6)"

    elapsed = int(time.time() - t0)
    # Estimation du temps restant (évite la division par zéro)
    avg_time = elapsed / i if i > 0 else 0
    eta = int(avg_time * (len(commands) - i))
    
    print(f" Étape {i+1}/{len(commands)} | {elapsed}s écoulés (Fin ~{eta}s) |Dataset: {ds} | Méthode: {method}")
    
    try:
        # capture_output=False permet de voir les erreurs si elles persistent
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f" Erreur critique sur l'étape {i+1}. Commande : {cmd}")
        # On continue quand même pour essayer les autres datasets