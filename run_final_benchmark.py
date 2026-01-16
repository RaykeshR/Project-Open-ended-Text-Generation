import os, subprocess, time

# --- CONFIGURATION COMMUNE ---
MODEL = "gpt2-xl"
GEN_LENGTH = 256      # Longueur standard pour la comparaison
NUM_SAMPLES = 100     # Nombre de générations (ajustez selon votre temps/GPU, ex: 1000 pour MAUVE précis)
BATCH_SIZE = 16       # Ajustez selon votre VRAM (ex: 8, 16, 32)

# Liste des datasets
DATASETS = ["wikitext", "cc_news", "bookcorpus"] # Vérifiez les noms exacts attendus par vos scripts

# Commandes pour chaque méthode
# Assurez-vous que les arguments correspondent exactement à ceux de vos scripts generate_*.py
commands = []

for dataset in DATASETS:
    print(f"--- Préparation des commandes pour {dataset} ---")
    
    # 1. Greedy Search (Baseline)
    commands.append(
        f"python open_text_gen/generate_baselines.py --model_name {MODEL} --dataset_name {dataset} "
        f"--method greedy --max_length {GEN_LENGTH} --num_return_sequences {NUM_SAMPLES} --batch_size {BATCH_SIZE}"
    )
    
    # 2. Nucleus Sampling (p=0.95)
    commands.append(
        f"python open_text_gen/generate_baselines.py --model_name {MODEL} --dataset_name {dataset} "
        f"--method nucleus --p 0.95 --max_length {GEN_LENGTH} --num_return_sequences {NUM_SAMPLES} --batch_size {BATCH_SIZE}"
    )
    
    # 3. Typical Sampling (p=0.95 ou tau=0.95 selon votre implémentation)
    commands.append(
        f"python open_text_gen/generate_baselines.py --model_name {MODEL} --dataset_name {dataset} "
        f"--method typical --p 0.95 --max_length {GEN_LENGTH} --num_return_sequences {NUM_SAMPLES} --batch_size {BATCH_SIZE}"
    )
    
    # 4. Contrastive Search (Meilleurs hyperparamètres : k=10, alpha=0.6)
    commands.append(
        f"python open_text_gen/generate_epsilon.py --model_name {MODEL} --dataset_name {dataset} "
        f"--k 10 --alpha 0.6 --max_length {GEN_LENGTH} --num_return_sequences {NUM_SAMPLES} --batch_size {BATCH_SIZE}"
    )

# --- EXÉCUTION ---
t0 = time.time()
for i, cmd in enumerate(commands):
    try:
        print(f"=> Étape {i+1}/{len(commands)} | {int(time.time()-t0)}s écoulés (Fin estimée dans ~{int((time.time()-t0)/(i if i>0 else 1)*(len(commands)-i))}s) | Dataset: {cmd.split('--dataset_name ')[1].split()[0]} | Méthode: {cmd.split('--method ')[1].split()[0] if '--method' in cmd else 'contrastive'}")
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de : {cmd}")
        # Continue ou break selon votre préférence