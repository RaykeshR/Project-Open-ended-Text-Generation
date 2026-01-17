import os
import subprocess
import time

# --- CONFIGURATION ---
MODEL = "gpt2-xl"
GEN_LENGTH = 256      
NUM_SAMPLES = 100     
BASE_DIR = "open_text_gen"

# Datasets
DATASETS = ["wikitext", "cc_news", "bookcorpus"] 

# Paramètres optimaux retenus
BEST_K = 10
BEST_ALPHA = 0.6
BEST_P = 0.95

# --- MAPPING CONFIGURATIONS DATASETS ---
DATASET_INFO = {
    "wikitext": {
        "config": "wikitext-103-raw-v1",
        "split": "test"
    },
    "cc_news": {
        "config": "plain_text", # ou None selon votre wrapper
        "split": "train"
    },
    "bookcorpus": {
        "config": "plain_text",
        "split": "train"
    }
}

# --- FONCTION DE VÉRIFICATION ---
def file_exists_pattern(directory, pattern_keywords):
    """
    Vérifie si un fichier contenant tous les mots-clés existe dans le dossier.
    """
    if not os.path.exists(directory):
        return False
    
    files = os.listdir(directory)
    for f in files:
        if all(kw in f for kw in pattern_keywords) and f.endswith(".jsonl"):
            print(f" Fichier trouvé : {f}")
            return True
    return False

# --- LISTE DES TÂCHES ---
tasks = []

for dataset in DATASETS:
    # Récupération des paramètres spécifiques au dataset
    ds_config = DATASET_INFO[dataset]["config"]
    ds_split = DATASET_INFO[dataset]["split"]

    if dataset in ["cc_news", "bookcorpus"]:
        current_gen_length = 32
        current_num_samples = 50
        dir_epsilon = os.path.join(BASE_DIR, f"{dataset}_epsilon_grid_search")
        dir_standard = os.path.join(BASE_DIR, f"{dataset}_grid_search")
    else:
        current_gen_length = GEN_LENGTH
        current_num_samples = NUM_SAMPLES
        # 1. Dossier standard (pour Greedy, Nucleus, Typical, Contrastive)
        dir_standard = os.path.join(BASE_DIR, f"{dataset}_grid_search")
        
        # 2. Dossier Grid Search (pour Epsilon Sampling)
        dir_epsilon = os.path.join(BASE_DIR, f"{dataset}_epsilon_grid_search_256")
    
    # Création des dossiers
    os.makedirs(dir_standard, exist_ok=True)
    os.makedirs(dir_epsilon, exist_ok=True)

    # --- MÉTHODE 1 : GREEDY ---
    tasks.append({
        "dataset": dataset,
        "method": "Greedy",
        "output_dir": dir_standard,
        "check_keywords": ["greedy", str(current_gen_length), MODEL],
        "cmd": (
            f"python open_text_gen/generate_baselines.py --model_name {MODEL} "
            f"--dataset_name {dataset} --dataset_config {ds_config} --dataset_split {ds_split} "
            f"--decoding_strategy greedy --decoding_len {current_gen_length} --num_prefixes {current_num_samples} "
            f"--output_dir {dir_standard}"
        )
    })

    # --- MÉTHODE 2 : NUCLEUS ---
    tasks.append({
        "dataset": dataset,
        "method": f"Nucleus (p={BEST_P})",
        "output_dir": dir_standard,
        "check_keywords": [f"p-{BEST_P}", str(current_gen_length), MODEL],
        "cmd": (
            f"python open_text_gen/generate_baselines.py --model_name {MODEL} "
            f"--dataset_name {dataset} --dataset_config {ds_config} --dataset_split {ds_split} "
            f"--decoding_strategy nucleus --probs {BEST_P} --decoding_len {current_gen_length} --num_prefixes {current_num_samples} "
            f"--output_dir {dir_standard}"
        )
    })

    # --- MÉTHODE 3 : TYPICAL ---
    tasks.append({
        "dataset": dataset,
        "method": f"Typical (p={BEST_P})",
        "output_dir": dir_standard,
        "check_keywords": [f"typical-{BEST_P}", str(current_gen_length), MODEL],
        "cmd": (
            f"python open_text_gen/generate_baselines.py --model_name {MODEL} "
            f"--dataset_name {dataset} --dataset_config {ds_config} --dataset_split {ds_split} "
            f"--decoding_strategy typical --probs {BEST_P} --decoding_len {current_gen_length} --num_prefixes {current_num_samples} "
            f"--output_dir {dir_standard}"
        )
    })

    # --- MÉTHODE 4 : EPSILON SAMPLING (Grid Search Winner) ---
    tasks.append({
        "dataset": dataset,
        "method": f"Epsilon (k={BEST_K}, α={BEST_ALPHA})",
        "output_dir": dir_epsilon,
        "check_keywords": [f"k{BEST_K}", f"alpha{BEST_ALPHA}", "epsilon", MODEL],
        "cmd": (
            f"python open_text_gen/generate_epsilon.py --model_name {MODEL} "
            f"--dataset_name {dataset} --dataset_config {ds_config} --dataset_split {ds_split} "
            f"--k {BEST_K} --alpha {BEST_ALPHA} --decoding_len {current_gen_length} --num_prefixes {current_num_samples} "
            f"--output_dir {dir_epsilon}"
        )
    })

    # --- MÉTHODE 5 : CONTRASTIVE SEARCH (Standard) ---
    tasks.append({
        "dataset": dataset,
        "method": f"Contrastive (k={BEST_K}, α={BEST_ALPHA})",
        "output_dir": dir_standard,
        "check_keywords": [f"k{BEST_K}", f"a{BEST_ALPHA}", MODEL],
        "cmd": (
            f"python open_text_gen/generate.py --model_name {MODEL} "
            f"--dataset_name {dataset} --dataset_config {ds_config} --dataset_split {ds_split} "
            f"--k {BEST_K} --alpha {BEST_ALPHA} --decoding_len {current_gen_length} --num_prefixes {current_num_samples} "
            f"--output_dir {dir_standard}"
        )
    })

# --- EXÉCUTION ---
t0 = time.time()
print(f" Démarrage du Benchmark 5 Méthodes ({len(tasks)} tâches)\n")

for i, task in enumerate(tasks):
    print(f" Étape {i+1}/{len(tasks)} | {task['dataset']} | {task['method']}")
    
    # print((task['output_dir'], task['check_keywords']))
    # Vérification
    if file_exists_pattern(task['output_dir'], task['check_keywords']):
        print(f" Déjà fait. On passe.")
        continue
    # Lancement
    print(f"  Génération en cours...")
    print(f"  CMD: {task['cmd']}") # Affichage pour debug
    try:
        # subprocess.run(task['cmd'], shell=True, check=True, capture_output=True, text=True)
        subprocess.run(task['cmd'], shell=True, check=True)
        print(f" Succès.")
    except subprocess.CalledProcessError as e:
        print(f" ERREUR CRITIQUE. Commande échouée: {task['cmd']} ERROR : {e}")

print("\n Benchmark terminé.")