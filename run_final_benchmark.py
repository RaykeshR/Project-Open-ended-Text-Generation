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
    if dataset in ["cc_news", "bookcorpus"]:
        GEN_LENGTH = 32      
        NUM_SAMPLES = 50 
        dir_epsilon = os.path.join(BASE_DIR, f"{dataset}_epsilon_grid_search")
        dir_standard = os.path.join(BASE_DIR, dataset)+"_grid_search"
    else: 
        # 1. D  ossier standard (pour Greedy, Nucleus, Typical, Contrastive)
        dir_standard = os.path.join(BASE_DIR, dataset)+"_grid_search"
        
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
        "check_keywords": ["greedy", str(GEN_LENGTH), MODEL],
        "cmd": (
            f"python open_text_gen/generate_baselines.py --model_name {MODEL} --dataset_name {dataset} "
            f"--decoding_strategy greedy --decoding_len {GEN_LENGTH} --num_prefixes {NUM_SAMPLES} "
            f"--output_dir {dir_standard}"
        )
    })

    # --- MÉTHODE 2 : NUCLEUS ---
    tasks.append({
        "dataset": dataset,
        "method": f"Nucleus (p={BEST_P})",
        "output_dir": dir_standard,
        "check_keywords": [f"p-{BEST_P}", str(GEN_LENGTH), MODEL],
        "cmd": (
            f"python open_text_gen/generate_baselines.py --model_name {MODEL} --dataset_name {dataset} "
            f"--decoding_strategy nucleus --probs {BEST_P} --decoding_len {GEN_LENGTH} --num_prefixes {NUM_SAMPLES} "
            f"--output_dir {dir_standard}"
        )
    })

    # --- MÉTHODE 3 : TYPICAL ---
    tasks.append({
        "dataset": dataset,
        "method": f"Typical (p={BEST_P})",
        "output_dir": dir_standard,
        "check_keywords": [f"typical-{BEST_P}", str(GEN_LENGTH), MODEL],
        "cmd": (
            f"python open_text_gen/generate_baselines.py --model_name {MODEL} --dataset_name {dataset} "
            f"--decoding_strategy typical --probs {BEST_P} --decoding_len {GEN_LENGTH} --num_prefixes {NUM_SAMPLES} "
            f"--output_dir {dir_standard}"
        )
    })

    # --- MÉTHODE 4 : EPSILON SAMPLING (Grid Search Winner) ---
    # Utilise generate_epsilon.py
    tasks.append({
        "dataset": dataset,
        "method": f"Epsilon (k={BEST_K}, α={BEST_ALPHA})",
        "output_dir": dir_epsilon,
        "check_keywords": [f"k{BEST_K}", f"alpha{BEST_ALPHA}", "epsilon", MODEL],
        "cmd": (
            f"python open_text_gen/generate_epsilon.py --model_name {MODEL} --dataset_name {dataset} "
            f"--k {BEST_K} --alpha {BEST_ALPHA} --decoding_len {GEN_LENGTH} --num_prefixes {NUM_SAMPLES} "
            f"--output_dir {dir_epsilon}"
        )
    })

    # --- MÉTHODE 5 : CONTRASTIVE SEARCH (Standard) ---
    # Utilise generate.py
    tasks.append({
        "dataset": dataset,
        "method": f"Contrastive (k={BEST_K}, α={BEST_ALPHA})",
        "output_dir": dir_standard, 
        "check_keywords": [f"k{BEST_K}", f"a{BEST_ALPHA}", MODEL], 
        # Note: Si generate.py ne met pas "contrastive" dans le nom, ajustez les keywords
        "cmd": (
            f"python open_text_gen/generate.py --model_name {MODEL} --dataset_name {dataset} "
            f"--k {BEST_K} --alpha {BEST_ALPHA} --decoding_len {GEN_LENGTH} --num_prefixes {NUM_SAMPLES} "
            f"--output_dir {dir_standard}"
        )
    })

# --- EXÉCUTION ---
t0 = time.time()
print(f" Démarrage du Benchmark 5 Méthodes ({len(tasks)} tâches)\n")

for i, task in enumerate(tasks):
    print(f" Étape {i+1}/{len(tasks)} | {task['dataset']} | {task['method']}")
    
    print((task['output_dir'], task['check_keywords']))
    # Vérification
    if file_exists_pattern(task['output_dir'], task['check_keywords']):
        print(f" Déjà fait. On passe.")
        continue
    # Lancement
    print(f"  Génération en cours...")
    try:
        print(subprocess.run(task['cmd'], shell=True, check=True, capture_output=True, text=True))
        print(f" Succès.")
    except subprocess.CalledProcessError as e:
        print(f" ERREUR CRITIQUE. Commande échouée: {task['cmd']} ERROR : {e}")

print("\n Benchmark terminé.")