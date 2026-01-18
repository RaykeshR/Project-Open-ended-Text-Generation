import os
import subprocess
import time
import sys
import datetime
import glob

# --- CONFIGURATION PRINCIPALE ---
MODEL = "gpt2-xl"
MODEL_SAFE = MODEL.replace("/", "-")

# Datasets à traiter
DATASETS = ["wikitext", "cc_news", "bookcorpus"]

# Dossier racine
BASE_DIR = "open_text_gen"

# Modèles juges pour la cohérence
COHERENCE_JUDGES = [
    "facebook/opt-125m",   # Rapide
    # "facebook/opt-1.3b", # Moyen
    "facebook/opt-2.7b",   # Précis (Lourd)

    # # # # --- FAMILLE GPT-2 (Les classiques) ---
    # 'gpt2',          # ~124M params (Très rapide)
    # 'gpt2-medium',   # ~355M params
    # 'gpt2-large',    # ~774M params
    # 'gpt2-xl',       # ~1.5B params (Lourd)
]

# ---> PARAMÈTRES OPTIMAUX (Doivent correspondre à analysis11.py)
# C'est ici qu'on définit k=5 pour cc_news spécifiquement
OPTIMAL_PARAMS = {
    "wikitext": {
        "p": 0.95,
        "contrastive": {"k": 10, "alpha": 0.6},
        "epsilon":     {"k": 10, "alpha": 0.6}
    },
    "bookcorpus": {
        "p": 0.95,
        "contrastive": {"k": 10, "alpha": 0.6},
        "epsilon":     {"k": 10, "alpha": 0.6}
    },
    "cc_news": {
        "p": 0.95,
        "contrastive": {"k": 10, "alpha": 0.6},
        "epsilon":     {"k": 5, "alpha": 0.6}  # <---
    }
}

# --- CONFIGURATION DATASETS ---
DATASET_INFO = {
    "wikitext": {
        "config": "wikitext-103-raw-v1",
        "split": "test",
        "len": 256,
        "samples": 100
    },
    "cc_news": {
        "config": "plain_text",
        "split": "train",
        "len": 32, 
        "samples": 50
    },
    "bookcorpus": {
        "config": "plain_text",
        "split": "train",
        "len": 32,
        "samples": 50
    }
}

def format_timedelta(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def run_command(cmd, step_name):
    """Exécute une commande shell et gère les erreurs."""
    print(f"   [RUN] {step_name}...")
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"   \033[91m[ERREUR] Échec de {step_name} (Code {e.returncode})\033[0m")
        print(f"   Commande: {cmd}")
        return False

def main():
    start_time = time.time()
    python_exe = sys.executable
    
    # Liste pour collecter les chemins des dossiers pour la perplexité finale
    folders_to_evaluate_perplexity = set()

    # On construit la liste des tâches
    tasks = []

    for dataset in DATASETS:
        info = DATASET_INFO[dataset]
        params = OPTIMAL_PARAMS[dataset]
        
        # Définition des dossiers
        if dataset in ["cc_news", "bookcorpus"]:
            dir_standard = os.path.join(BASE_DIR, f"{dataset}_grid_search")
            dir_epsilon = os.path.join(BASE_DIR, f"{dataset}_epsilon_grid_search")
        else:
            dir_standard = os.path.join(BASE_DIR, f"{dataset}_grid_search")
            dir_epsilon = os.path.join(BASE_DIR, f"{dataset}_epsilon_grid_search_256")

        os.makedirs(dir_standard, exist_ok=True)
        os.makedirs(dir_epsilon, exist_ok=True)
        
        folders_to_evaluate_perplexity.add(dir_standard)
        folders_to_evaluate_perplexity.add(dir_epsilon)

        # 1. GREEDY
        tasks.append({
            "name": f"{dataset} - Greedy",
            "output_dir": dir_standard,
            "expected_file": f"{dataset}_greedy_{MODEL_SAFE}_{info['len']}.jsonl",
            "gen_cmd": f'"{python_exe}" open_text_gen/generate_baselines.py --model_name {MODEL} --dataset_name {dataset} --dataset_config {info["config"]} --dataset_split {info["split"]} --decoding_strategy greedy --decoding_len {info["len"]} --num_prefixes {info["samples"]} --output_dir {dir_standard}'
        })

        # 2. NUCLEUS
        p_val = params['p']
        tasks.append({
            "name": f"{dataset} - Nucleus (p={p_val})",
            "output_dir": dir_standard,
            "expected_file": f"{dataset}_p-{p_val}_{MODEL_SAFE}_{info['len']}.jsonl",
            "gen_cmd": f'"{python_exe}" open_text_gen/generate_baselines.py --model_name {MODEL} --dataset_name {dataset} --dataset_config {info["config"]} --dataset_split {info["split"]} --decoding_strategy nucleus --probs {p_val} --decoding_len {info["len"]} --num_prefixes {info["samples"]} --output_dir {dir_standard}'
        })

        # 3. TYPICAL
        tasks.append({
            "name": f"{dataset} - Typical (p={p_val})",
            "output_dir": dir_standard,
            "expected_file": f"{dataset}_typical-{p_val}_{MODEL_SAFE}_{info['len']}.jsonl",
            "gen_cmd": f'"{python_exe}" open_text_gen/generate_baselines.py --model_name {MODEL} --dataset_name {dataset} --dataset_config {info["config"]} --dataset_split {info["split"]} --decoding_strategy typical --probs {p_val} --decoding_len {info["len"]} --num_prefixes {info["samples"]} --output_dir {dir_standard}'
        })

        # 4. CONTRASTIVE
        ck = params['contrastive']['k']
        ca = params['contrastive']['alpha']
        tasks.append({
            "name": f"{dataset} - Contrastive (k={ck}, a={ca})",
            "output_dir": dir_standard,
            "expected_file": f"{dataset}_k{ck}_a{ca}_e0.0_{MODEL_SAFE}.jsonl",
            "gen_cmd": f'"{python_exe}" open_text_gen/generate.py --model_name {MODEL} --dataset_name {dataset} --dataset_config {info["config"]} --dataset_split {info["split"]} --k {ck} --alpha {ca} --decoding_len {info["len"]} --num_prefixes {info["samples"]} --output_dir {dir_standard}'
        })

        # 5. EPSILON
        ek = params['epsilon']['k']
        ea = params['epsilon']['alpha']
        tasks.append({
            "name": f"{dataset} - Epsilon (k={ek}, a={ea})",
            "output_dir": dir_epsilon,
            "expected_file": f"{dataset}_epsilon_k{ek}_alpha{ea}_{MODEL_SAFE}.jsonl",
            "gen_cmd": f'"{python_exe}" open_text_gen/generate_epsilon.py --model_name {MODEL} --dataset_name {dataset} --dataset_config {info["config"]} --dataset_split {info["split"]} --k {ek} --alpha {ea} --decoding_len {info["len"]} --num_prefixes {info["samples"]} --output_dir {dir_epsilon}'
        })

    print(f"\n{'='*60}")
    print(f"DÉMARRAGE DU BENCHMARK COMPLET ({len(tasks)} tâches)")
    print(f"{'='*60}\n")

    for i, task in enumerate(tasks):
        print(f"➤ Tâche {i+1}/{len(tasks)} : \033[1m{task['name']}\033[0m")
        file_path = os.path.join(task['output_dir'], task['expected_file'])
        
        # --- 1. GÉNÉRATION ---
        if os.path.exists(file_path):
            print(f"   [INFO] Fichier déjà existant : {task['expected_file']}")
        else:
            # Recherche floue
            possible = glob.glob(os.path.join(task['output_dir'], f"*{task['name'].split()[0]}*"))            
            print(f"   [INFO] Génération de : {task['expected_file']}")
            success = run_command(task['gen_cmd'], "Génération du texte")
            if not success:
                print(f"   [SKIP] Impossible d'évaluer car la génération a échoué.")
                continue

        # Vérification existence fichier
        if not os.path.exists(file_path):
            # Tentative de rattrapage sur noms alternatifs
            found = False
            # On récupère les params actuels pour le rattrapage
            target_params = ""
            if "contrastive" in task['name'].lower():
                ck = OPTIMAL_PARAMS[task['name'].split()[0]]['contrastive']['k']
                target_params = f"k{ck}"
            elif "epsilon" in task['name'].lower():
                ek = OPTIMAL_PARAMS[task['name'].split()[0]]['epsilon']['k']
                target_params = f"k{ek}"

            for f in os.listdir(task['output_dir']):
                if f.endswith(".jsonl") and MODEL_SAFE in f:
                    if "contrastive" in task['name'].lower() and ("contrastive" in f or (target_params in f and "e0.0" in f)):
                         file_path = os.path.join(task['output_dir'], f)
                         print(f"   [FIX] Utilisation du fichier alternatif : {f}")
                         found = True
                         break
                    if "epsilon" in task['name'].lower() and "epsilon" in f and target_params in f:
                         file_path = os.path.join(task['output_dir'], f)
                         print(f"   [FIX] Utilisation du fichier alternatif : {f}")
                         found = True
                         break
            if not found and not os.path.exists(file_path):
                print(f"   [ERREUR] Fichier introuvable.")
                continue

        # --- 2. ÉVALUATIONS ---
        
        # Diversity / MAUVE
        res_div = file_path.replace(".jsonl", "_diversity_mauve_gen_length_result.json")
        res_div_alt = file_path.replace(".jsonl", "._diversity_mauve_gen_length_result.json")
        
        if not (os.path.exists(res_div) or os.path.exists(res_div_alt)):
            run_command(
                f'"{python_exe}" open_text_gen/measure_diversity_mauve_gen_length.py --test_path "{file_path}"',
                "Mesure Diversity/MAUVE"
            )
        else:
            print("   [SKIP] Diversity/MAUVE déjà calculé.")

        # SimCSE
        res_sim = file_path.replace(".jsonl", "_simcse_result.json")
        if not os.path.exists(res_sim):
            run_command(
                f'"{python_exe}" open_text_gen/compute_simcse.py --test_path "{file_path}"',
                "Mesure SimCSE"
            )
        else:
            print("   [SKIP] SimCSE déjà calculé.")

        # Cohérence (Likelihood)
        for judge in COHERENCE_JUDGES:
            judge_safe = judge.split("/")[-1]
            res_coh = file_path.replace(".jsonl", f"._{judge_safe}_coherence_result.json")
            
            if not os.path.exists(res_coh):
                run_command(
                    f'"{python_exe}" open_text_gen/compute_coherence.py --test_path "{file_path}" --opt_model_name {judge}',
                    f"Mesure Cohérence ({judge_safe})"
                )
            else:
                print(f"   [SKIP] Cohérence ({judge_safe}) déjà calculée.")

    print(f"\n{'='*60}")
    print(f"CALCUL DE LA PERPLEXITÉ (Optimisé)")
    print(f"{'='*60}\n")
    
    for folder in folders_to_evaluate_perplexity:
        if not os.path.exists(folder): continue
        
        files_in_folder = [f for f in os.listdir(folder) if f.endswith(".jsonl")]
        if not files_in_folder:
            print(f"   [INFO] Dossier vide : {folder}")
            continue

        # Vérifier si TOUS les fichiers ont déjà leur résultat de perplexité
        all_done = True
        for f in files_in_folder:
            ppl_res = os.path.join(folder, f.replace(".jsonl", "_perplexity_result.json"))
            if not os.path.exists(ppl_res):
                all_done = False
                break
        
        if all_done:
            print(f"➤ [SKIP] Perplexité déjà calculée pour tous les fichiers de : {folder}")
        else:
            print(f"➤ [RUN] Calcul Perplexité pour le dossier : {folder}")
            run_command(
                f'"{python_exe}" open_text_gen/measure_perplexity.py --folder "{folder}" --model_name gpt2-xl',
                "Mesure Perplexité (Batch)"
            )

    total_time = time.time() - start_time
    print(f"\n\n\033[42m TERMINÉ \033[0m Durée totale : {format_timedelta(total_time)}")

if __name__ == "__main__":
    main()