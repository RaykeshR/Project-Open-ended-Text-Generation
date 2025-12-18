import os
import subprocess
import sys
import glob

def main():
    # ==========================================
    # CONFIGURATION
    # ==========================================
    # Dossier où sont les résultats Ollama
    results_dir = 'open_text_gen/ollama_results'
    Juge_model_name = 'facebook/opt-125m'
    python_exe = sys.executable
    # ==========================================

    # Extraction du nom court du juge pour le fichier (ex: 'opt-125m')
    try:
        judge_short_name = Juge_model_name.split('/')[1]
    except IndexError:
        judge_short_name = Juge_model_name

    # On cherche tous les fichiers .jsonl dans le dossier
    jsonl_files = glob.glob(f"{results_dir}/*.jsonl")
    
    if not jsonl_files:
        print(f"Aucun fichier .jsonl trouvé dans {results_dir}. Lancez generate_ollama.py d'abord !")
        return

    print(f" Fichiers trouvés à évaluer : {len(jsonl_files)}")
    print(f" Juge utilisé : {judge_short_name}")

    for file_path in jsonl_files:
        print(f"\n{'='*60}")
        print(f"Évaluation de : {os.path.basename(file_path)}")
        print(f"{'='*60}")
        
        # 1. COHÉRENCE
        # Le script compute_coherence.py génère LUI-MÊME le nom de sortie :
        # {nom_fichier}_{judge_short_name}_coherence_result.json
        # (Juge (Juge_model_name): OPT-125M ou celui de votre choix)
        coh_cmd = [
            python_exe, 'open_text_gen/compute_coherence.py',
            '--opt_model_name', Juge_model_name,
            '--test_path', file_path
        ]
        try:
            print(f"Calcul Cohérence (via {judge_short_name})...")
            subprocess.run(coh_cmd, check=True)
        except Exception as e:
            print(f"Erreur Cohérence: {e}")

        # 2. DIVERSITÉ & MAUVE
        # Ici on doit construire le nom nous-mêmes.
        # On ajoute judge_short_name pour que le fichier soit unique par configuration de juge.
        base_name = os.path.splitext(file_path)[0]
        div_output_path = f"{base_name}_{judge_short_name}_diversity_mauve_gen_length_result.json"
        
        div_cmd = [
            python_exe, 'open_text_gen/measure_diversity_mauve_gen_length.py',
            '--test_path', file_path,
            '--result_path', div_output_path
        ]
        try:
            print(f"Calcul MAUVE & Diversité -> {os.path.basename(div_output_path)}...")
            subprocess.run(div_cmd, check=True)
        except Exception as e:
            print(f"Erreur MAUVE: {e}")

    print("\n Toutes les évaluations Ollama sont terminées !")

if __name__ == '__main__':
    main()