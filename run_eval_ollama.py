import os
import subprocess
import sys
import glob

def main():
    # Dossier où sont les résultats Ollama
    results_dir = 'open_text_gen/ollama_results'
    Juge_model_name = 'facebook/opt-125m'
    python_exe = sys.executable

    # On cherche tous les fichiers .jsonl dans le dossier
    jsonl_files = glob.glob(f"{results_dir}/*.jsonl")
    
    if not jsonl_files:
        print(f"Aucun fichier .jsonl trouvé dans {results_dir}. Lancez generate_ollama.py d'abord !")
        return

    print(f" Fichiers trouvés à évaluer : {len(jsonl_files)}")

    for file_path in jsonl_files:
        print(f"\n{'='*40}")
        print(f"Évaluation de : {os.path.basename(file_path)}")
        print(f"{'='*40}")
        
        # 1. COHÉRENCE (Juge (Juge_model_name): OPT-125M ou celui de votre choix)
        coh_cmd = [
            python_exe, 'open_text_gen/compute_coherence.py',
            '--opt_model_name', Juge_model_name,
            '--test_path', file_path
        ]
        try:
            print("Running Coherence...")
            subprocess.run(coh_cmd, check=True)
        except Exception as e:
            print(f"Erreur Cohérence: {e}")

        # 2. DIVERSITÉ & MAUVE
        # On définit le nom du fichier de sortie
        base_name = os.path.splitext(file_path)[0]
        div_output_path = f"{base_name}_diversity_mauve_gen_length_result.json"
        
        div_cmd = [
            python_exe, 'open_text_gen/measure_diversity_mauve_gen_length.py',
            '--test_path', file_path,
            '--result_path', div_output_path
        ]
        try:
            print("Running MAUVE & Diversity...")
            subprocess.run(div_cmd, check=True)
        except Exception as e:
            print(f"Erreur MAUVE: {e}")

    print("\n Toutes les évaluations Ollama sont terminées !")

if __name__ == '__main__':
    main()