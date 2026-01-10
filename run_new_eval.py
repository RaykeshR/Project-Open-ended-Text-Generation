import os
import glob
import subprocess
import sys

def main():
    # Dossiers où chercher vos résultats de génération
    target_dirs = [
        # 'open_text_gen/wikitext_grid_search',
        # 'open_text_gen/wikitext_epsilon_grid_search',
        'open_text_gen/ollama_results'
        # 'open_text_gen/cc_news_epsilon_grid_search',
        # 'open_text_gen/bookcorpus_epsilon_grid_search'
    ]
    
    python_exe = sys.executable
    script_path = 'open_text_gen/compute_simcse.py'
    
    print("=== DÉMARRAGE DE L'ÉVALUATION SIMCSE MASSIVE ===")
    
    files_processed = 0
    
    for directory in target_dirs:
        if not os.path.exists(directory):
            print(f"Dossier introuvable (ignoré) : {directory}")
            continue
            
        # On cherche tous les .jsonl
        jsonl_files = glob.glob(os.path.join(directory, '*.jsonl'))
        
        for jsonl_file in jsonl_files:
            # Vérifier si le résultat SimCSE existe déjà pour gagner du temps
            output_file = jsonl_file.replace('.jsonl', '_simcse_result.json')
            if os.path.exists(output_file):
                print(f"Skipping (déjà fait) : {os.path.basename(jsonl_file)}")
                continue
            
            print(f"\nTraitement de : {os.path.basename(jsonl_file)}")
            try:
                subprocess.run(
                    [python_exe, script_path, '--test_path', jsonl_file],
                    check=True
                )
                files_processed += 1
            except subprocess.CalledProcessError as e:
                print(f"ERREUR sur {jsonl_file} : {e}")

    print(f"\n=== FINI : {files_processed} fichiers évalués ===")

if __name__ == "__main__":
    main()