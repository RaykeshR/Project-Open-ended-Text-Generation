import torch
import json
import argparse
import os
import glob
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def calculate_perplexity(text, model, tokenizer, device):
    """Calcule la perplexité d'un texte donné."""
    encodings = tokenizer(text, return_tensors='pt')
    # Si le texte est vide ou trop long pour le modèle (rare ici mais possible)
    if encodings.input_ids.size(1) == 0:
        return None
        
    input_ids = encodings.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    
    loss = outputs.loss
    return torch.exp(loss).item()

def process_file(file_path, model, tokenizer, device):
    """Traite un fichier .jsonl et sauvegarde le résultat .json correspondant."""
    print(f"Traitement de : {os.path.basename(file_path)}")
    
    ppl_values = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                # On évalue le texte généré ('gen_text') ou le texte complet ('text') selon le besoin.
                # Ici, on se concentre souvent sur la qualité de la génération seule.
                text = data.get('gen_text', '').strip()
                
                if text:
                    ppl = calculate_perplexity(text, model, tokenizer, device)
                    if ppl is not None and not np.isinf(ppl) and not np.isnan(ppl):
                        ppl_values.append(ppl)
    except Exception as e:
        print(f"Erreur lors de la lecture de {file_path}: {e}")
        return

    if ppl_values:
        mean_ppl = float(np.mean(ppl_values))
        
        # Création du nom de fichier de sortie compatible avec votre script d'analyse
        # Ex: fichier.jsonl -> fichier_perplexity_result.json
        output_filename = file_path.replace('.jsonl', '_perplexity_result.json')
        
        result_data = {
            'filename': os.path.basename(file_path),
            'mean_perplexity': mean_ppl,
            'ppl': mean_ppl,  # Double clé pour être sûr que votre script le trouve
            'perplexity': mean_ppl
        }
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4)
        
        print(f" -> Sauvegardé : {os.path.basename(output_filename)} (PPL: {mean_ppl:.2f})")
    else:
        print(f" -> Aucune donnée valide trouvée pour {file_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2-xl', help='Nom du modèle pour calculer la PPL')
    parser.add_argument('--folder', type=str, required=True, help='Dossier contenant les fichiers .jsonl')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Chargement du modèle {args.model_name} sur {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    # Trouver tous les fichiers .jsonl dans le dossier cible
    jsonl_files = glob.glob(os.path.join(args.folder, '*.jsonl'))
    
    if not jsonl_files:
        print(f"Aucun fichier .jsonl trouvé dans {args.folder}")
        return

    print(f"{len(jsonl_files)} fichiers trouvés. Calcul en cours...")
    
    for f in tqdm(jsonl_files):
        # Vérifier si le résultat existe déjà pour éviter de tout refaire
        expected_output = f.replace('.jsonl', '_perplexity_result.json')
        if os.path.exists(expected_output):
            print(f"Skipping {os.path.basename(f)} (déjà fait)")
            continue
            
        process_file(f, model, tokenizer, device)

if __name__ == "__main__":
    main()
    # python open_text_gen/measure_perplexity.py --folder open_text_gen/wikitext_epsilon_grid_search --model_name gpt2-xl
    # python open_text_gen/measure_perplexity.py --folder open_text_gen/cc_news_epsilon_grid_search --model_name gpt2-xl
    # python open_text_gen/measure_perplexity.py --folder open_text_gen/bookcorpus_epsilon_grid_search --model_name gpt2-xl
    # python open_text_gen/measure_perplexity.py --folder open_text_gen/wikitext_epsilon_grid_search_256 --model_name gpt2-xl