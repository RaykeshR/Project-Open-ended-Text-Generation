import argparse
import json
import ollama
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    # Liste des modèles Ollama à tester (ex: llama3, mistral)
    parser.add_argument('--models', type=str, nargs='+', default=['llama3', 'mistral'], help='Liste des modèles Ollama')
    parser.add_argument('--dataset_name', type=str, default='wikitext', help='Nom du dataset')
    parser.add_argument('--dataset_config', type=str, default='wikitext-103-raw-v1', help='Config du dataset')
    parser.add_argument('--dataset_split', type=str, default='test', help='Split du dataset')
    parser.add_argument('--output_dir', type=str, default='open_text_gen/ollama_results', help='Dossier de sortie')
    parser.add_argument('--num_prefixes', type=int, default=100, help='Nombre de textes à générer')
    
    args = parser.parse_args()

    # On utilise le tokenizer GPT-2 juste pour découper le texte de la même manière 
    # que dans vos autres expériences (32 tokens de prompt / 128 tokens de gold).
    # Cela garantit que la comparaison est 100% équitable.
    splitter_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    for model_name in args.models:
        print(f"\n Lancement de la génération avec Ollama : {model_name}")
        
        # Nom de fichier compatible avec vos scripts d'analyse
        # Format : {dataset}_{model}_ollama.jsonl
        safe_model_name = model_name.replace(':', '-') # Ollama utilise ':' (ex: llama3:8b), on remplace pour le fichier
        output_filename = f'{args.dataset_name}_{safe_model_name}_ollama.jsonl'
        output_path = f'{args.output_dir}/{output_filename}'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in tqdm(range(args.num_prefixes)):
                full_text = dataset[i]['text']
                tokens = splitter_tokenizer.encode(full_text)
                
                prefix_len = 32
                gold_len = 128
                
                if len(tokens) < prefix_len + 10:
                    continue

                prefix_tokens = tokens[:prefix_len]
                gold_tokens = tokens[prefix_len : prefix_len + gold_len]
                
                prefix_text = splitter_tokenizer.decode(prefix_tokens, skip_special_tokens=True)
                gold_text = splitter_tokenizer.decode(gold_tokens, skip_special_tokens=True)

                # --- APPEL OLLAMA ---
                try:
                    response = ollama.generate(model=model_name, prompt=prefix_text)
                    generated_text = response['response']
                    
                    # On nettoie un peu si le modèle répète le prompt (dépend des modèles)
                    if generated_text.startswith(prefix_text):
                        generated_text = generated_text[len(prefix_text):]

                except Exception as e:
                    print(f"Erreur avec {model_name}: {e}")
                    continue
                # --------------------

                result = {
                    "prompt": prefix_text,
                    "gen_text": generated_text,
                    "gold_ref": gold_text,
                    "model": model_name,
                    "source": "ollama"
                }
                
                f.write(json.dumps(result) + '\n')
        
        print(f" Fini pour {model_name}. Résultat : {output_path}")

if __name__ == '__main__':
    main()