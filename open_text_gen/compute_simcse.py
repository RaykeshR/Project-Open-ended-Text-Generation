import torch
import json
import argparse
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True, help='Path to the .jsonl file')
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Chargement du modèle SimCSE (Supervised)
    # C'est le modèle standard demandé pour cette tâche
    model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
    print(f"Chargement du modèle SimCSE : {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # 2. Chargement des données
    prompts = []
    generated_texts = []
    
    print(f"Lecture du fichier : {args.test_path}")
    with open(args.test_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Gestion des différentes clés possibles selon vos scripts
            prompt = data.get('prompt') or data.get('prefix')
            gen = data.get('gen_text') or data.get('generated')
            
            if prompt and gen:
                prompts.append(prompt)
                generated_texts.append(gen)

    if not prompts:
        print("Erreur: Aucune donnée trouvée dans le fichier.")
        return

    # 3. Calcul des Embeddings et de la Similarité
    similarities = []
    
    print(f"Calcul de la cohérence sémantique sur {len(prompts)} exemples...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), args.batch_size)):
            batch_prompts = prompts[i : i + args.batch_size]
            batch_gens = generated_texts[i : i + args.batch_size]
            
            # Tokenization
            inputs_p = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt").to(device)
            inputs_g = tokenizer(batch_gens, padding=True, truncation=True, return_tensors="pt").to(device)
            
            # Forward pass (On prend le token [CLS] comme représentation de la phrase)
            # SimCSE est entraîné pour que le [CLS] contienne le sens global
            emb_p = model(**inputs_p).last_hidden_state[:, 0]
            emb_g = model(**inputs_g).last_hidden_state[:, 0]
            
            # Calcul du Cosinus Similarité
            # Sim(A, B) = (A . B) / (||A|| * ||B||)
            cosine_sim = torch.cosine_similarity(emb_p, emb_g)
            
            similarities.extend(cosine_sim.cpu().tolist())

    # 4. Statistiques
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    
    print(f"Résultat moyen : {mean_sim:.4f}")

    # 5. Sauvegarde
    # On sauvegarde avec un suffixe clair pour ne pas écraser l'ancienne métrique
    output_path = args.test_path.replace('.jsonl', '_simcse_result.json')
    
    result = {
        "metric": "SimCSE_Coherence",
        "model_judge": model_name,
        "coherence_mean": mean_sim, # Clé standardisée
        "coherence_std": std_sim,
        "scores": similarities
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
    
    print(f"Sauvegardé dans : {output_path}")

if __name__ == "__main__":
    main()