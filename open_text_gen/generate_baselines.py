import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2-xl')
    parser.add_argument('--dataset_name', type=str, default='wikitext')
    parser.add_argument('--dataset_config', type=str, default='wikitext-103-raw-v1')
    parser.add_argument('--dataset_split', type=str, default='test')
    parser.add_argument('--output_dir', type=str, default='.')
    
    # Stratégies
    parser.add_argument('--decoding_strategy', type=str, required=True, help='greedy, nucleus, typical, beam')
    parser.add_argument('--probs', type=float, default=1.0, help='p value for nucleus/typical')
    parser.add_argument('--num_beams', type=int, default=1, help='Beam size for beam search') 
    parser.add_argument('--decoding_len', type=int, default=256)
    parser.add_argument('--num_prefixes', type=int, default=100)
    
    # Arguments ignorés (pour compatibilité si appelés par erreur)
    parser.add_argument('--alphas', nargs='+', default=[])
    parser.add_argument('--ks', nargs='+', default=[])
    parser.add_argument('--epsilons', nargs='+', default=[])

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device} | Strategy: {args.decoding_strategy}')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # On utilise le modèle standard HF, plus robuste pour les baselines
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split, trust_remote_code=True)

    # Nom du fichier de sortie
    safe_model_name = args.model_name.replace('/', '-')
    if args.decoding_strategy == 'greedy':
        suffix = "greedy"
    elif args.decoding_strategy == 'nucleus':
        suffix = f"p-{args.probs}"
    elif args.decoding_strategy == 'typical':
        suffix = f"typical-{args.probs}"
    elif args.decoding_strategy == 'beam':
        suffix = f"beam-{args.num_beams}"
    else:
        suffix = args.decoding_strategy

    output_filename = f'{args.dataset_name}_{suffix}_{safe_model_name}_{args.decoding_len}.jsonl'
    output_path = os.path.join(args.output_dir, output_filename)
    print(f"Saving to: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        compteur = 0
        idx = 0
        pbar = tqdm(total=args.num_prefixes)

        while compteur < args.num_prefixes:
            if idx >= len(dataset): break
            
            text = dataset[idx]['text']
            tokens = tokenizer.encode(text)
            
            prefix_len = 32
            gold_len = 128
            
            if len(tokens) < prefix_len + 10:
                idx += 1
                continue

            prefix_tokens = tokens[:prefix_len]
            gold_tokens = tokens[prefix_len : prefix_len + gold_len]
            prefix_text = tokenizer.decode(prefix_tokens, skip_special_tokens=True)
            gold_text = tokenizer.decode(gold_tokens, skip_special_tokens=True)

            input_ids = torch.LongTensor(prefix_tokens).unsqueeze(0).to(device)

            try:
                # Configuration de la génération selon la stratégie
                gen_kwargs = {
                    "input_ids": input_ids,
                    "max_new_tokens": args.decoding_len,
                    "pad_token_id": tokenizer.eos_token_id,
                    "do_sample": False # Par défaut (Greedy)
                }

                if args.decoding_strategy == 'nucleus':
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["top_p"] = args.probs
                elif args.decoding_strategy == 'typical':
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["typical_p"] = args.probs
                elif args.decoding_strategy == 'beam': 
                    gen_kwargs["do_sample"] = False
                    gen_kwargs["num_beams"] = args.num_beams
                    gen_kwargs["early_stopping"] = True
                
                # Génération
                # output contient [prompt + generated_text]
                output = model.generate(**gen_kwargs)[0]
                
                # Décodage
                gen_tokens = output[prefix_len:]
                generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

                # =================================================================
                # CALCUL DE LA PERPLEXITÉ (AJOUT)
                # =================================================================
                ppl_value = 0.0
                if len(gen_tokens) > 0:
                    with torch.no_grad():
                        # On prépare les labels : ce sont les mêmes tokens que l'input
                        target_ids = output.clone()
                        # On remplace la partie "prompt" par -100 pour que la Loss l'ignore
                        target_ids[:prefix_len] = -100
                        
                        # On remet en dimension [batch_size, seq_len] pour le modèle
                        model_inputs = output.unsqueeze(0)
                        target_ids = target_ids.unsqueeze(0)

                        # Calcul de la loss
                        outputs = model(model_inputs, labels=target_ids)
                        
                        # La loss renvoyée est la CrossEntropy moyenne sur les tokens valides
                        neg_log_likelihood = outputs.loss
                        ppl_value = torch.exp(neg_log_likelihood).item()

                # --- FORMAT JSON ---
                result = {
                    "ended": False,
                    "tokens": output.tolist(),     # Obligatoire pour certains scripts
                    "prompt": prefix_text,
                    "gen_text": generated_text,
                    "len": len(gen_tokens),
                    "nll4tok": [],                 # Champ vide pour compatibilité
                    "ppl": ppl_value,              # <--- Valeur calculée ici
                    "gold_ref": gold_text,
                    "strategy": args.decoding_strategy
                }
                
                f.write(json.dumps(result) + '\n')
                compteur += 1
                pbar.update(1)

            except Exception as e:
                print(f"Error: {e}")
            
            idx += 1
        print(f"{compteur} itérations")
        pbar.close()

if __name__ == '__main__':
    main()