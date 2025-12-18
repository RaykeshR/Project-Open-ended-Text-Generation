import torch
from transformers import AutoTokenizer
from _utlis_.simctg.simctggpt import SimCTGGPT
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2-xl', help='the name of the language model')
    parser.add_argument('--dataset_name', type=str, default='wikitext', help='the name of the dataset to use')
    parser.add_argument('--dataset_config', type=str, default='wikitext-103-raw-v1', help='the config of the dataset to use')
    parser.add_argument('--dataset_split', type=str, default='test', help='the split of the dataset to use')
    parser.add_argument('--output_dir', type=str, default='.', help='the directory to save the output file')
    parser.add_argument('--decoding_strategy', type=str, default='contrastive', help='the decoding strategy to use')
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.6], help='the list of alpha values for contrastive search')
    parser.add_argument('--ks', type=int, nargs='+', default=[5], help='Liste des valeurs de K')
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.0], help='Liste des seuils epsilon')
    parser.add_argument('--beam_width', type=int, default=5, help='the beam width for contrastive search')
    parser.add_argument('--decoding_len', type=int, default=256, help='the decoding length')
    parser.add_argument('--num_prefixes', type=int, default=100, help='the number of prefixes to use from the dataset')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')

    # Chargement du tokenizer et du modèle
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = SimCTGGPT(args.model_name)
    model.to(device)
    model.eval()

    # Chargement du dataset
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    
    for k in args.ks:
        for alpha in args.alphas:
            for epsilon in args.epsilons:
                output_filename = f'{args.dataset_name}_k{k}_a{alpha}_e{epsilon}_{args.model_name}.jsonl'
                output_path = f'{args.output_dir}/{output_filename}'
                
                with open(output_path, 'w', encoding='utf-8') as f: # Ouverture du fichier
                    for i in tqdm(range(args.num_prefixes)):
                        full_text = dataset[i]['text']
                        
                        # Tokenisation du texte original pour découpage précis
                        tokens = tokenizer.encode(full_text)
                        
                        # Paramètres standards pour l'évaluation (modifiable)
                        prefix_len = 32  # Longueur du texte donné à l'IA
                        gold_len = 128   # Longueur de la référence humaine à conserver
                        
                        # On vérifie qu'on a assez de texte pour faire un test
                        if len(tokens) < prefix_len + 10:
                            continue

                        prefix_tokens = tokens[:prefix_len]
                        # La "Gold Reference" est ce qui vient juste après le préfixe dans le texte original
                        gold_tokens = tokens[prefix_len : prefix_len + gold_len]
                        
                        prefix_text = tokenizer.decode(prefix_tokens, skip_special_tokens=True)
                        gold_text = tokenizer.decode(gold_tokens, skip_special_tokens=True)

                        input_ids = torch.LongTensor(prefix_tokens).unsqueeze(0).to(device)

                        # Génération par l'IA
                        if args.decoding_strategy == 'contrastive':
                            output = model.fast_contrastive_search(
                                input_ids=input_ids,
                                beam_width=k,
                                alpha=alpha,
                                decoding_len=args.decoding_len
                            )
                        else:
                            raise NotImplementedError
                        
                        # On ne décode que la partie générée (on ignore les tokens du préfixe)
                        gen_tokens = output[prefix_len:]
                        generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

                        # Construction de l'objet de résultat final
                        result = {
                            "ended": False,
                            "tokens": output, 
                            "prompt": prefix_text,         # Le début
                            "gen_text": generated_text,    # Ce que l'IA a écrit
                            "len": len(gen_tokens),
                            "nll4tok": [], 
                            "ppl": 0,
                            "gold_ref": gold_text,        # Ce que l'humain a écrit
                        }
                        
                        # Sauvegarde immédiate ligne par ligne (JSONL)
                        f.write(json.dumps(result) + '\n')

if __name__ == '__main__':
    main()
