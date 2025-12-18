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
    parser.add_argument('--beam_width', type=int, default=5, help='the beam width for contrastive search')
    parser.add_argument('--decoding_len', type=int, default=256, help='the decoding length')
    parser.add_argument('--num_prefixes', type=int, default=100, help='the number of prefixes to use from the dataset')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = SimCTGGPT(args.model_name)
    model.to(device)
    model.eval()

    # Load the dataset
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    
    for alpha in args.alphas:
        output_filename = f'{args.dataset_name}_contrastive-alpha-{alpha}_{args.model_name}_{args.decoding_len}.jsonl'
        output_path = f'{args.output_dir}/{output_filename}'
        
        with open(output_path, 'w') as f:
            results_list = []
            for i in tqdm(range(args.num_prefixes)):
                prefix_text = dataset[i]['text']
                if len(prefix_text) == 0:
                    continue

                input_ids = tokenizer.encode(prefix_text, return_tensors='pt').to(device)

                if args.decoding_strategy == 'contrastive':
                    output = model.fast_contrastive_search(
                        input_ids=input_ids,
                        beam_width=args.beam_width,
                        alpha=alpha,
                        decoding_len=args.decoding_len
                    )
                else:
                    raise NotImplementedError
                generated_full_text = tokenizer.decode(output, skip_special_tokens=True)
                result = {
                    "ended": False, # Valeur par défaut
                    "tokens": output, # La liste des IDs
                    "prompt": prefix_text,
                    "gen_text": generated_full_text,
                    "len": len(output) - input_ids.size(1),
                    "nll4tok": [], # Calculable via model.forward si nécessaire
                    "ppl": 0,      # Calculable via model.eval_loss si nécessaire
                    "gold_ref": "" # Si vous avez une référence dans votre dataset
                }
                results_list.append(result)

            # Sauvegarde finale sous forme de liste JSON (format [{}, {}])
            json.dump(results_list, f, indent=4)

if __name__ == '__main__':
    main()
