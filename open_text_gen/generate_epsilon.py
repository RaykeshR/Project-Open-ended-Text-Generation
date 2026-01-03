import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from _utlis_.simctg.simctggpt import SimCTGGPT
except ImportError:
    from open_text_gen._utlis_.simctg.simctggpt import SimCTGGPT
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2-xl', help='the name of the language model')
    parser.add_argument('--dataset_name', type=str, default='wikitext', help='the name of the dataset to use')
    parser.add_argument('--dataset_config', type=str, default='wikitext-103-raw-v1', help='the config of the dataset to use')
    parser.add_argument('--dataset_split', type=str, default='test', help='the split of the dataset to use')
    parser.add_argument('--output_dir', type=str, default='open_text_gen/wikitext', help='the directory to save the output file')
    
    # Epsilon Greedy specific arguments
    parser.add_argument('--alpha', type=float, default=0.5, help='Threshold alpha for Epsilon Greedy')
    parser.add_argument('--k', type=int, default=10, help='K for Top-K sampling in exploration mode')
    
    parser.add_argument('--decoding_len', type=int, default=256, help='the decoding length')
    parser.add_argument('--num_prefixes', type=int, default=100, help='the number of prefixes to use from the dataset')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')

    # Load Model and Tokenizer
    # We use SimCTGGPT wrapper if possible to match generate.py, but we access the underlying model
    # for our custom loop.
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = SimCTGGPT(args.model_name)
    model.to(device)
    model.eval()
    
    # Access the underlying Hugging Face model
    hf_model = model.model
    
    safe_model_name = args.model_name.replace('/', '-')

    # Load Dataset
    print(f"Loading dataset {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split, trust_remote_code=True)
    
    # Output filename
    output_filename = f'{args.dataset_name}_epsilon_k{args.k}_alpha{args.alpha}_{safe_model_name}.jsonl'
    output_path = f'{args.output_dir}/{output_filename}'
    
    print(f"\n--- Generation Epsilon Greedy: k={args.k}, alpha={args.alpha} ---")
    print(f"Saving to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        compteur_valide = 0
        index_dataset = 0
        pbar = tqdm(total=args.num_prefixes)

        while compteur_valide < args.num_prefixes:
            if index_dataset >= len(dataset):
                break 

            full_text = dataset[index_dataset]['text']
            
            # Tokenize
            tokens = tokenizer.encode(full_text)
            
            prefix_len = 32
            gold_len = 128
            
            # Check length
            if len(tokens) < prefix_len + 10:
                index_dataset += 1
                continue

            prefix_tokens = tokens[:prefix_len]
            gold_tokens = tokens[prefix_len : prefix_len + gold_len]
            
            prefix_text = tokenizer.decode(prefix_tokens, skip_special_tokens=True)
            gold_text = tokenizer.decode(gold_tokens, skip_special_tokens=True)

            input_ids = torch.LongTensor(prefix_tokens).unsqueeze(0).to(device)

            # --- Epsilon Greedy Generation Loop ---
            curr_input_ids = input_ids.clone()
            
            with torch.no_grad():
                for _ in range(args.decoding_len):
                    # Forward pass
                    outputs = hf_model(curr_input_ids)
                    next_token_logits = outputs.logits[:, -1, :] # (batch_size, vocab_size)
                    
                    # Epsilon Logic
                    epsilon = random.random()
                    
                    if epsilon < args.alpha:
                        # Exploration: Top-K Sampling
                        probs = torch.softmax(next_token_logits, dim=-1)
                        top_k_probs, top_k_indices = torch.topk(probs, args.k, dim=-1)
                        
                        # Re-normalize probabilities
                        top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=-1, keepdim=True)
                        
                        # Sample
                        next_token_index = torch.multinomial(top_k_probs, num_samples=1) # index within top-k
                        next_token = torch.gather(top_k_indices, -1, next_token_index)
                    else:
                        # Exploitation: Greedy (Argmax)
                        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                    
                    # Append prediction
                    curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)
            
            # Extract generated tokens (excluding prefix)
            output_full_tokens = curr_input_ids[0].tolist()
            gen_tokens = output_full_tokens[prefix_len:]
            generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            # Result object
            result = {
                "ended": False,
                "tokens": output_full_tokens, 
                "prompt": prefix_text,
                "gen_text": generated_text,
                "len": len(gen_tokens),
                "nll4tok": [], 
                "ppl": 0,
                "gold_ref": gold_text,
            }
            
            f.write(json.dumps(result) + '\n')
            compteur_valide += 1
            pbar.update(1)
            index_dataset += 1
            
        pbar.close()
        print(f"Generation completed. {compteur_valide} examples generated.")

if __name__ == '__main__':
    main()
