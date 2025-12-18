import os
import subprocess
import json
import argparse
import sys  # Ajout important pour utiliser le python courant

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.3, 0.5, 0.7], help='the list of alpha values for contrastive search')
    parser.add_argument('--dataset_name', type=str, default='wikinews', help='the name of the dataset to use')
    parser.add_argument('--model_name', type=str, default='gpt2-xl', help='the name of the language model')
    parser.add_argument('--decoding_len', type=int, default=256, help='the decoding length')
    parser.add_argument('--num_prefixes', type=int, default=100, help='the number of prefixes to use from the dataset')
    parser.add_argument('--beam_width', type=int, default=5, help='the beam width for contrastive search')
    
    args = parser.parse_args()

    output_dir = f'open_text_gen/{args.dataset_name}'
    os.makedirs(output_dir, exist_ok=True)

    # Chemin vers l'interpréteur python actuel (celui du venv)
    python_exe = sys.executable

    for alpha in args.alphas:
        print(f'Running experiment for alpha = {alpha}')
        
        # 1. Generate text
        generation_output_filename = f'{args.dataset_name}_contrastive-alpha-{alpha}_{args.model_name}_{args.decoding_len}.jsonl'
        generation_output_path = f'{output_dir}/{generation_output_filename}'
        
        generation_cmd = [
            python_exe, 'open_text_gen/generate.py', # Utilise sys.executable
            '--model_name', args.model_name,
            '--dataset_name', args.dataset_name,
            # '--dataset_config', 'wikinews',
            '--output_dir', output_dir,
            '--decoding_strategy', 'contrastive',
            '--alphas', str(alpha),
            '--beam_width', str(args.beam_width),
            '--decoding_len', str(args.decoding_len),
            '--num_prefixes', str(args.num_prefixes)
        ]
        print("Generating text...")
        subprocess.run(generation_cmd, check=True)

        # 2. Evaluate
        # 2.1 Coherence
        # CORRECTION : Appel direct au script python au lieu du script bash
        coherence_output_filename = generation_output_filename.replace('.jsonl', '_opt-2.7b_coherence_result.json')
        # Note: compute_coherence.py calcule lui-même le chemin de sortie basé sur test_path, 
        # on ne passe donc que les arguments qu'il accepte.
        
        coherence_cmd = [
            python_exe, 'open_text_gen/compute_coherence.py',
            '--opt_model_name', 'facebook/opt-125m',
            # '--opt_model_name', 'facebook/opt-2.7b', # Vous pouvez changer pour 'facebook/opt-125m' pour tester plus vite
            '--test_path', generation_output_path
        ]
        print("Measuring coherence...")
        subprocess.run(coherence_cmd, check=True)

        # 2.2 Diversity, Mauve, Gen Length
        diversity_output_filename = generation_output_filename.replace('.jsonl', '_diversity_mauve_gen_length_result.json')
        diversity_output_path = f'{output_dir}/{diversity_output_filename}'
        diversity_cmd = [
            python_exe, 'open_text_gen/measure_diversity_mauve_gen_length.py',
            '--test_path', generation_output_path,
            '--result_path', diversity_output_path
        ]
        print("Measuring diversity and MAUVE...")
        subprocess.run(diversity_cmd, check=True)
        
    print('Experiment finished.')

if __name__ == '__main__':
    main()