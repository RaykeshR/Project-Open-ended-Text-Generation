import subprocess
import os
import sys

def main():
    # Parameters to test
    alphas = [0.2, 0.4, 0.6, 0.8]
    ks = [5, 10, 50]
    
    # Fixed configuration
    model_name = "gpt2-xl"
    dataset_name = "wikitext"
    num_prefixes = 50
    decoding_len = 256
    
    # Directory structure
    # Based on the project structure, results seem to go into open_text_gen/{dataset_name}
    output_dir = os.path.join("open_text_gen", dataset_name)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Safe model name for filename construction (matching generate_epsilon.py logic)
    safe_model_name = model_name.replace('/', '-')

    for k in ks:
        for alpha in alphas:
            print(f"\n{'='*60}")
            print(f"Running Epsilon Search with alpha={alpha}, k={k}")
            print(f"{'='*60}\n")
            
            # --- Step 1: Generation ---
            gen_cmd = [
                sys.executable, "open_text_gen/generate_epsilon.py",
                "--model_name", model_name,
                "--dataset_name", dataset_name,
                "--alpha", str(alpha),
                "--k", str(k),
                "--num_prefixes", str(num_prefixes),
                "--decoding_len", str(decoding_len),
                "--output_dir", output_dir
            ]
            
            print(f"[Generation] Executing: {' '.join(gen_cmd)}")
            try:
                subprocess.run(gen_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error during generation for alpha={alpha}, k={k}. Skipping evaluation.")
                print(e)
                continue
            
            # --- Step 2: Evaluation ---
            # Construct the expected output filename
            # Format from generate_epsilon.py: f'{args.dataset_name}_epsilon_k{args.k}_alpha{args.alpha}_{safe_model_name}.jsonl'
            output_filename = f"{dataset_name}_epsilon_k{k}_alpha{alpha}_{safe_model_name}.jsonl"
            output_path = os.path.join(output_dir, output_filename)
            
            if not os.path.exists(output_path):
                print(f"[Error] Generated file not found: {output_path}")
                continue
                
            eval_cmd = [
                sys.executable, "open_text_gen/measure_diversity_mauve_gen_length.py",
                "--test_path", output_path
            ]
            
            print(f"[Evaluation] Executing: {' '.join(eval_cmd)}")
            try:
                subprocess.run(eval_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error during evaluation for file {output_path}.")
                print(e)

if __name__ == "__main__":
    main()
