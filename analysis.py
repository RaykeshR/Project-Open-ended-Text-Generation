import json
import os
import glob
import pandas as pd
import re
import numpy as np
from collections import defaultdict

# --- Core Logic from compute_diversity.py ---
# I'm embedding the necessary functions here to avoid changing library files.

def eval_text(text, ngram):
    token_list = text.strip().split()
    start_idx, end_idx = 0, ngram
    total_num = 0
    ngram_set = set()
    while end_idx < len(token_list):
        one_ngram_list = token_list[start_idx:end_idx]
        assert len(one_ngram_list) == ngram
        one_ngram = ' '.join(one_ngram_list)
        total_num += 1
        ngram_set.add(one_ngram)
        start_idx += 1
        end_idx += 1
    # Handle division by zero if total_num is 0
    return len(ngram_set), total_num if total_num > 0 else 1

def eval_one_instance(text, ngram_list):
    res_dict = {}
    for n in ngram_list:
        n_unique, n_total = eval_text(text, n)
        res_dict[n] = {'unique':n_unique, 'total':n_total}
    unique_token_set = set(text.strip().split())
    return res_dict, unique_token_set

def calculate_correct_diversity_metrics(text_list):
    """
    This function correctly calculates repetition rates (rep-N).
    It's adapted from measure_repetition_and_diversity in the project files.
    """
    ngram_list = [2, 3, 4]
    pred_res_dict = {n: {'unique': 0, 'total': 0} for n in ngram_list}
    
    for text in text_list:
        text = text.strip('\n').strip()
        one_pred_res_dict, _ = eval_one_instance(text, ngram_list)
        for n in ngram_list:
            pred_res_dict[n]['unique'] += one_pred_res_dict[n]['unique']
            pred_res_dict[n]['total'] += one_pred_res_dict[n]['total']

    # Calculate repetition scores
    rep_2 = 1 - (pred_res_dict[2]['unique'] / pred_res_dict[2]['total'])
    rep_3 = 1 - (pred_res_dict[3]['unique'] / pred_res_dict[3]['total'])
    rep_4 = 1 - (pred_res_dict[4]['unique'] / pred_res_dict[4]['total'])
    
    return {
        'rep-2': round(rep_2, 4),
        'rep-3': round(rep_3, 4),
        'rep-4': round(rep_4, 4)
    }

def load_predictions_from_jsonl(filepath):
    """Loads the generated text from a .jsonl file."""
    predictions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                predictions.append(item.get('gen_text') or item.get('generated') or "")
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line in {filepath}")
    return predictions

# --- Main Analysis Script ---

def main():
    """Main function to run the complete, corrected analysis."""
    directory = 'open_text_gen/wikitext_grid_search'
    print(f"Analyzing results from: {directory}")

    # Set up pandas display
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 220)
    pd.set_option('display.float_format', '{:.4f}'.format)

    # --- Step 1: Find all .jsonl generation files ---
    jsonl_files = glob.glob(os.path.join(directory, '*.jsonl'))
    
    # Regex to parse the .jsonl filename
    # e.g., wikitext_k10_a0.4_e0.0_gpt2-xl.jsonl
    jsonl_pattern = re.compile(
        r'wikitext_k(\d+)_a([\d.]+)_e([\d.]+)_([^\.]+)\.jsonl'
    )
    
    results_data = defaultdict(dict)

    # --- Step 2: Process each .jsonl file for diversity and other metrics ---
    for f_path in jsonl_files:
        basename = os.path.basename(f_path)
        match = jsonl_pattern.match(basename)
        
        if not match:
            print(f"Warning: Skipping file with unexpected name format: {basename}")
            continue
            
        k, alpha, epsilon, gen_model = match.groups()
        key = (gen_model, int(k), float(alpha)) # Epsilon is always 0.0, so we can omit

        # Calculate correct diversity metrics
        predictions = load_predictions_from_jsonl(f_path)
        if predictions:
            diversity_metrics = calculate_correct_diversity_metrics(predictions)
            results_data[key].update(diversity_metrics)

        # We still need MAUVE and Gen_Length from the old files.
        # Let's try to read them from the malformed json results.
        div_result_file = f_path.replace('.jsonl', '_diversity_mauve_gen_length_result.json')
        if os.path.exists(div_result_file):
            try:
                with open(div_result_file, 'r') as f:
                    data = json.load(f)[0]
                if 'mauve_dict' in data:
                    results_data[key]['MAUVE'] = float(data['mauve_dict']['mauve_mean'])
                if 'gen_length_dict' in data:
                     results_data[key]['Gen_Length'] = float(data['gen_length_dict']['gen_len_mean'])
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Warning: Could not extract MAUVE/Gen_Length from {div_result_file}: {e}")

    # --- Step 3: Process coherence files ---
    coh_files = glob.glob(os.path.join(directory, '*_coherence_result.json'))
    coh_pattern = re.compile(
        r'wikitext_k(\d+)_a([\d.]+)_e([\d.]+)_([^_]+)\._([^_]+)_coherence_result\.json'
    )
    for c_path in coh_files:
        basename = os.path.basename(c_path)
        match = coh_pattern.match(basename)
        if not match:
            continue
        
        k, alpha, epsilon, gen_model, coh_model = match.groups()
        key = (gen_model, int(k), float(alpha))
        
        try:
            with open(c_path, 'r') as f:
                data = json.load(f)[0]
            coh_key = f"Coh_{coh_model.replace('facebook/', '')}"
            results_data[key][coh_key] = float(data['coherence_mean'])
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Warning: Could not process coherence file {c_path}: {e}")

    # --- Step 4: Create and display the DataFrame ---
    final_list = []
    for (gen_model, k, alpha), metrics in results_data.items():
        record = {'Model': gen_model, 'k': k, 'alpha': alpha, **metrics}
        final_list.append(record)

    if not final_list:
        print("ERROR: No data was successfully processed.")
        return
        
    df = pd.DataFrame(final_list)
    
    # Reorder columns
    cols = ['Model', 'k', 'alpha']
    coh_cols = sorted([c for c in df.columns if c.startswith('Coh_')])
    other_metrics = ['MAUVE', 'Gen_Length', 'rep-2', 'rep-3', 'rep-4']
    final_cols = cols + [c for c in coh_cols + other_metrics if c in df.columns]
    df = df[final_cols]
    
    df = df.sort_values(by=['Model', 'k', 'alpha']).reset_index(drop=True)

    print("\n" + "="*120)
    print("Grid Search Results Analysis (Corrected)")
    print("="*120)
    print(df.to_string())
    print("="*120)

    # --- Step 5: Highlights ---
    print("\n--- HIGHLIGHTS ---")
    if coh_cols:
        main_coh_col = coh_cols[0] # Rank by the first coherence judge
        best_coherence = df.sort_values(by=main_coh_col, ascending=False).head(3)
        print(f"\n🏆 Top 3 for Coherence ({main_coh_col}):")
        display_cols = ['Model', 'k', 'alpha', main_coh_col, 'MAUVE', 'rep-3']
        print(best_coherence[[c for c in display_cols if c in df.columns]])

    if 'MAUVE' in df.columns:
        best_mauve = df.sort_values(by='MAUVE', ascending=False).head(3)
        print(f"\n🏆 Top 3 for MAUVE (Distribution Similarity):")
        display_cols = ['Model', 'k', 'alpha', 'MAUVE', coh_cols[0] if coh_cols else 'MAUVE', 'rep-3']
        print(best_mauve[[c for c in display_cols if c in df.columns]])
        
    if 'rep-3' in df.columns:
        lowest_rep3 = df.sort_values(by='rep-3', ascending=True).head(3)
        print(f"\n🏆 Top 3 for Diversity (Lowest rep-3):")
        display_cols = ['Model', 'k', 'alpha', 'rep-3', 'MAUVE', coh_cols[0] if coh_cols else 'MAUVE']
        print(lowest_rep3[[c for c in display_cols if c in df.columns]])

    print("\nAnalysis complete.")

if __name__ == '__main__':
    main()