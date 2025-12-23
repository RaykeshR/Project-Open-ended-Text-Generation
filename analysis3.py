import json
import os
import glob
import pandas as pd
import re
import numpy as np
from collections import defaultdict

# --- Core Logic from compute_diversity.py ---
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
    return len(ngram_set), total_num if total_num > 0 else 1

def eval_one_instance(text, ngram_list):
    res_dict = {}
    for n in ngram_list:
        n_unique, n_total = eval_text(text, n)
        res_dict[n] = {'unique':n_unique, 'total':n_total}
    unique_token_set = set(text.strip().split())
    return res_dict, unique_token_set

def calculate_correct_diversity_metrics(text_list):
    ngram_list = [2, 3, 4]
    pred_res_dict = {n: {'unique': 0, 'total': 0} for n in ngram_list}
    
    for text in text_list:
        text = text.strip('\n').strip()
        one_pred_res_dict, _ = eval_one_instance(text, ngram_list)
        for n in ngram_list:
            pred_res_dict[n]['unique'] += one_pred_res_dict[n]['unique']
            pred_res_dict[n]['total'] += one_pred_res_dict[n]['total']

    # Calculate repetition scores
    # Avoid division by zero
    rep_2 = 1 - (pred_res_dict[2]['unique'] / pred_res_dict[2]['total']) if pred_res_dict[2]['total'] > 0 else 0
    rep_3 = 1 - (pred_res_dict[3]['unique'] / pred_res_dict[3]['total']) if pred_res_dict[3]['total'] > 0 else 0
    rep_4 = 1 - (pred_res_dict[4]['unique'] / pred_res_dict[4]['total']) if pred_res_dict[4]['total'] > 0 else 0
    
    return {
        'rep-2': round(rep_2, 4),
        'rep-3': round(rep_3, 4),
        'rep-4': round(rep_4, 4)
    }

def load_predictions_from_jsonl(filepath):
    predictions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                # Handle various keys used in different output formats
                predictions.append(item.get('gen_text') or item.get('generated') or "")
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line in {filepath}")
    return predictions

# --- Coloring & Printing Logic ---

def print_highlighted_table(df):
    """
    Prints the DataFrame with ANSI colors:
    - Red: Best result per column.
    - Green: Overall best row (Highest MAUVE/Coherence).
    - Blue: Best row per model (Highest MAUVE/Coherence).
    """
    if df.empty:
        print("Empty DataFrame.")
        return

    # ANSI Codes
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

    # 1. Determine "Best" direction for each column
    col_criteria = {
        'MAUVE': 'max',
        'Gen_Length': 'max',
        'rep-2': 'min',
        'rep-3': 'min',
        'rep-4': 'min'
    }
    for col in df.columns:
        if col.startswith('Coh_'):
            col_criteria[col] = 'max'

    # Calculate best value for each column
    best_values = {}
    for col in df.columns:
        if col in col_criteria:
            if col_criteria[col] == 'max':
                best_values[col] = df[col].max()
            else:
                best_values[col] = df[col].min()

    # 2. Identify Best Rows (Green/Blue) based on MAUVE or Coherence
    # Prefer MAUVE if available, otherwise first coherence column
    if 'MAUVE' in df.columns:
        sort_metric = 'MAUVE'
    else:
        available_coh = [c for c in df.columns if c.startswith('Coh_')]
        sort_metric = available_coh[0] if available_coh else None

    best_overall_idx = -1
    best_per_model_indices = []

    if sort_metric:
        best_overall_idx = df[sort_metric].idxmax()
        # Best per Model
        best_per_model_indices = df.groupby('Model')[sort_metric].idxmax().values

    # 3. Format and Print
    headers = df.columns.tolist()
    
    # Prepare formatters
    formatters = {}
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            formatters[col] = "{:.4f}"
        else:
            formatters[col] = "{}"

    # Calculate Column Widths
    col_widths = {col: len(col) for col in headers}
    for idx, row in df.iterrows():
        for col in headers:
            val_str = formatters[col].format(row[col])
            col_widths[col] = max(col_widths[col], len(val_str))
    
    for col in col_widths:
        col_widths[col] += 2

    # Print Header
    header_str = "".join([h.ljust(col_widths[h]) for h in headers])
    print(header_str)
    print("-" * len(header_str))

    # Print Rows
    for idx, row in df.iterrows():
        row_str = ""
        is_green = (idx == best_overall_idx)
        is_blue = (idx in best_per_model_indices)

        row_code = RESET
        if is_green:
            row_code = GREEN
        elif is_blue:
            row_code = BLUE
        
        for col in headers:
            val = row[col]
            val_str = formatters[col].format(val)
            
            is_red = False
            if col in best_values:
                if isinstance(val, float) or isinstance(val, int):
                    # Handle NaNs
                    if pd.notna(val) and pd.notna(best_values[col]):
                        if np.isclose(val, best_values[col]):
                            is_red = True
            
            if is_red:
                color_code = RED
            else:
                color_code = row_code
            
            padded_str = val_str.ljust(col_widths[col])
            
            if color_code != RESET:
                row_str += f"{color_code}{padded_str}{RESET}"
            else:
                row_str += padded_str
                
        print(row_str)

# --- Parsing Logic ---

def parse_filename(filename):
    """
    Parses the filename to extract Method, Model, and Parameters.
    Returns None if no pattern matches.
    """
    # 1. Grid Search Pattern (Contrastive)
    # wikitext_k5_a0.6_e0.0_gpt2-large.jsonl
    grid_pattern = re.compile(r'wikitext_k(\d+)_a([\d.]+)_e([\d.]+)_([^\.]+)\.jsonl')
    match = grid_pattern.match(filename)
    if match:
        k, alpha, epsilon, model = match.groups()
        return {
            'Method': 'Contrastive',
            'Model': model,
            'k': int(k),
            'alpha': float(alpha),
            'Parameters': f"k={k}, a={alpha}"
        }

    # 2. Nucleus Sampling (p-X)
    # wikitext_p-0.95_gpt2-large_256.jsonl  OR wikitext_p-0.95_gpt2-large.jsonl
    p_pattern = re.compile(r'wikitext_p-([\d.]+)_([^_]+(?:-[^_]+)*)(?:_\d+)?\.jsonl')
    match = p_pattern.match(filename)
    if match:
        p, model = match.groups()
        return {
            'Method': 'Nucleus',
            'Model': model,
            'p': float(p),
            'Parameters': f"p={p}"
        }

    # 3. Typical Sampling (typical-X)
    # wikitext_typical-0.95_gpt2-large_256.jsonl
    typical_pattern = re.compile(r'wikitext_typical-([\d.]+)_([^_]+(?:-[^_]+)*)(?:_\d+)?\.jsonl')
    match = typical_pattern.match(filename)
    if match:
        p, model = match.groups()
        return {
            'Method': 'Typical',
            'Model': model,
            'p': float(p),
            'Parameters': f"p={p}"
        }

    # 4. Greedy Search
    # wikitext_greedy_gpt2-large_256.jsonl
    greedy_pattern = re.compile(r'wikitext_greedy_([^_]+(?:-[^_]+)*)(?:_\d+)?\.jsonl')
    match = greedy_pattern.match(filename)
    if match:
        model = match.groups()[0]
        return {
            'Method': 'Greedy',
            'Model': model,
            'Parameters': "N/A"
        }

    return None

def main():
    directory = 'open_text_gen/wikitext_grid_search'
    print(f"Analyzing all results from: {directory} (Grid Search + Baselines)")

    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 240)

    # Find all .jsonl files
    jsonl_files = glob.glob(os.path.join(directory, '*.jsonl'))
    
    rows = []

    for f_path in jsonl_files:
        basename = os.path.basename(f_path)
        
        # Parse metadata from filename
        meta = parse_filename(basename)
        if not meta:
            # print(f"Skipping unrecognized file: {basename}")
            continue

        row_data = meta.copy()

        # 1. Load predictions & Compute Diversity (rep-n)
        predictions = load_predictions_from_jsonl(f_path)
        if predictions:
            div_metrics = calculate_correct_diversity_metrics(predictions)
            row_data.update(div_metrics)
        else:
            print(f"Warning: No predictions found in {basename}")

        # 2. Extract MAUVE and Gen Length from sidecar JSON files
        # Pattern: [basename_no_ext]_diversity_mauve_gen_length_result.json
        base_no_ext = os.path.splitext(f_path)[0]
        div_res_file = base_no_ext + '_diversity_mauve_gen_length_result.json'
        
        if os.path.exists(div_res_file):
            try:
                with open(div_res_file, 'r') as f:
                    data = json.load(f)[0]
                if 'mauve_dict' in data:
                    row_data['MAUVE'] = float(data['mauve_dict']['mauve_mean'])
                if 'gen_length_dict' in data:
                    row_data['Gen_Length'] = float(data['gen_length_dict']['gen_len_mean'])
            except Exception:
                pass

        # 3. Extract Coherence Scores
        # Pattern: [basename_no_ext]._opt-125m_coherence_result.json  (Note the dot)
        # We glob for all coherence files starting with this base name
        coh_glob_pattern = base_no_ext + '.*_coherence_result.json'
        found_coh_files = glob.glob(coh_glob_pattern)
        
        for c_file in found_coh_files:
            try:
                # Extract the coherence model name from the suffix
                # filename looks like: .../wikitext_p-0.95_gpt2. _opt-125m_coherence_result.json
                # suffix part: ._opt-125m_coherence_result.json
                suffix = c_file.replace(base_no_ext, '')
                # extract 'opt-125m' from '._opt-125m_coherence_result.json'
                # Remove starting '._' and ending '_coherence_result.json'
                coh_model_name = suffix.replace('._', '').replace('_coherence_result.json', '')
                coh_model_name = coh_model_name.replace('facebook/', '') # Clean up
                
                with open(c_file, 'r') as f:
                    data = json.load(f)[0]
                row_data[f'Coh_{coh_model_name}'] = float(data['coherence_mean'])
            except Exception:
                pass

        rows.append(row_data)

    if not rows:
        print("No valid result files found.")
        return

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns for clean display
    # Base cols
    base_cols = ['Model', 'Method', 'Parameters']
    
    # Dynamic cols
    coh_cols = sorted([c for c in df.columns if c.startswith('Coh_')])
    metric_cols = ['MAUVE', 'Gen_Length', 'rep-2', 'rep-3', 'rep-4']
    
    # Filter only columns that exist
    final_cols = base_cols + [c for c in metric_cols if c in df.columns] + coh_cols
    
    # Add any extra columns not in our list, just in case
    existing_others = [c for c in df.columns if c not in final_cols and c not in ['k', 'alpha', 'p']]
    final_cols += existing_others
    
    df = df[final_cols]
    
    # Sort: Model -> Method -> Parameters
    df = df.sort_values(by=['Model', 'Method', 'Parameters']).reset_index(drop=True)

    print("\n" + "="*140)
    print("Consolidated Results: Grid Search vs Baselines")
    print("Red = Best in Column | Blue = Best Row per Model | Green = Best Row Overall (based on MAUVE)")
    print("="*140)
    
    print_highlighted_table(df)
    print("="*140)

if __name__ == '__main__':
    main()