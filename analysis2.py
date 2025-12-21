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
                predictions.append(item.get('gen_text') or item.get('generated') or "")
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line in {filepath}")
    return predictions

# --- Coloring & Printing Logic ---

def print_highlighted_table(df):
    """
    Prints the DataFrame with ANSI colors:
    - Red: Best result per column.
    - Green: Overall best row (Highest MAUVE).
    - Blue: Best row per model (Highest MAUVE).
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
    # MAUVE, Coh_*, Gen_Length -> Max (Higher is better/more quantity)
    # rep-* -> Min (Lower repetition is better)
    col_criteria = {
        'MAUVE': 'max',
        'Gen_Length': 'max',
        'rep-2': 'min',
        'rep-3': 'min',
        'rep-4': 'min'
    }
    # Add dynamic coherence columns
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

    # 2. Identify Best Rows (Green/Blue) based on MAUVE
    if 'MAUVE' in df.columns:
        sort_metric = 'MAUVE'
    else:
        # Fallback to Coherence if MAUVE missing
        available_coh = [c for c in df.columns if c.startswith('Coh_')]
        sort_metric = available_coh[0] if available_coh else None

    best_overall_idx = -1
    best_per_model_indices = []

    if sort_metric:
        # Best Overall (Green)
        best_overall_idx = df[sort_metric].idxmax()
        # Best per Model (Blue)
        # Group by Model and find index of max sort_metric
        # idxmax() might return nan if group is empty, handle carefully
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

    # Calculate Column Widths (based on text length, ignoring color codes)
    col_widths = {col: len(col) for col in headers}
    for idx, row in df.iterrows():
        for col in headers:
            val_str = formatters[col].format(row[col])
            col_widths[col] = max(col_widths[col], len(val_str))
    
    # Add padding
    for col in col_widths:
        col_widths[col] += 2

    # Print Header
    header_str = "".join([h.ljust(col_widths[h]) for h in headers])
    print(header_str)
    print("-" * len(header_str))

    # Print Rows
    for idx, row in df.iterrows():
        row_str = ""
        
        # Determine Row Color
        is_green = (idx == best_overall_idx)
        is_blue = (idx in best_per_model_indices)

        # Base color for the row text
        row_code = RESET
        if is_green:
            row_code = GREEN
        elif is_blue:
            row_code = BLUE
        
        for col in headers:
            val = row[col]
            val_str = formatters[col].format(val)
            
            # Determine if Cell is Red (Best in Column)
            is_red = False
            if col in best_values:
                # Use numpy isclose for float comparison
                if isinstance(val, float):
                    if np.isclose(val, best_values[col]):
                        is_red = True
                elif val == best_values[col]:
                    is_red = True
            
            # Apply color: Red overrides Row Color
            if is_red:
                color_code = RED
            else:
                color_code = row_code
            
            # Pad the plain string first to maintain alignment
            padded_str = val_str.ljust(col_widths[col])
            
            # Wrap in color codes
            # Note: We only color the text if it's not RESET, but we must apply color
            # around the padded string or inside? 
            # To avoid background color issues with padding, usually coloring the whole block is fine in terminal.
            if color_code != RESET:
                row_str += f"{color_code}{padded_str}{RESET}"
            else:
                row_str += padded_str
                
        print(row_str)


# --- Main Analysis Script ---

def main():
    directory = 'open_text_gen/wikitext_grid_search'
    print(f"Analyzing results from: {directory}")

    # Set up pandas display
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 220)

    # --- Step 1: Find all .jsonl generation files ---
    jsonl_files = glob.glob(os.path.join(directory, '*.jsonl'))
    
    # Regex to parse the .jsonl filename
    jsonl_pattern = re.compile(
        r'wikitext_k(\d+)_a([\d.]+)_e([\d.]+)_([^\.]+)\.jsonl'
    )
    
    results_data = defaultdict(dict)

    # --- Step 2: Process each .jsonl file for diversity and other metrics ---
    for f_path in jsonl_files:
        basename = os.path.basename(f_path)
        match = jsonl_pattern.match(basename)
        
        if not match:
            # print(f"Warning: Skipping file with unexpected name format: {basename}")
            continue
            
        k, alpha, epsilon, gen_model = match.groups()
        key = (gen_model, int(k), float(alpha))

        # Calculate correct diversity metrics
        predictions = load_predictions_from_jsonl(f_path)
        if predictions:
            diversity_metrics = calculate_correct_diversity_metrics(predictions)
            results_data[key].update(diversity_metrics)

        # Extract MAUVE / Gen_Length from older result files
        div_result_file = f_path.replace('.jsonl', '_diversity_mauve_gen_length_result.json')
        if os.path.exists(div_result_file):
            try:
                with open(div_result_file, 'r') as f:
                    data = json.load(f)[0]
                if 'mauve_dict' in data:
                    results_data[key]['MAUVE'] = float(data['mauve_dict']['mauve_mean'])
                if 'gen_length_dict' in data:
                     results_data[key]['Gen_Length'] = float(data['gen_length_dict']['gen_len_mean'])
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

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
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    # --- Step 4: Create DataFrame and Display ---
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
    print("Grid Search Results Analysis (Colored)")
    print("Red = Best in Column | Blue = Best Row per Model | Green = Best Row Overall")
    print("="*120)
    
    # Call the new print function instead of df.to_string()
    print_highlighted_table(df)
    
    print("="*120)
    print("\nAnalysis complete.")

if __name__ == '__main__':
    main()