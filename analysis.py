
import json
import os
import glob
import pandas as pd
import re
from collections import defaultdict

# Configuration de l'affichage de pandas pour une meilleure lisibilité dans le terminal
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)

def parse_and_load_results(directory):
    """
    Parses filenames to extract hyperparameters and loads the result data.
    """
    results_data = defaultdict(dict)
    
    # Regex to capture params from diversity/mauve files
    # e.g., wikitext_k10_a0.4_e0.0_gpt2-xl_diversity_mauve_gen_length_result.json
    div_pattern = re.compile(
        r'wikitext_k(\d+)_a([\d.]+)_e([\d.]+)_([^_]+)_diversity_mauve_gen_length_result\.json'
    )
    
    # Regex to capture params from coherence files
    # e.g., wikitext_k5_a0.8_e0.0_gpt2-xl._opt-1.3b_coherence_result.json
    coh_pattern = re.compile(
        r'wikitext_k(\d+)_a([\d.]+)_e([\d.]+)_([^_]+)\._([^_]+)_coherence_result\.json'
    )

    all_files = glob.glob(os.path.join(directory, '*.json'), recursive=True)

    for f_path in all_files:
        basename = os.path.basename(f_path)
        
        div_match = div_pattern.match(basename)
        coh_match = coh_pattern.match(basename)

        try:
            with open(f_path, 'r') as f:
                data = json.load(f)[0]

            if div_match:
                k, alpha, epsilon, gen_model = div_match.groups()
                key = (gen_model, int(k), float(alpha), float(epsilon))
                
                results_data[key].update({
                    'MAUVE': float(data['mauve_dict']['mauve_mean']),
                    'Gen_Length': float(data['gen_length_dict']['gen_len_mean']),
                    'rep-2': float(data['diversity_dict']['rep_2']),
                    'rep-3': float(data['diversity_dict']['rep_3']),
                    'rep-4': float(data['diversity_dict']['rep_4']),
                })

            elif coh_match:
                k, alpha, epsilon, gen_model, coh_model = coh_match.groups()
                key = (gen_model, int(k), float(alpha), float(epsilon))

                # Add coherence score with the judge model name
                coh_key = f"Coherence_{coh_model.replace('facebook/', '')}"
                results_data[key][coh_key] = float(data['coherence_mean'])

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Warning: Could not process file {f_path}: {e}")
            continue

    return results_data

def main():
    """Main function to run the analysis."""
    directory = 'open_text_gen/wikitext_grid_search'
    print(f"Analyzing results from: {directory}")

    # Load and process all data
    results_map = parse_and_load_results(directory)

    if not results_map:
        print("No result files found or parsed. Exiting.")
        return

    # Convert the defaultdict to a list of records for DataFrame creation
    final_list = []
    for (gen_model, k, alpha, epsilon), metrics in results_map.items():
        record = {
            'Model': gen_model,
            'k': k,
            'alpha': alpha,
            'epsilon': epsilon,
            **metrics
        }
        final_list.append(record)

    # Create DataFrame
    df = pd.DataFrame(final_list)
    
    # Reorder columns for better readability
    cols = ['Model', 'k', 'alpha']
    # Dynamically add coherence columns and other metrics
    coh_cols = sorted([c for c in df.columns if c.startswith('Coherence')])
    other_metrics = ['MAUVE', 'Gen_Length', 'rep-2', 'rep-3', 'rep-4']
    
    # Filter out columns that might not be present if some files were missing
    final_cols = cols + [c for c in coh_cols + other_metrics if c in df.columns]
    
    df = df[final_cols]
    
    # Sort the results
    df = df.sort_values(by=['Model', 'k', 'alpha']).reset_index(drop=True)

    print("\n" + "="*100)
    print("Grid Search Results Analysis")
    print("="*100)
    print(df.to_string())
    print("="*100)

    # --- Highlights ---
    print("\n--- HIGHLIGHTS ---")
    
    # Identify the coherence column for ranking (use the first one found if multiple)
    if coh_cols:
        main_coh_col = coh_cols[0]
        # Best Coherence
        best_coherence = df.nlargest(3, main_coh_col)
        print(f"\n🏆 Top 3 for Coherence ({main_coh_col}):")
        display_cols = ['Model', 'k', 'alpha', main_coh_col]
        if 'MAUVE' in df.columns:
            display_cols.append('MAUVE')
        print(best_coherence[display_cols])
    
    # Best MAUVE
    if 'MAUVE' in df.columns:
        best_mauve = df.nlargest(3, 'MAUVE')
        print("\n🏆 Top 3 for MAUVE (Distribution similarity):")
        display_cols = ['Model', 'k', 'alpha', 'MAUVE']
        if coh_cols:
            display_cols.append(coh_cols[0])
        print(best_mauve[display_cols])
        
    # Lowest Repetition (good for diversity)
    if 'rep-3' in df.columns:
        lowest_rep3 = df.nsmallest(3, 'rep-3')
        print("\n🏆 Top 3 for Diversity (Lowest rep-3):")
        display_cols = ['Model', 'k', 'alpha', 'rep-3']
        if 'MAUVE' in df.columns:
            display_cols.append('MAUVE')
        print(lowest_rep3[display_cols])
        
    print("\nAnalysis complete.")

if __name__ == '__main__':
    main()
