import os
import json
import glob
from tabulate import tabulate

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def get_value_from_nested(data, primary_key, secondary_key=None):
    """
    Extrait une valeur de data (dict ou list) de manière robuste.
    Gère le cas où data est une liste [dict].
    """
    # 1. Normalisation : on veut travailler sur un dictionnaire
    if isinstance(data, list):
        if len(data) == 0: return 'N/A'
        obj = data[0]
    elif isinstance(data, dict):
        obj = data
    else:
        return 'N/A'

    # 2. Récupération de la clé principale (ex: 'diversity_dict')
    val = obj.get(primary_key, 'N/A')
    
    # 3. Si c'est un sous-dictionnaire et qu'on cherche une clé secondaire
    if isinstance(val, dict) and secondary_key:
        return val.get(secondary_key, 'N/A')
    
    # 4. Si on a trouvé la valeur directe ou si ce n'était pas un dict
    return val

def main():
    base_dir = 'open_text_gen/wikitext_grid_search'
    # On cherche tous les fichiers contenant "beam-10"
    files = glob.glob(os.path.join(base_dir, '*beam-10*.json'))
    
    results = {}
    print(f"Fichiers trouvés : {len(files)}")
    
    # --- 1. CHARGEMENT ET REGROUPEMENT ---
    for f in files:
        filename = os.path.basename(f)
        
        # Nettoyage de la clé pour fusionner les fichiers
        if '_coherence_result.json' in filename:
            # On retire le suffixe Juge (.opt-2.7b...) pour avoir la clé commune
            base_key = filename.replace('_coherence_result.json', '')
            if '._opt' in base_key:
                base_key = base_key.split('._opt')[0]
            metric_type = 'coherence'
            
        elif '_diversity_mauve_gen_length_result.json' in filename:
            base_key = filename.replace('_diversity_mauve_gen_length_result.json', '')
            metric_type = 'div_mauve'
            
        elif '_perplexity_result.json' in filename:
            base_key = filename.replace('_perplexity_result.json', '')
            metric_type = 'ppl'
        else:
            continue
            
        if base_key not in results:
            results[base_key] = {}
            
        results[base_key][metric_type] = load_json(f)

    # --- 2. EXTRACTION DES DONNÉES ---
    table_data = []
    headers = ["Configuration", "Beam Size", "Gen Length", "Coherence", "Diversity", "MAUVE", "Perplexity"]
    
    for config, metrics in results.items():
        # Parsing du nom (ex: wikitext_beam-10_gpt2-xl_32)
        parts = os.path.basename(config).split('_')
        try:
            model = parts[2] # gpt2-xl
            length = parts[3]
        except:
            model = "Unknown"
            length = "?"

        # A. COHÉRENCE (Liste -> 'coherence_mean')
        raw_coh = metrics.get('coherence', [])
        coh = get_value_from_nested(raw_coh, 'coherence_mean')
        
        # B. DIVERSITÉ / MAUVE / LONGUEUR
        raw_div = metrics.get('div_mauve', [])
        
        # Vos scripts utilisent 'prediction_div_mean', 'mauve_mean', 'gen_len_mean'
        div = get_value_from_nested(raw_div, 'diversity_dict', 'prediction_div_mean')
        mauve = get_value_from_nested(raw_div, 'mauve_dict', 'mauve_mean')
        gen_len_val = get_value_from_nested(raw_div, 'gen_length_dict', 'gen_len_mean')
        
        if gen_len_val != 'N/A':
            length = gen_len_val

        # C. PERPLEXITÉ (Dict -> 'mean_perplexity')
        raw_ppl = metrics.get('ppl', {})
        ppl = get_value_from_nested(raw_ppl, 'mean_perplexity')
        if ppl == 'N/A':
            ppl = get_value_from_nested(raw_ppl, 'ppl')

        # Formatage des chiffres
        def fmt(x):
            try: return f"{float(x):.3f}"
            except: return str(x)

        table_data.append([
            f"{model} (Beam)", 
            "10", 
            fmt(length), # Affiche la longueur réelle mesurée
            fmt(coh), 
            fmt(div), 
            fmt(mauve), 
            fmt(ppl)
        ])

    print("\n" + "="*30 + " RÉSULTATS BEAM SEARCH " + "="*30)
    if not table_data:
        print("Aucune donnée trouvée.")
    else:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*80)

if __name__ == "__main__":
    main()