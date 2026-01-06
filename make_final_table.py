import os
import json
import glob
from tabulate import tabulate

def load_json(path):
    if not os.path.exists(path): return {}
    with open(path, 'r', encoding='utf-8') as f:
        try: return json.load(f)
        except: return {}

def get_val(data, *keys):
    """Récupère une valeur profonde dans une structure dict/list"""
    curr = data
    if isinstance(curr, list) and curr: curr = curr[0]
    for k in keys:
        if isinstance(curr, dict):
            curr = curr.get(k, {})
        else:
            return 'N/A'
    return curr if not isinstance(curr, (dict, list)) else 'N/A'

def main():
    root_dirs = [
        'open_text_gen/wikitext_grid_search',
        'open_text_gen/wikitext_epsilon_grid_search'
    ]
    
    data_map = {} # Key -> {simcse: ..., mauve: ..., ppl: ...}

    # 1. SCAN DES FICHIERS
    for d in root_dirs:
        if not os.path.exists(d): continue
        files = glob.glob(os.path.join(d, '*.json'))
        
        for f in files:
            fname = os.path.basename(f)
            
            # Détection du type de fichier et de la clé unique (Config)
            key = None
            m_type = None
            
            if '_simcse_result.json' in fname:
                key = fname.replace('_simcse_result.json', '')
                m_type = 'simcse'
            elif '_diversity_mauve_gen_length_result.json' in fname:
                key = fname.replace('_diversity_mauve_gen_length_result.json', '')
                m_type = 'div_mauve'
            elif '_perplexity_result.json' in fname:
                key = fname.replace('_perplexity_result.json', '')
                m_type = 'ppl'
            # (On ignore les anciens fichiers de cohérence OPT)
            
            if key and m_type:
                if key not in data_map: data_map[key] = {}
                data_map[key][m_type] = load_json(f)

    # 2. CONSTRUCTION DU TABLEAU
    rows = []
    headers = ["Method/Model", "SimCSE (Coh)", "Diversity", "MAUVE", "PPL", "Len"]
    
    # Ordre de tri pour que ce soit joli (Baselines d'abord, puis Epsilon)
    sorted_keys = sorted(data_map.keys())
    
    for key in sorted_keys:
        metrics = data_map[key]
        
        # Extraction SimCSE
        simcse_val = metrics.get('simcse', {}).get('coherence_mean', 'N/A')
        
        # Extraction Div/Mauve/Len (Attention aux clés imbriquées de vos scripts)
        dm = metrics.get('div_mauve', {})
        div = get_val(dm, 'diversity_dict', 'prediction_div_mean')
        mauve = get_val(dm, 'mauve_dict', 'mauve_mean')
        length = get_val(dm, 'gen_length_dict', 'gen_len_mean')
        
        # Extraction PPL
        ppl = get_val(metrics.get('ppl', {}), 'mean_perplexity')
        if ppl == 'N/A': ppl = get_val(metrics.get('ppl', {}), 'ppl')

        # Nettoyage du nom pour l'affichage
        display_name = key.replace('wikitext_', '')
        
        # Formatage chiffres
        def fmt(x): 
            try: return f"{float(x):.3f}" 
            except: return "N/A"

        rows.append([
            display_name,
            fmt(simcse_val),
            fmt(div),
            fmt(mauve),
            fmt(ppl),
            fmt(length)
        ])

    # 3. GÉNÉRATION MARKDOWN
    print("\n### Comparaison des Performances (SimCSE Updated)\n")
    print(tabulate(rows, headers=headers, tablefmt="github"))

if __name__ == "__main__":
    main()