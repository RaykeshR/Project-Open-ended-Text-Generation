import os
import json
import pandas as pd
import glob

def analyze_epsilon_results(results_dir='open_text_gen/wikitext_epsilon_grid_search'):
    # 1. Identifier les fichiers
    print(f"🔍 Recherche des fichiers dans : {results_dir}")
    all_json_files = glob.glob(os.path.join(results_dir, '*.json'))
    
    # On filtre pour ne garder que ceux qui nous intéressent
    # Note: On inclut 'resuult' au cas où la typo persiste, et 'diversity'
    div_files = [f for f in all_json_files if ('diversity' in f or 'resuult' in f) and 'coherence' not in f]
    coh_files = [f for f in all_json_files if 'coherence' in f]
    
    if not div_files:
        print("⚠️ Aucun fichier de métriques (diversity/mauve) trouvé.")
        return

    data_store = {}

    # 2. Fonction pour parser les noms de fichiers
    def get_params(filename):
        basename = os.path.basename(filename)
        k = 5 # Valeur par défaut
        alpha = None
        
        parts = basename.replace('-', '_').split('_')
        for part in parts:
            if part.startswith('k') and part[1:].isdigit():
                try: k = int(part[1:])
                except: pass
            if part.startswith('alpha'):
                try:
                    val = part.replace('alpha', '')
                    alpha = float(val)
                except:
                    pass
        return k, alpha

    # 3. Chargement des métriques (Diversité, MAUVE, Longueur)
    for f in div_files:
        k, alpha = get_params(f)
        if alpha is not None:
            key = (k, alpha)
            if key not in data_store: data_store[key] = {'k': k, 'alpha': alpha}
            
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    content = json.load(file)
                    
                    # Gestion liste vs dict
                    if isinstance(content, list):
                        content = content[0] if len(content) > 0 else {}
                    
                    # --- EXTRACTION ROBUSTE (Adaptée à votre JSON) ---
                    
                    # A. Longueur
                    if 'gen_length_dict' in content and 'gen_len_mean' in content['gen_length_dict']:
                        data_store[key]['gen_length'] = content['gen_length_dict']['gen_len_mean']
                    elif 'gen_length' in content:
                        data_store[key]['gen_length'] = content['gen_length']
                    
                    # B. Diversité
                    if 'diversity_dict' in content and 'prediction_div_mean' in content['diversity_dict']:
                        data_store[key]['diversity'] = content['diversity_dict']['prediction_div_mean']
                    elif 'diversity' in content:
                        data_store[key]['diversity'] = content['diversity']
                        
                    # C. MAUVE
                    if 'mauve_dict' in content and 'mauve_mean' in content['mauve_dict']:
                        data_store[key]['mauve'] = content['mauve_dict']['mauve_mean']
                    elif 'mauve' in content:
                        data_store[key]['mauve'] = content['mauve']

            except Exception as e:
                print(f"Erreur lecture {f}: {e}")

    # 4. Chargement des scores de Cohérence
    for f in coh_files:
        k, alpha = get_params(f)
        if alpha is not None:
            key = (k, alpha)
            if key in data_store:
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        content = json.load(file)
                        if isinstance(content, list): content = content[0] if len(content) > 0 else {}
                        
                        # Extraction Cohérence
                        if 'coherence_score' in content:
                            data_store[key]['coherence_score'] = content['coherence_score']
                        elif 'mean_score' in content:
                            data_store[key]['coherence_score'] = content['mean_score']
                            
                except Exception as e:
                    print(f"Erreur lecture {f}: {e}")

    # 5. Création et Affichage du DataFrame
    df = pd.DataFrame(list(data_store.values()))
    
    # Colonnes à afficher
    target_cols = ['k', 'alpha', 'gen_length', 'coherence_score', 'diversity', 'mauve']
    final_cols = [c for c in target_cols if c in df.columns]
    
    if not df.empty:
        # Conversion numérique
        for col in final_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df[final_cols].sort_values(by='alpha')
        
        print("\n" + "="*80)
        print("📊 RÉSULTATS : EPSILON GREEDY SEARCH")
        print("="*80)
        print(df.to_string(index=False, float_format="%.2f"))
        print("="*80)
        
        # Petit commentaire automatique
        print("\n💡 Analyse rapide :")
        for idx, row in df.iterrows():
            l = row.get('gen_length', 0)
            d = row.get('diversity', 0)
            print(f"- Alpha {row['alpha']:.1f} (k={int(row['k'])}) : Longueur ~{l:.0f} mots, Diversité {d:.1f}%")
            
    else:
        print("Aucune donnée extraite. Vérifiez les noms de fichiers et le parsing.")

if __name__ == "__main__":
    analyze_epsilon_results()