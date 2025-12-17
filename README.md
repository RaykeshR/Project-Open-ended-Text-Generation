# Project: Open-ended Text Generation

[![My Skills](https://skillicons.dev/icons?i=py)](https://www.python.org/)
[![My Skills](https://skillicons.dev/icons?i=tensorflow)](https://www.tensorflow.org/?hl=fr)
[![My Skills](https://skillicons.dev/icons?i=github)](https://github.com/RaykeshR/PFE-Roguelike)
[![My Skills](https://skillicons.dev/icons?i=git)](https://git-scm.com/)
[![My Skills](https://skillicons.dev/icons?i=bash)](https://fr.wikibooks.org/wiki/Programmation_Bash/Scripts)
[![My Skills](https://skillicons.dev/icons?i=md)](https://docs.github.com/fr/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
[![My Skills](https://skillicons.dev/icons?i=vscode)](https://code.visualstudio.com/)
[![My Skills](https://skillicons.dev/icons?i=powershell)](https://learn.microsoft.com/fr-fr/powershell/scripting/overview?view=powershell-7.4)
[![My Skills](https://skillicons.dev/icons?i=windows)](https://www.microsoft.com/fr-fr/windows?r=1)

<!-- Le site est accessible via ce [Lien qui fait une Redirection d'URL](https://raykeshr.github.io/PFE-Roguelike/) vers une page d'accueil pour le site du Github : PFE-Roguelike -->
Rapport : [Fichier Word](https://reseaueseo-my.sharepoint.com/:w:/r/personal/lea_ludet_reseau_eseo_fr/Documents/Rapport_Projet_NLP.docx?d=w0ac3619c8d6b469e8e2f4897154e7524&csf=1&web=1&e=r4mTnR) 


#### Sommaire 

TODO :octocat: :neckbeard: :bowtie: :shipit:

## Generation

```py
python -m open_text_gen.generate --alphas 0.2 0.5 0.8 --dataset_name wikitext --output_dir open_text_gen/wikitext
python -m open_text_gen.generate --alphas 0.2 0.5 0.8 --dataset_name wikitext --output_dir open_text_gen/wikitext --num_prefixes 5
```

## Evaluation

### Coherence

```bash
python open_text_gen/compute_coherence.py --opt_model_name facebook/opt-2.7b --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.2_gpt2-xl_256.jsonl
python open_text_gen/compute_coherence.py --opt_model_name facebook/opt-2.7b --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.5_gpt2-xl_256.jsonl
python open_text_gen/compute_coherence.py --opt_model_name facebook/opt-2.7b --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.8_gpt2-xl_256.jsonl
```

### Diversity, MAUVE, and Generation Length

```bash
python open_text_gen/measure_diversity_mauve_gen_length.py --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.2_gpt2-xl_256.jsonl
python open_text_gen/measure_diversity_mauve_gen_length.py --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.5_gpt2-xl_256.jsonl
python open_text_gen/measure_diversity_mauve_gen_length.py --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.8_gpt2-xl_256.jsonl
```



Téléchargement des Données/ Model
```CMD
%USERPROFILE%\.cache\huggingface
```

### Mise en place (Windows):

<!-- <details open> -->
<details>
<summary>Création d'environnements virtuels : </summary>

## Un package manquant : 
<!-- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -->
```
.venv\Scripts\activate && python -m pip install --upgrade pip && python -m pip install -r requirements.txt
```
## Version 1 ligne/rapide : 
```py
python -m venv .venv && .venv\Scripts\activate && python -m pip install --upgrade pip && python -m pip install -r requirements.txt && pip freeze > requirements.txt
```
(en cas de bug faire les étapes ci-dessous ou essayer sur CMD ou powershell>=7.0)
```$env:SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL='True'; ./.venv/Scripts/python.exe -m pip install -r requirements.txt```

### 1. Cloner le Repo

avec GitHub (Copie les fichiers localement)

### 2. `python -m venv .venv`

peut nécessiter le passage par CMD (Crée le Dossier .venv)

### 3. `.venv\Scripts\activate`
 

Créer un environnement virtuel Python (Sur Linux/Mac) :
```bash
source venv/bin/activate  # Sur Linux/Mac
```

Lancer avec le CMD peut éviter les erreurs. (Lance l'environnement virtuel)
EN ADMIN : `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned` en cas d'erreur
([détail](https://tutorial.djangogirls.org/fr/django_installation/))

Résultat : 

$\color{rgba(100,255,100, 0.75)}{\textsf{(.venv)}}$ PS C:\Users...\Portfolio_Django> |

On peut aussi (Si c'est un problème de l'éditeur) `$ . .venv\Scripts\activate.ps1`
(lance l'environnement virtuel)

### 4. `python -m pip install --upgrade pip`

(met à jour pip)

### 5. `python -m pip install -r requirements.txt`

```pip freeze > requirements.txt``` pour remplir automatiquement les requirements

Pour toutes les étapes précédentes (sur CMD ou powershell>=7) : 


```
python -m venv .venv && .venv\Scripts\activate && python -m pip install --upgrade pip && python -m pip install -r requirements.txt
```

en cas d'erreur (supprimer le dossier .venv ou lancer): 

```.venv\Scripts\activate && python -m pip install --upgrade pip && python -m pip install -r requirements.txt```

Avec  pip freeze  :

    Pour toutes les étapes précédentes (sur CMD ou powershell>=7) : 
    ```python -m venv .venv && .venv\Scripts\activate && python -m pip install --upgrade pip && python -m pip install -r requirements.txt && pip freeze > requirements.txt```
    
    en cas d'erreur (supprimer le dossier .venv ou lancer): 
    ```.venv\Scripts\activate && python -m pip install --upgrade pip && python -m pip install -r requirements.txt && pip freeze > requirements.txt```
    
### 6. Modifier .git\info\exclude 

Ajouter : `.venv`
(Ne prend pas en compte la modification du dossier .venv)

### 7. Lancer le fichier main.py 

Commande : `python main.py `
(Lance le fichier principal avec python)

### Linux/Mac :

"""
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
"""
Modifier .git\info\exclude 

</details> 


<details>
<summary>Nettoyer un dépôt git : </summary>
Télécharger BFG Repo-Cleaner sur le site (.jar): 
https://rtyley.github.io/bfg-repo-cleaner/

Lancer : 

git clone --mirror https://github.com/RaykeshR/PFE-Roguelike.git
cd PFE-Roguelike.git
<!-- java -jar ../bfg-1.15.0.jar --delete-files database/.env -->
java -jar ../bfg-1.15.0.jar --delete-files .env

git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force --all
git push --force --tags

<!-- git push --force origin Dev -->

<!-- git push --force origin --all
git push --force origin --tags -->

git push --mirror

</details> 

<br>

<details>
<summary>Autre : </summary>
<details>
<summary>Gemini-cli : </summary>
1. ouvrir un terminal (WSL, ...)
2. taper : `npm install -g @google/gemini-cli` / `sudo npm install -g @google/gemini-cli`
3. Changer de dossier : `cd .../PFE-Roguelike`
4. lancer gemini : avec `gemini`
5. login avec google
6. tester avec une question
7. lancer la commande : `/init`
    
</details> 
<details>
<summary>Gemini-cli + MCP Github : </summary>

Le Model Context Protocol (MCP) est un protocole standard ouvert conçu pour connecter des modèles d'intelligence artificielle (IA) (LLM, ...)
Ici, le MCP Github permettra à gemini d'accéder aux code source sans passer par une recherche web à chaque fois.

1. ouvrir un terminal (WSL, ...)
2. taper : `cd ~/.gemini`
3. modifier le fichier settings.json et ajouter au json : 
```
    , 
    "mcpServers": {
        "github": {
            "httpUrl": "https://api.githubcopilot.com/mcp/",
            "headers": {
                    "Authorization": "Bearer ghp_..."
                },
                "timeout": 5000
        }
    }

```    

> [!NOTE]  
> Pour obtenir le "Bearer ghp_..." ou plus précisément le `ghp_...` il faut mettre un PAT ( Personal access tokens (classic) : [https://github.com/settings/tokens](https://github.com/settings/tokens) ) nommé de préférence "Gemini MCP" avec les droits voulus.

</details> 

</details> 