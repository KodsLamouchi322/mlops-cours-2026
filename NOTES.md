# Notes du projet : PromptShield (titre provisoire)

Garde-fou MLOps pour LLM : un classifieur de texte détecte si un prompt est une attaque
(prompt injection ou jailbreak) ou un message normal. Coût métier : laisser passer une
attaque (faux négatif) coûte 5 fois plus cher que bloquer à tort un message normal.

## Données

Deux jeux Hugging Face fusionnés par `src/download.py`, chacun épinglé sur un commit précis
(le téléchargement donne le même résultat dans le temps) :

| Source | Commit | Licence (page du dataset) |
|---|---|---|
| jackhhao/jailbreak-classification | `2f2ceeb39658696fd3f462403562b6eea5306287` | apache-2.0 |
| deepset/prompt-injections | `4f61ecb038e9c3fb77e21034b22511b523772cdd` | apache-2.0 |

- 2636 prompts après retrait de 24 doublons, environ 35 % d'attaques.
- Colonnes : `prompt`, `label` (attack / benign), `source`, `prompt_len`, `n_words`.
- `source` est déclarée « sensible » dans `dataset.yaml` : mesurée, jamais apprise.
- Le jeu deepset mélange anglais et allemand ; certaines étiquettes sont discutables.
- Contrat `dataset.yaml` : clé `features.text` ajoutée au contrat standard du cours
  (TF-IDF dans `src/dataset_config.py`), à faire valider par l'enseignant.

## Résultats (Lab 1)

- Modèle de référence : random forest, 400 arbres, ROC AUC 0.987, coût métier 77,
  seuil de décision 0.274 (calculé depuis la matrice de coûts).
- Seuil de promotion : ROC AUC 0.95, placé entre les forêts peu profondes (0.92) et les
  meilleurs modèles (0.98).
- 25 runs comparés (dummy, régression logistique, random forest, gradient boosting).

## Limites connues

- Les deux sources sont de styles très différents (longueur médiane : 42 à 126 caractères
  pour deepset, 234 à 1566 pour jailbreak).
- Test de généralisation (régression logistique sur le texte, pas le modèle final) :
  entraînement sur jailbreak puis test sur deepset : 0.70 ; sens inverse : 0.89.
  Le 0.987 vaut pour des prompts proches de ceux d'entraînement.
- Ancien sujet (issues GitHub, NLBSE'24) : conservé dans le tag Git `v1-nlbse-issues`.

## Reproduire le projet (Lab 1)

- Cloner le repo, puis `make up`, `make pf`, `uv sync` et `source .venv/bin/activate`.
- Charger l'environnement : `set -a && source .env && set +a`.
- Identifiants du remote DVC (jamais commités) :
  `dvc remote modify --local storage access_key_id mlops` et
  `dvc remote modify --local storage secret_access_key mlops-lab-secret`.
- Ajouter `127.0.0.1 s3.storage` dans `/etc/hosts` (liens présignés MLflow) :
  `echo "127.0.0.1 s3.storage" | sudo tee -a /etc/hosts`.
- Même cluster (remote déjà rempli) : `dvc pull` puis `dvc repro` (tout est « skipping »).
- Autre cluster (remote vide) : `dvc repro` puis `dvc push` ; ne pas commiter le `dvc.lock`
  régénéré (seul le hash du modèle change).
- Promotion du champion : `python -m src.promote --filter "tags.launched_by = 'dvc' and tags.dataset = 'prompt-injection'"`.
- Feast : `mkdir -p features/data`, puis `cd features && feast apply`, puis
  `feast materialize-incremental "$(date -u +%Y-%m-%dT%H:%M:%S)"`.
- Vérification : `bash labs/m1-versioning/check.sh` (18/18 attendu).
