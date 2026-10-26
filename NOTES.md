# Notes du projet : assistant DevOps

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
- Feast : `cd features && feast apply`, puis
  `feast materialize-incremental "$(date -u +%Y-%m-%dT%H:%M:%S)"`.
- Dataset : NLBSE'24 (issues GitHub), dépôt archivé ; le remote DVC sert de sauvegarde.
