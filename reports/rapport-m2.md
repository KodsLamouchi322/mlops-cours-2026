# Rapport comparatif — Kubeflow Pipelines vs Airflow (Lab 2)

**Auteur :** Prénom Nom · **Date :** 14 novembre 2026 · **Machine :** 12th Gen Intel(R) Core(TM) i7-12650H, 16 threads, 16 Go de RAM (WSL limité à 10 Go), Windows + WSL2 ·
**Dataset :** prompt-injection

## 1. Contexte

Le pipeline détecte les prompts malveillants (prompt injection, jailbreak) en 6 étapes : ingest, validate, prepare, train, evaluate, register. Il traite 2636 prompts issus de deux jeux Hugging Face. Chaque étape tourne dans un pod du cluster kind `mlops-lab`, avec la même image `scoring-pipeline:1.0.0` qui embarque `src/` et `dataset.yaml`. KFP et Airflow partagent donc le même code métier, `pipelines/stage.py`, l'image, le serveur MLflow et le registre `scoring-model`.

## 2. Résultats quantitatifs

| Mesure | KFP | Airflow |
|---|---|---|
| Installation de zéro (min) | non mesurable proprement (réseau lent, imports manuels d'images) | ≈ 32 (surtout téléchargement d'images) |
| Temps jusqu'au premier run réussi (min) | 56,9 (dont ~50 min d'`ImagePullBackOff`) | 1,3 (image déjà importée) |
| Durée d'un run complet (min) | 3,3 (sans cache) | 1,3 |
| Relance avec les mêmes paramètres (min) | 2,4 (3 étapes sur 6 en cache) | ≈ 1,3 (pas de cache natif, non remesuré) |
| Hot-reload (min) | non mesuré | non mesuré |
| Lignes de code propres à l'orchestrateur | 283 | 72 |
| Mémoire de l'outil (Go) | ≈ 1,2 (au repos, `crictl`) | ≈ 2,0 (`docker stats`, 7 conteneurs) |

## 3. Analyse qualitative

### Developer experience

KFP demande plus de code (283 lignes contre 72) parce que les artefacts typés (Dataset, Model, Metrics) s'écrivent explicitement, mais l'interface les montre ensuite : métriques, matrice de confusion, hash des données. Airflow ne transporte aucun fichier : `stage.py` recopie l'arborescence du run depuis et vers S3, et XCom ne porte que de petites valeurs. C'est plus court à écrire, mais le lineage est à reconstituer dans les logs et le préfixe S3.

### Scaling

Les deux outils lancent un pod par étape, et les ressources se règlent par étape (`set_memory_limit` pour KFP, `container_resources` pour Airflow). KFP gère retries, timeouts et cache dans la définition du pipeline. Airflow tourne ici en Docker Compose, hors du cluster, ce qui ne dit rien de son comportement en production. Je n'ai pas testé le parallélisme.

### Debugging

Dans les deux cas les logs sont dans l'interface. Mon pire incident a touché KFP : une image introuvable laissait le run en `RUNNING` pendant 30 minutes, sans alerte, jusqu'au timeout de mon script. Il a fallu lire `kubectl describe pod`. Chez Airflow, `list-import-errors` signale directement un DAG cassé.

## 4. Recommandation

Je choisirais KFP pour un projet centré sur le ML, quand le lineage, les artefacts typés et le cache comptent, et qu'un cluster Kubernetes existe déjà. Je choisirais Airflow pour des pipelines de données variés, planifiés, ou quand l'équipe le connaît déjà. Dans les deux cas, il faut une image du pipeline accessible à tous les nœuds, sinon le premier run échoue.

## 5. Limites de la comparaison

Un seul petit dataset, un seul poste, un seul run par mesure. Airflow est passé après KFP, avec l'image déjà importée dans les nœuds : son avantage de vitesse est en partie un biais d'ordre. La mémoire de KFP est mesurée au repos, celle d'Airflow en conteneurs Compose, donc les deux ne sont pas strictement comparables. Je n'ai pas testé le `KubernetesExecutor`, ni un vrai cluster managé, ni le parallélisme, ni le hot-reload.
