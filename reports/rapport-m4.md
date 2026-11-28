# Rapport — Lab 4 : packaging du modèle

Dataset : prompt-injection · Modèle : `scoring-model` version 10 (random forest de 400 arbres,
TF-IDF sur le prompt) · Seuil de décision : 0,274

## 1. Comparatif des tailles

| Image | Taille compressée (`docker image inspect`) | Taille décompressée (`docker images`) | CVE HIGH + CRITICAL (Trivy) |
|---|---|---|---|
| `scoring-api:naive` | 631 Mo | 2,46 Go | 545 (529 HIGH, 16 CRITICAL) |
| `scoring-api:optimized` | 160 Mo | 721 Mo | 55 (55 HIGH, 0 CRITICAL) |
| `scoring-api:bentoml` | 370 Mo | 1,46 Go | 98 (98 HIGH, 0 CRITICAL) |

Docker utilise ici le stockage containerd : `docker images` affiche l'espace décompressé,
`docker image inspect` la taille compressée, que le critère de 500 Mo mesure. L'image naïve
copie tout le dépôt (`COPY . /app`), donc elle embarque aussi `.env` (identifiants S3), et elle
tourne en root.

## 2. Comparatif des latences

| Backend | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|
| sklearn_pipeline (1 ligne) | 30,281 | 96,974 | 319,961 |
| sklearn_batch_100 | 50,088 | 136,193 | 372,061 |
| onnx_python (1 ligne, non servi) | 0,085 | 0,174 | 0,333 |
| onnx_batch_100 (non servi) | 8,790 | 36,569 | 71,471 |
| api_http_sklearn (1 ligne) | 32,386 | 71,499 | 197,177 |

Accélération ONNX contre scikit-learn (P50, une ligne) : x357. Mesures faites sur un seul poste,
avec le cluster Kubernetes et plusieurs conteneurs actifs en arrière-plan : ce sont des ordres de
grandeur, pas des performances de production.

## 3. Interprétation

**Pourquoi l'image optimisée est plus petite.** Elle part de `python:3.12-slim` et utilise un
build multi-stage : seules les dépendances installées sont copiées, sans compilateur ni cache pip.
Une liste blanche dans le `.dockerignore` empêche `.env`, `.git` et `data/` d'entrer dans
l'image. Résultat : 721 Mo sur disque contre 2,46 Go pour l'image naïve.

**Pourquoi ONNX est plus rapide.** 0,085 ms contre 30,3 ms, soit x357. Avec scikit-learn, chaque
appel paie un coût fixe (validation du DataFrame, un appel Python par transformateur). ONNX
Runtime exécute un seul graphe. Un lot de 100 lignes coûte 50 ms avec scikit-learn contre 8,8 ms
avec ONNX : l'écart se réduit quand le coût fixe est partagé. L'API HTTP ajoute peu à
scikit-learn (32,4 ms contre 30,3 ms).

**Ce que montrent les scans.** Trivy trouve 55 vulnérabilités HIGH dans l'image servie, 98 dans
l'image BentoML et 545 dans l'image naïve, dont 16 CRITICAL. Les paquets les plus touchés de
l'image servie sont des paquets Debian de l'image de base (zlib1g, util-linux, openssl,
perl-base) : Trivy indique qu'un correctif existe, mais je n'ai pas reconstruit l'image sur une
base plus récente. ModelScan ne lit ni l'ONNX ni le skops. L'ONNX est sûr par construction (un
graphe sans code), et skops refuse tout type absent de la liste de confiance que je lui donne
(`numpy.dtype` et `sklearn.tree._tree.Tree`). ModelScan détecte en revanche le pickle piégé
(`posix.system`).

## 4. Choix recommandé

J'ai choisi de servir le pipeline scikit-learn et non l'ONNX. La conversion du TF-IDF par
skl2onnx n'est pas fidèle : le graphe ONNX tokenise avec `[a-zA-Z0-9_]+` et garde les mots d'une
lettre, alors que scikit-learn utilise `\w\w+`. Sur les 528 prompts de test, 5 décisions changent
au seuil de 0,274 et les probabilités s'écartent jusqu'à 0,175. Pour un garde-fou, laisser passer
une attaque coûte 5 fois plus cher qu'un faux blocage : je préfère garantir que l'API prédit
exactement comme le modèle validé. J'ai vérifié cette équivalence sur les 528 lignes (écart
maximal nul, 100 % de décisions identiques). Le coût de ce choix est d'environ 30 ms par requête,
avec un P99 à 320 ms. L'image fait 160 Mo compressés, sous la limite de 500 Mo.

## 5. Pistes d'amélioration

- Reconstruire l'image sur une image de base Debian plus récente pour corriger les CVE de la base.
- Réentraîner avec un TF-IDF dont la tokenisation est reproductible en ONNX (par exemple
  `token_pattern` à `[a-zA-Z0-9_]+`, mots d'une lettre gardés) : l'ONNX pourrait alors être servi
  sans changer les décisions. Le modèle changerait, donc il faudrait revalider son score.
- Réduire la forêt (30 Mo, 400 arbres sans profondeur limite) pour gagner en taille et en
  latence, en mesurant l'effet sur le ROC AUC.
- Importer l'image dans les nœuds du cluster (containerd) avant le déploiement du Lab 5.
