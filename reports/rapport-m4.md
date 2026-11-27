# Rapport — Lab 4 : packaging du modèle

Dataset : prompt-injection · Modèle : `scoring-model` version 10 (random forest 400 arbres,
TF-IDF sur le prompt) · Seuil de décision : 0,274

## 1. Comparatif des tailles

| Image | Taille compressée (`docker image inspect`) | Taille décompressée (`docker images`) | CVE HIGH + CRITICAL (Trivy) |
|---|---|---|---|
| `scoring-api:naive` | à compléter | 2,46 Go | 545 (529 HIGH, 16 CRITICAL) |
| `scoring-api:optimized` | 160 Mo | 721 Mo | 55 (55 HIGH, 0 CRITICAL) |
| `scoring-api:bentoml` | 370 Mo | à compléter | 98 (98 HIGH, 0 CRITICAL) |

Docker utilise ici le stockage containerd : `docker images` affiche l'espace décompressé,
`docker image inspect` la taille compressée, que le critère de 500 Mo mesure.
L'image naïve embarque `.env` (secrets S3) et tourne en root.

## 2. Comparatif des latences

(Coller ici le tableau de `reports/benchmark.md`.)
Mesures sur un seul poste, avec le cluster actif : ordre de grandeur, pas performance de production.

## 3. Interprétation

(Pourquoi l'image est plus petite ; pourquoi ONNX est plus rapide ; ce que montrent les scans.)

## 4. Choix recommandé

(Servir le pipeline scikit-learn, et pourquoi : 5 décisions sur 528 changent avec l'ONNX.)

## 5. Pistes d'amélioration

(À toi : base d'image plus récente pour les CVE Debian, tokenisation convertible, etc.)
