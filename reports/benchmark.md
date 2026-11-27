# Benchmark de latence — prompt-injection, modèle v10

1000 appels par mesure, après 10 appels de chauffe · x86_64 · Python 3.12.3

| Backend | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|
| sklearn_pipeline (1 ligne) | 30.281 | 96.974 | 319.961 |
| sklearn_batch_100 | 50.088 | 136.193 | 372.061 |
| onnx_python (1 ligne, non servi) | 0.085 | 0.174 | 0.333 |
| onnx_batch_100 (non servi) | 8.790 | 36.569 | 71.471 |
| api_http_sklearn (1 ligne) | 32.386 | 71.499 | 197.177 |

Accélération ONNX vs scikit-learn (P50, 1 ligne) : x357.3

L'ONNX n'est pas servi : sa conversion du TF-IDF change des décisions (5 sur 528).
