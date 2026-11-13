"""Lance 22 runs d'entraînement (un sous-processus par run) dans l'expérience « scoring ».

Robustesse : un run qui échoue (par ex. port-forward MLflow coupé) est relancé jusqu'à
RETRIES fois. GRID_START=<n> reprend à partir du run n (ex. GRID_START=12).
"""

import os
import subprocess
import sys
import time
from itertools import product

RETRIES = 3
PAUSE_S = 5

CONFIGS = [
    # Référence naïve : tout modèle utile doit faire mieux (ROC AUC = 0,5)
    ("dummy", []),
    # Régression logistique : 4 x 2 = 8 runs
    *[
        ("logistic_regression", ["--C", str(c), "--class_weight", w])
        for c, w in product([0.01, 0.1, 1.0, 10.0], ["none", "balanced"])
    ],
    # Random Forest : 3 x 3 = 9 runs
    *[
        ("random_forest", ["--n_estimators", str(n), "--max_depth", str(d)])
        for n, d in product([100, 200, 400], [4, 8, "none"])
    ],
    # Gradient Boosting : 2 x 2 = 4 runs
    *[
        ("gradient_boosting", ["--n_estimators", str(n), "--max_depth", str(d)])
        for n, d in product([100, 200], [2, 3])
    ],
]


def main() -> None:
    start = int(os.getenv("GRID_START", "1"))
    print(f"Lancement de {len(CONFIGS)} runs (à partir du run {start})...")
    env = {**os.environ, "LAUNCHED_BY": "grid_search"}  # tag MLflow launched_by
    for i, (model, args) in enumerate(CONFIGS, 1):
        if i < start:
            continue
        cmd = [sys.executable, "-m", "src.train", "--model", model, *args]
        for attempt in range(1, RETRIES + 1):
            print(f"[{i}/{len(CONFIGS)}] {' '.join(cmd[1:])} (essai {attempt})", flush=True)
            result = subprocess.run(cmd, env=env)
            if result.returncode == 0:
                break
            if attempt == RETRIES:
                raise SystemExit(f"Run {i} en échec après {RETRIES} essais : reprendre avec GRID_START={i}")
            time.sleep(PAUSE_S)
    print("✓ Tous les runs sont terminés.")


if __name__ == "__main__":
    main()
