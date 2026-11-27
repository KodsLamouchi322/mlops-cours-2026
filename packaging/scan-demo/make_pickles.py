"""Étape 7.2 — fabrique deux pickles pour la démo ModelScan. NE PAS LES CHARGER.

- model.pkl     : le modèle sérialisé avec pickle ;
- malicious.pkl : le même modèle + une charge cachée. Au chargement (pickle.load,
                  joblib.load), elle exécuterait une commande shell : ici un simple
                  `echo` inoffensif, mais ce pourrait être n'importe quoi.

Usage (racine du repo) : uv run python packaging/scan-demo/make_pickles.py
"""

import json
import os
import pickle
from pathlib import Path

import mlflow

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

meta = json.loads((ROOT / "packaging/model/metadata.json").read_text())
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
_local_model_dir = ROOT / "models" / "model"
_use_mlflow = os.getenv("USE_MLFLOW", "auto")
if _use_mlflow == "auto":
    import socket
    try:
        with socket.create_connection(("localhost", 5000), timeout=3):
            _use_mlflow = "yes"
    except OSError:
        _use_mlflow = "no"

if _use_mlflow == "yes":
    model = mlflow.sklearn.load_model(f"models:/{meta['model_name']}/{meta['model_version']}")
else:
    model = mlflow.sklearn.load_model(str(_local_model_dir))


class Payload:
    """__reduce__ dit à pickle comment « reconstruire » l'objet : ici, appeler os.system."""

    def __reduce__(self):
        return (os.system, ("echo '[pickle piégé] ce code vient de s exécuter'",))


with open(HERE / "model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(HERE / "malicious.pkl", "wb") as f:
    pickle.dump({"model": model, "metadata": Payload()}, f)
print(f"Written: {HERE.relative_to(ROOT)}/model.pkl, {HERE.relative_to(ROOT)}/malicious.pkl")
