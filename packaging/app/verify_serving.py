"""Vérifie que l'API prédit EXACTEMENT comme le modèle scikit-learn du registre.

Usage (racine du repo, make pf actif) : uv run python packaging/app/verify_serving.py
"""

import json
import os
import sys
from pathlib import Path

import mlflow
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
sys.path.append(str(ROOT))
from app import serve  # noqa: E402
from src.dataset_config import load_config  # noqa: E402

cfg = load_config(ROOT / "dataset.yaml")
meta = json.loads((ROOT / "packaging/model/metadata.json").read_text())
X, _ = cfg.read_xy(ROOT / "data/processed/test.csv")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
ref = mlflow.sklearn.load_model(f"models:/{meta['model_name']}/{meta['model_version']}")
p_ref = ref.predict_proba(X)[:, 1]
rows = [{c: (None if v != v else v) for c, v in r.items()} for r in X.to_dict("records")]
p_api = serve.score(rows)

diff = float(np.abs(p_ref - p_api).max())
same = float(((p_ref >= serve.THRESHOLD) == (p_api >= serve.THRESHOLD)).mean())
print(f"{len(X)} lignes · écart max = {diff:.2e} · mêmes décisions : {same:.1%} · seuil {serve.THRESHOLD:.3f}")
assert diff <= 1e-12 and same == 1.0, "L'API ne prédit pas comme le modèle du registre !"
print("OK : l'API prédit exactement comme le modèle scikit-learn.")
