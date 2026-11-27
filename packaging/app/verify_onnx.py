"""Étape 1.3 — vérifie que le modèle ONNX prédit comme le modèle scikit-learn.

Usage (racine du repo) : uv run python packaging/app/verify_onnx.py
"""

import json
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from src.dataset_config import load_config  # noqa: E402

TOL = 1e-2  # TF-IDF + RandomForest : légères divergences float32 tolérées

cfg = load_config(ROOT / "dataset.yaml")
meta = json.loads((ROOT / "packaging/model/metadata.json").read_text())
X, _ = cfg.read_xy(ROOT / "data/processed/test.csv")

_local_model_dir = ROOT / "models" / "model"
_use_mlflow = os.getenv("USE_MLFLOW", "auto")
if _use_mlflow == "auto":
    import socket
    try:
        with socket.create_connection(("localhost", 5000), timeout=3):
            _use_mlflow = "yes"
    except OSError:
        _use_mlflow = "no"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
if _use_mlflow == "yes":
    skm = mlflow.sklearn.load_model(f"models:/{meta['model_name']}/{meta['model_version']}")
else:
    skm = mlflow.sklearn.load_model(str(_local_model_dir))

sess = ort.InferenceSession(str(ROOT / "packaging/model/model.onnx"))

# Feed ONNX : numériques en float64, texte en string
feed = {c: X[[c]].astype(np.float64).to_numpy() for c in cfg.numeric}
feed |= {c: X[[c]].fillna("").astype(str).to_numpy(dtype=object) for c in cfg.text}

p_onnx = sess.run(["probabilities"], feed)[0][:, 1]
p_sklearn = skm.predict_proba(X)[:, 1]

t = meta["threshold"]
max_diff = float(np.abs(p_sklearn - p_onnx).max())
same = float(((p_sklearn >= t) == (p_onnx >= t)).mean())
print(
    f"{len(X)} rows · max |p_sklearn - p_onnx| = {max_diff:.2e} · "
    f"same decision (threshold {t:.3f}) : {same:.1%}"
)
assert max_diff < TOL, f"Divergence detected! max_diff={max_diff:.2e} >= {TOL}"
print("OK: predictions match.")
