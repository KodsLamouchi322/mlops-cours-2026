"""Exporte scoring-model@champion au format skops pour l'image de serving.

L'ONNX ne reproduit pas fidèlement le TF-IDF de scikit-learn (tokenisation différente) : l'API
sert donc directement le pipeline scikit-learn. Usage (racine du repo, make pf actif) :
    uv run python packaging/app/export_sklearn.py
Produit packaging/model/model.skops et complète packaging/model/metadata.json.
"""

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from skops.io import dump

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from src.dataset_config import load_config  # noqa: E402

MODEL_NAME = os.getenv("MODEL_NAME", "scoring-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_DIR = ROOT / "packaging" / "model"

cfg = load_config(ROOT / "dataset.yaml")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
client = MlflowClient()
mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")
threshold = float(client.get_run(mv.run_id).data.params["decision_threshold"])

MODEL_DIR.mkdir(parents=True, exist_ok=True)
path = MODEL_DIR / "model.skops"
dump(model, path)

metadata = {
    "dataset": cfg.name,
    "model_name": MODEL_NAME,
    "model_alias": MODEL_ALIAS,
    "model_version": str(mv.version),
    "run_id": mv.run_id,
    "target": cfg.target,
    "positive_class": str(cfg.positive_class),
    "threshold": round(threshold, 6),
    "threshold_source": "param decision_threshold du run",
    "features": {"numeric": cfg.numeric, "categorical": cfg.categorical, "text": cfg.text},
    "serving_format": "skops",
    "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
}
(MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
print(f"skops exporté : {path.stat().st_size / 1024:.0f} kB · version {mv.version} · seuil {threshold:.3f}")
