"""API de scoring : sert le pipeline scikit-learn (TF-IDF + random forest) exporté en skops.

Le schéma d'entrée est construit à partir de dataset.yaml : aucune colonne n'est écrite ici.
Le modèle est chargé avec une liste FIXE de types de confiance (skops refuse tout autre type).

Lancement local (racine du repo) :
    uv run python -m uvicorn app.serve:app --app-dir packaging --port 8000
"""

import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model
from skops.io import load as skops_load

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from src.dataset_config import load_config  # noqa: E402

API_VERSION = "1.1.0"
MAX_BATCH = int(os.getenv("MAX_BATCH", "1000"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "packaging" / "model" / "model.skops"))
TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]  # liste exacte, jamais trusted=True

cfg = load_config(os.getenv("DATASET_CONFIG", ROOT / "dataset.yaml"))
model = skops_load(MODEL_PATH, trusted=TRUSTED_TYPES)
meta_path = MODEL_PATH.with_name("metadata.json")
metadata: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}

THRESHOLD = float(os.getenv("DECISION_THRESHOLD", metadata.get("threshold", cfg.cost_threshold())))
MODEL_VERSION = str(metadata.get("model_version", "unknown"))
MODEL_KEYS = ("model_name", "model_alias", "model_version", "run_id", "serving_format",
              "model_sha256", "exported_at")  # fmt: skip

# --- Schéma d'entrée généré depuis dataset.yaml -----------------------------------
Numeric = Annotated[float | None, Field(strict=True, allow_inf_nan=False)]
Text = Annotated[str | None, Field(strict=True)]
Categorical = Annotated[str | None, Field(strict=True, min_length=1)]

field_defs: dict[str, Any] = {c: (Numeric, ...) for c in cfg.numeric}
field_defs |= {c: (Categorical, ...) for c in cfg.categorical}
field_defs |= {c: (Text, ...) for c in cfg.text}
Instance = create_model("Instance", __config__=ConfigDict(extra="forbid"), **field_defs)


class PredictRequest(BaseModel):
    instances: list[Instance] = Field(..., min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    probability: float
    prediction: int


class PredictResponse(BaseModel):
    model_version: str
    positive_class: str
    threshold: float
    predictions: list[Prediction]


app = FastAPI(title=f"Scoring API — {cfg.name}", version=API_VERSION)


def score(rows: list[dict[str, Any]]) -> np.ndarray:
    """Probabilité de la classe positive, préparée comme CFG.read_xy (même typage)."""
    df = pd.DataFrame(rows, columns=cfg.features)
    df[cfg.numeric] = df[cfg.numeric].astype("float64")
    if cfg.text:
        df[cfg.text] = df[cfg.text].fillna("").astype(str)
    return model.predict_proba(df)[:, 1]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/metadata")
def get_metadata():
    return {
        "api_version": API_VERSION,
        "dataset": cfg.name,
        "target": cfg.target,
        "positive_class": str(cfg.positive_class),
        "threshold": THRESHOLD,
        "features": {"numeric": cfg.numeric, "categorical": cfg.categorical, "text": cfg.text},
        "model": {k: metadata.get(k) for k in MODEL_KEYS},
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        proba = score([i.model_dump() for i in req.instances])
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    return PredictResponse(
        model_version=MODEL_VERSION,
        positive_class=str(cfg.positive_class),
        threshold=THRESHOLD,
        predictions=[
            Prediction(probability=float(p), prediction=int(p >= THRESHOLD)) for p in proba
        ],
    )
