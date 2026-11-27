"""Étape 5 — latence d'inférence : pipeline scikit-learn, ONNX Runtime et API HTTP.

Usage (racine du repo, make pf actif, API lancée sur le port 8010) :
    uv run python packaging/app/benchmark.py
Écrit aussi le tableau dans reports/benchmark.md.

L'ONNX n'est PAS servi : sa conversion du TF-IDF diffère de scikit-learn (tokenisation). Il
est mesuré pour chiffrer le gain de vitesse auquel on renonce au profit de la fidélité.
"""

import json
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from src.dataset_config import load_config  # noqa: E402

N = 1000
URL = os.getenv("API_URL", "http://localhost:8010")

cfg = load_config(ROOT / "dataset.yaml")
meta = json.loads((ROOT / "packaging/model/metadata.json").read_text())
X, _ = cfg.read_xy(ROOT / "data/processed/test.csv")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
sk = mlflow.sklearn.load_model(f"models:/{meta['model_name']}/{meta['model_version']}")
ox = ort.InferenceSession(
    str(ROOT / "packaging/model/model.onnx"), providers=["CPUExecutionProvider"]
)


def onnx_feed(df: pd.DataFrame) -> dict:
    feed = {c: df[[c]].astype(np.float64).to_numpy() for c in cfg.numeric}
    feed |= {c: df[[c]].fillna("").astype(str).to_numpy(dtype=object) for c in cfg.categorical}
    feed |= {c: df[[c]].fillna("").astype(str).to_numpy(dtype=object) for c in cfg.text}
    return feed


def http_call(body: bytes) -> None:
    req = urllib.request.Request(
        f"{URL}/predict", body, {"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=30).read()


def bench(name, fn):
    for _ in range(10):
        fn()  # chauffe
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    a = np.array(times)
    return {"name": name, "p50": np.percentile(a, 50), "p95": np.percentile(a, 95),
            "p99": np.percentile(a, 99)}  # fmt: skip


x1, x100 = X.head(1), X.head(100)
f1, f100 = onnx_feed(x1), onnx_feed(x100)
rows = [{c: (None if v != v else v) for c, v in r.items()} for r in x1.to_dict("records")]
body = json.dumps({"instances": rows}).encode()

ox.run(None, f1)  # échoue ici, avant de mesurer, si une entrée manque
results = [
    bench("sklearn_pipeline (1 ligne)", lambda: sk.predict_proba(x1)),
    bench("sklearn_batch_100", lambda: sk.predict_proba(x100)),
    bench("onnx_python (1 ligne, non servi)", lambda: ox.run(None, f1)),
    bench("onnx_batch_100 (non servi)", lambda: ox.run(None, f100)),
    bench("api_http_sklearn (1 ligne)", lambda: http_call(body)),
]

lines = [f"{'Backend':<36} {'P50':>8} {'P95':>8} {'P99':>8} (ms)"]
lines += [f"{r['name']:<36} {r['p50']:>8.3f} {r['p95']:>8.3f} {r['p99']:>8.3f}" for r in results]
print("\n".join(lines))
speedup = results[0]["p50"] / results[2]["p50"]
print(f"ONNX vs sklearn (P50, 1 ligne) : x{speedup:.1f}")

report = ROOT / "reports" / "benchmark.md"
report.parent.mkdir(exist_ok=True)
table = "\n".join(
    f"| {r['name']} | {r['p50']:.3f} | {r['p95']:.3f} | {r['p99']:.3f} |" for r in results
)
report.write_text(
    f"# Benchmark de latence — {cfg.name}, modèle v{meta['model_version']}\n\n"
    f"{N} appels par mesure, après 10 appels de chauffe · "
    f"{platform.processor() or platform.machine()} · Python {platform.python_version()}\n\n"
    "| Backend | P50 (ms) | P95 (ms) | P99 (ms) |\n|---|---|---|---|\n"
    f"{table}\n\nAccélération ONNX vs scikit-learn (P50, 1 ligne) : x{speedup:.1f}\n"
    "\nL'ONNX n'est pas servi : sa conversion du TF-IDF change des décisions (5 sur 528).\n"
)
print(f"Report written: {report.relative_to(ROOT)}")
