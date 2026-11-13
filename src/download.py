"""Télécharge deux jeux Hugging Face (jailbreak + prompt injection), les fusionne et écrit
data/raw/dataset.csv.

Chaque jeu est épinglé sur un commit précis : le téléchargement donne le même résultat dans
six mois (sinon dvc.lock décrirait un fichier que personne ne peut régénérer).

Colonnes produites : prompt, label (attack / benign), source, prompt_len, n_words.
L'identifiant (prompt_id) est ajouté ensuite par src/prepare.py.
"""

import io
from pathlib import Path

import pandas as pd
import requests

from src.dataset_config import load_config

HF = "https://huggingface.co/datasets"
JAILBREAK_SHA = "2f2ceeb39658696fd3f462403562b6eea5306287"
DEEPSET_SHA = "4f61ecb038e9c3fb77e21034b22511b523772cdd"
JAILBREAK_URL = (
    f"{HF}/jackhhao/jailbreak-classification/resolve/{JAILBREAK_SHA}"
    "/default/jailbreak_dataset_full.csv"
)
DEEPSET_URLS = [
    f"{HF}/deepset/prompt-injections/resolve/{DEEPSET_SHA}/data/{name}"
    for name in (
        "train-00000-of-00001-9564e8b05b4757ab.parquet",
        "test-00000-of-00001-701d16158af87368.parquet",
    )
]


def fetch(url: str) -> io.BytesIO:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    print(f"  {len(resp.content) // 1024} Ko <- {url}")
    return io.BytesIO(resp.content)


def main() -> int:
    cfg = load_config()

    jb = pd.read_csv(fetch(JAILBREAK_URL))
    jb = pd.DataFrame(
        {
            "prompt": jb["prompt"],
            "label": jb["type"].map({"jailbreak": "attack", "benign": "benign"}),
            "source": "jailbreak",
        }
    )
    ds = pd.concat([pd.read_parquet(fetch(u)) for u in DEEPSET_URLS], ignore_index=True)
    ds = pd.DataFrame(
        {
            "prompt": ds["text"],
            "label": ds["label"].map({1: "attack", 0: "benign"}),
            "source": "deepset",
        }
    )

    df = pd.concat([jb, ds], ignore_index=True)
    if df["label"].isna().any():
        raise SystemExit("Étiquette inconnue : vérifier les valeurs de type / label.")

    # Nettoyage AVANT le split : un texte présent deux fois se retrouverait dans train
    # et dans test, et le ROC AUC serait gonflé.
    df["prompt"] = df["prompt"].fillna("").astype(str).str.strip()
    df = df[df["prompt"] != ""]
    n_before = len(df)
    df = df.drop_duplicates(subset="prompt", keep="first").reset_index(drop=True)
    print(f"  {n_before - len(df)} doublon(s) retiré(s)")

    df["prompt_len"] = df["prompt"].str.len().astype("float64")
    df["n_words"] = df["prompt"].str.split().str.len().astype("float64")

    cfg.validate_columns(df)
    out = Path(cfg.raw_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[cfg.columns].to_csv(out, index=False)

    print(df.groupby(["source", "label"]).size().to_string())
    share = (df[cfg.target] == cfg.positive_class).mean()
    print(f"{len(df)} prompts -> {out} · classe '{cfg.positive_class}' : {share:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
