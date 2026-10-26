"""Télécharge le dataset NLBSE'24 (issues GitHub) et écrit data/raw/dataset.csv."""

from pathlib import Path

import pandas as pd

from src.dataset_config import load_config

OFFICIAL = "https://raw.githubusercontent.com/nlbse2024/issue-report-classification/main/data"
FORK = "https://raw.githubusercontent.com/HarrisAamir/Issue-Report-Classification-NLBSE-2024/main"

RENAMES = {
    "repository": "repo",
    "repository_name": "repo",
    "label": "issue_type",
    "labels": "issue_type",
    "type": "issue_type",
    "issue_title": "title",
    "issue_body": "body",
}


def read_split(split: str) -> pd.DataFrame | None:
    """Lit issues_<split>.csv : dépôt officiel d'abord, puis le dépôt de secours."""
    for base in (OFFICIAL, FORK):
        url = f"{base}/issues_{split}.csv"
        try:
            df = pd.read_csv(url)
            print(f"  {split}: {len(df)} issues <- {url}")
            return df
        except Exception as exc:  # 404, réseau...
            print(f"  {split}: échec ({type(exc).__name__}) <- {url}")
    return None


def main() -> int:
    cfg = load_config()
    parts = [d for d in (read_split("train"), read_split("test")) if d is not None]
    if not parts:
        raise SystemExit("Aucun fichier téléchargé : vérifie ta connexion et les URL.")

    df = pd.concat(parts, ignore_index=True)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns=RENAMES)
    df["issue_type"] = df["issue_type"].replace({"enhancement": "feature"})

    missing = [c for c in ("title", "body", "repo", "issue_type") if c not in df.columns]
    if missing:
        raise SystemExit(f"Colonnes manquantes : {missing}. Colonnes trouvées : {list(df.columns)}")

    title = df["title"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    df["text"] = title + "\n\n" + body.str[:5000]
    df["title_len"] = title.str.len().astype("float64")
    df["body_len"] = body.str.len().astype("float64")
    df["n_code_blocks"] = (body.str.count("```") // 2).astype("float64")

    cfg.validate_columns(df)
    out = Path(cfg.raw_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[cfg.columns].to_csv(out, index=False)

    print(df["issue_type"].value_counts().to_string())
    share = (df[cfg.target] == cfg.positive_class).mean()
    print(f"{len(df)} issues -> {out} · classe '{cfg.positive_class}' : {share:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
