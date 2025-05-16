from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils.config import load_config
from tasks.rank_teams.data import load_data, preprocess, weight_map
from tasks.rank_teams.model import load_model


def evaluate_holdout(model_path: Path, csv_path: Path):
    

    df = pd.read_csv(csv_path)
    X, y_enc, groups, le = preprocess(df)
    model = load_model(model_path)
    y_pred = model.predict(X)
    print("Accuracy:", accuracy_score(y_enc, y_pred))
    print(classification_report(y_enc, y_pred, target_names=le.classes_))


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    if cfg.get("evaluate_holdout"):
        evaluate_holdout(
            model_path=Path(cfg["training"]["output_dir"]) / "rank_teams.pkl",
            csv_path=Path(cfg["evaluate_holdout"]["csv"]),
        )
        return

    # else cross‑val
    df = load_data(cfg["data"]["path"])
    X, y_enc, groups, le = preprocess(df)
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    wmap = weight_map()
    waccs = []
    for tr, te in gkf.split(X, y_enc, groups):
        model = load_model(Path(cfg["training"]["output_dir"]) / "rank_teams.pkl")
        y_pred = model.fit(X.iloc[tr], y_enc[tr]).predict(X.iloc[te])  # refit inside loop
        weights = np.array([wmap[le.inverse_transform([t])[0]] for t in y_enc[te]])
        correct = (y_pred == y_enc[te]).astype(int)
        waccs.append(weights.dot(correct) / weights.sum())
    print("Weighted accuracy (model re-trained each fold):", np.mean(waccs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ranking model")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
