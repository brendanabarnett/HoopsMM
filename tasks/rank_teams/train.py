from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from utils.config import load_config
from tasks.rank_teams.data import load_data, preprocess, weight_map
from tasks.rank_teams.model import get_model, save_model


def topk_accuracy(proba, y_true, k: int) -> float:
    topk = np.argsort(proba, axis=1)[:, -k:]
    return np.mean([t in row for t, row in zip(y_true, topk)])


def main(cfg_path: str):
    cfg = load_config(cfg_path)

    df = load_data(cfg["data"]["path"])
    X, y_enc, groups, le = preprocess(df)

    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    buckets = cfg["evaluation"]["buckets"]
    k_vals  = cfg["evaluation"]["k_values"]
    bucket_idxs = {b: le.transform([b])[0] for b in buckets}
    res = {b: {k: [] for k in k_vals} for b in buckets}
    waccs = []
    wmap  = weight_map()

    for tr_idx, te_idx in gkf.split(X, y_enc, groups=groups):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]

        model = get_model(cfg.get("model", {}).get("params"))
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        y_pred = model.predict(X_te)
        y_te_lbl  = le.inverse_transform(y_te)
        y_pred_lbl = le.inverse_transform(y_pred)

        # weighted accuracy this fold
        weights = np.array([wmap[t] for t in y_te_lbl])
        correct = (y_pred_lbl == y_te_lbl).astype(int)
        waccs.append(weights.dot(correct) / weights.sum())

        # per‑bucket top‑k
        for b in buckets:
            mask = y_te_lbl == b
            if not mask.any():
                continue
            proba_sub = proba[mask]
            y_true_sub = y_te[mask]
            for k in k_vals:
                res[b][k].append(topk_accuracy(proba_sub, y_true_sub, k))

    # aggregate
    print("=== Season-Held-Out Metrics ===")
    for b in buckets:
        print(f"Bucket {b}:")
        for k in k_vals:
            arr = np.array(res[b][k])
            print(f"  Top-{k} acc: {arr.mean():.3f} ± {arr.std():.3f}")
        print()
    print("Weighted accuracy per fold:", np.round(waccs, 3))
    print(f"Mean weighted accuracy: {np.mean(waccs):.3f} ± {np.std(waccs):.3f}")

    # fit on full data and save final model
    final_model = get_model(cfg.get("model", {}).get("params"))
    final_model.fit(X, y_enc)
    save_model(final_model, Path(cfg["training"]["output_dir"]) / "rank_teams.pkl")
    print("[train] Final model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train season-ranking model")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
