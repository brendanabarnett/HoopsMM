from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import LabelEncoder

_FEATURES = [
    "wins", "avg_pts", "avg_pts_opp", "avg_margin",
    "avg_off_eff", "avg_def_eff", "win_pct", "seed_guess",
]
_BUCKET_RANK = {
    "1": 1.0,
    "2": 2.0,
    "3-4": 3.5,
    "5-8": 6.5,
    "9-16": 12.5,
    "17-32": 24.5,
    "33-68": 50.5,
}


def load_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, LabelEncoder]:
    """Return X, encoded y, season groups, and fitted LabelEncoder."""
    le = LabelEncoder()
    y_enc = le.fit_transform(df["rank_group"].values)
    X = df[_FEATURES].copy()
    groups = df["season"].copy()
    return X, y_enc, groups, le


def weight_map() -> dict[str, float]:
    return {grp: 1.0 / r for grp, r in _BUCKET_RANK.items()}
