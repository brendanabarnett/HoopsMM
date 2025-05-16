from __future__ import annotations

import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier

_DEFAULT_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 3,
    "n_estimators": 100,
    "subsample": 0.8,
    "random_state": 42,
}

def get_model(custom_params: dict | None = None):
    params = _DEFAULT_PARAMS if custom_params is None else {**_DEFAULT_PARAMS, **custom_params}
    return GradientBoostingClassifier(**params)


def save_model(model, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    return joblib.load(path)
