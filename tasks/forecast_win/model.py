import joblib
from xgboost import XGBClassifier


def get_model(params: dict) -> XGBClassifier:
    """
    Instantiate an XGBoost classifier with given parameters.
    """
    return XGBClassifier(**params)


def save_model(model, path: str):
    """Persist model to disk."""
    joblib.dump(model, path)


def load_model(path: str):
    """Load persisted model from disk."""
    return joblib.load(path)