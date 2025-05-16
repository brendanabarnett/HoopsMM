import pandas as pd


def load_data(path: str):
    """
    Load raw game-level data from CSV.
    """
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Drop missing values, separate features and target.
    """
    df = df.dropna(subset=feature_cols + [target_col])
    X = df[feature_cols].reset_index(drop=True)
    y = df[target_col].reset_index(drop=True)
    return X, y