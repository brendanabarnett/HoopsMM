import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from utils.config import load_config
from tasks.forecast_win.data import load_data, preprocess
from tasks.forecast_win.model import get_model, save_model
from utils.split import temporal_split


def main(config_path: str):
    cfg = load_config(config_path)

    df = load_data(cfg['data']['path'])
    X, y = preprocess(df, cfg['data']['features'], cfg['data']['target'])
    X_train, X_test, y_train, y_test = temporal_split(X, y, cfg['split'])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = get_model(cfg['model']['params'])
    model.fit(X_train, y_train)

    out_dir = Path(cfg['training']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, out_dir / "forecast_win.pkl")
    print("[train] Model saved to", out_dir / "forecast_win.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train win-forecast model")
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)