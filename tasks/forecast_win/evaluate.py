import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.config import load_config
from tasks.forecast_win.data import load_data, preprocess
from tasks.forecast_win.model import load_model
from utils.split import temporal_split


def main(config_path: str):
    cfg = load_config(config_path)

    df = load_data(cfg['data']['path'])
    X, y = preprocess(df, cfg['data']['features'], cfg['data']['target'])
    X_train, X_test, y_train, y_test = temporal_split(X, y, cfg['split'])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = load_model(cfg['training']['output_dir'] + "/forecast_win.pkl")

    y_pred = model.predict(X_test)

    print("[evaluate] Accuracy:", accuracy_score(y_test, y_pred))
    print("[evaluate] Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate win-forecast model")
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)
