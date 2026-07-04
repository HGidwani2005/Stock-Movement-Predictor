import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

DATA_DIR = "data/processed"
MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)

files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

features = [
    "Return",
    "MA5",
    "MA10",
    "MA20",
    "EMA10",
    "RSI",
    "Volatility"
]

for file in files:

    stock = file.replace(".csv", "")

    print(f"\nEvaluating {stock}")

    df = pd.read_csv(os.path.join(DATA_DIR, file))

    X = df[features]
    y = df["Target"]

    split = int(len(df) * 0.8)

    X_test = X[split:]
    y_test = y[split:]

    model = joblib.load(
        os.path.join(MODEL_DIR,
                     f"{stock}_model.pkl")
    )

    y_pred = model.predict(X_test)

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    # Confusion Matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"{stock} Confusion Matrix")
    plt.savefig(f"{RESULT_DIR}/{stock}_confusion_matrix.png")
    plt.close()

    # ROC Curve
    if hasattr(model, "predict_proba"):
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title(f"{stock} ROC Curve")
        plt.savefig(f"{RESULT_DIR}/{stock}_roc_curve.png")
        plt.close()

    # Feature Importance
    if hasattr(model, "feature_importances_"):

        importance = pd.Series(
            model.feature_importances_,
            index=features
        )

        importance.sort_values().plot(
            kind="barh",
            figsize=(8,5)
        )

        plt.title(f"{stock} Feature Importance")
        plt.tight_layout()

        plt.savefig(
            f"{RESULT_DIR}/{stock}_feature_importance.png"
        )

        plt.close()

print("\nEvaluation completed!")