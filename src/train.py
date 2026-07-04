import os
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

for file in files:

    print(f"\nTraining on {file}")

    df = pd.read_csv(os.path.join(DATA_DIR, file))

    features = [
        "Return",
        "MA5",
        "MA10",
        "MA20",
        "EMA10",
        "RSI",
        "Volatility"
    ]

    X = df[features]
    y = df["Target"]

    split = int(len(df) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    models = {

        "Logistic Regression":
        LogisticRegression(max_iter=1000),

        "Decision Tree":
        DecisionTreeClassifier(random_state=42),

        "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

    }

    best_model = None
    best_accuracy = 0

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, pred)

        print(f"{name}: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    stock = file.replace(".csv", "")

    joblib.dump(best_model,
                f"{MODEL_DIR}/{stock}_model.pkl")

    print(f"Best model saved for {stock}")

print("\nTraining completed!")