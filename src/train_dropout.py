import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

def generate_synthetic_data(n=500):
    """
    Temporary: Generate synthetic student dataset with 100+ features.
    Replace this with real dataset later.
    """
    np.random.seed(42)
    data = {
        "attendance": np.random.randint(50, 100, n),
        "grades": np.random.randint(40, 100, n),
        "participation": np.random.randint(1, 10, n),
        "family_support": np.random.randint(1, 5, n),
        "stress_level": np.random.randint(1, 10, n),
        "dropout": np.random.choice([0, 1], n, p=[0.7, 0.3])  # 0 = stays, 1 = dropout
    }
    return pd.DataFrame(data)

def train_dropout(model_path="models/dropout_model.pkl"):
    # Load or create dataset
    try:
        df = pd.read_csv("data/processed/student_data.csv")
    except FileNotFoundError:
        print("⚠️ No processed dataset found. Using synthetic data instead.")
        df = generate_synthetic_data()

    X = df.drop("dropout", axis=1)
    y = df["dropout"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Dropout Prediction Report:")
    print(classification_report(y_test, preds))
    print("Accuracy:", accuracy_score(y_test, preds))

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Dropout model saved to {model_path}")

if __name__ == "__main__":
    train_dropout()
