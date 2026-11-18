"""
Train the Titanic survival model and save it to model/titanic_model.joblib.

Run:  python train_model.py
"""

import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/raw_dataset.csv"
MODEL_PATH = "model/titanic_model.joblib"

NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Sex", "Embarked"]


def build_pipeline():
    """Preprocessing and model in one object, so the API can accept raw values."""
    numeric = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore"))])
    preprocess = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000))])


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC + CATEGORICAL]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, pred):.3f}")
    print(f"Test F1-score: {f1_score(y_test, pred):.3f}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print("Model saved to", MODEL_PATH)


if __name__ == "__main__":
    main()
