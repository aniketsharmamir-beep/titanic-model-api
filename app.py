"""
Flask API for the Titanic survival model.

Endpoints:
    GET  /health   -> {"status": "ok"}
    POST /predict  -> survival prediction for one passenger
"""

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("model/titanic_model.joblib")

REQUIRED = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send a JSON object in the request body"), 400

    missing = [field for field in REQUIRED if field not in data]
    if missing:
        return jsonify(error=f"Missing fields: {missing}"), 400
    if data["Sex"] not in ("male", "female"):
        return jsonify(error="Sex must be 'male' or 'female'"), 400
    if data["Pclass"] not in (1, 2, 3):
        return jsonify(error="Pclass must be 1, 2 or 3"), 400

    passenger = pd.DataFrame([{
        "Pclass": data["Pclass"], "Sex": data["Sex"], "Age": data["Age"],
        "SibSp": data["SibSp"], "Parch": data["Parch"], "Fare": data["Fare"],
        "Embarked": data.get("Embarked", "S"),
    }])

    try:
        probability = float(model.predict_proba(passenger)[0][1])
    except (ValueError, TypeError):
        return jsonify(error="Age, SibSp, Parch and Fare must be numbers"), 400

    return jsonify(survived=int(probability >= 0.5),
                   survival_probability=round(probability, 3))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
