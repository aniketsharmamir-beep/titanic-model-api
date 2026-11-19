# Titanic Survival Model API

## Overview
A logistic regression model that predicts whether a Titanic passenger survived, served through a Flask API, packaged in Docker, and deployed on Hugging Face Spaces (a free cloud platform that runs Docker containers).


## Architecture
Source files -> Hugging Face Spaces builds the Docker image from the Dockerfile (training the model during the build) -> runs the container with gunicorn and Flask on port 7860 -> public HTTPS endpoint.

## API Endpoints

### GET /health
Checks that the service is running.
Response: {"status": "ok"}

### POST /predict
Predicts survival for one passenger.

Request body (JSON):
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 29,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 80,
  "Embarked": "C"
}
Pclass must be 1, 2 or 3. Sex must be "male" or "female". Embarked is optional (default "S").

Response:
{"survived": 1, "survival_probability": 0.954}

Errors (HTTP 400): missing fields, invalid Sex or Pclass, non-numeric values, or a body that is not JSON.

## Deployment Process
1. Train the model (python train_model.py), which saves model/titanic_model.joblib.
2. Serve it with Flask (app.py) behind gunicorn.
3. Package it with the Dockerfile.
4. Create a Docker Space on Hugging Face, upload the files, and add the YAML settings (sdk: docker, app_port: 7860) to its README.
5. Hugging Face builds the image and starts the container.
6. Test the live URL with test_api.py (see docs/live_api_test.png).

## Docker Image
The image is defined in the Dockerfile and built automatically by Hugging Face Spaces.

## Why Hugging Face Spaces
It runs Docker containers on a free tier with a public URL, so no paid cloud account was needed. The same Dockerfile also works on AWS or Google Cloud Run.

## Model
Logistic Regression with imputation, scaling, and one-hot encoding in one pipeline. Test accuracy 0.804, F1-score 0.724.

## Run Locally
pip install -r requirements.txt, then python train_model.py, then python app.py.


