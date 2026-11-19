# Titanic Survival Model API

## Overview
A machine learning model that predicts whether a Titanic passenger would have survived, served through a Flask API and packaged as a Docker image. The model is a Logistic Regression pipeline that handles missing values, scaling, and encoding, so the API accepts raw passenger details.

## Project Structure
- app.py: the Flask API (endpoints /health and /predict)
- train_model.py: trains the model and saves it to model/titanic_model.joblib
- Dockerfile: builds the Docker image
- requirements.txt: Python libraries used by the image
- test_api.py: script that sends test requests to the API
- data/raw_dataset.csv: the Titanic dataset used for training

## Model
- Algorithm: Logistic Regression
- Inputs: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- Preprocessing: median fill and scaling for numeric columns, most-frequent fill and one-hot encoding for categorical columns, all inside one scikit-learn Pipeline
- Data split: 80% training, 20% testing
- Test accuracy: 0.804
- Test F1-score: 0.724

## API Endpoints

### GET /health
Checks that the service is running.

Response:
{"status": "ok"}

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

Rules: Pclass must be 1, 2 or 3. Sex must be "male" or "female". Embarked is optional (default "S").

Response:
{"survived": 1, "survival_probability": 0.954}

Errors (HTTP 400): missing fields, invalid Sex or Pclass, non-numeric values, or a request body that is not JSON.

## Docker Image
The Dockerfile builds the image in these steps:
1. Start from python:3.11-slim.
2. Install the libraries in requirements.txt.
3. Copy the code and the dataset.
4. Train the model during the build (python train_model.py), so the saved model always matches the installed library versions.
5. Start the API with gunicorn on the port given by the PORT variable (default 8080).

Build and run locally (requires Docker):
docker build -t titanic-api .
docker run -p 8080:8080 titanic-api

Then open http://localhost:8080/health.

## Deployment Process
1. Train the model: python train_model.py
2. Serve it with Flask (app.py) behind gunicorn.
3. Package the service with the Dockerfile.
4. Test the API (see Testing below).
5. To publish it, the same Docker image can be deployed to any platform that runs containers:
   - Google Cloud Run: connect the GitHub repository, choose the Dockerfile as the build type, allow public access, and set the container port.
   - Hugging Face Spaces: create a Docker Space, upload these files, and add sdk: docker and app_port to the Space README.

## Testing
The API was tested in Google Colab using Flask's test client:

from app import app
client = app.test_client()
print(client.get("/health").get_json())
print(client.post("/predict", json={"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 80, "Embarked": "C"}).get_json())

Output:
{'status': 'ok'}
{'survival_probability': 0.954, 'survived': 1}

When the API is running as a server, test_api.py sends the same requests over HTTP. Set BASE_URL inside the file to the server's address first.

## Deployment Status
The API and Docker image are complete and the API was tested in Colab, but the service has not been deployed to a public cloud host. The platform I considered first (Google Cloud Run) requires a billing account, which I did not set up. The deployment steps above describe how to publish the same image.

## Run Locally Without Docker
pip install -r requirements.txt
python train_model.py
python app.py

The API then runs at http://localhost:8080.
