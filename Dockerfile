FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_model.py app.py ./
COPY data/raw_dataset.csv data/raw_dataset.csv

# Train the model inside the image, so it matches the installed library versions
RUN python train_model.py

EXPOSE 8080
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} app:app
