from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

MODEL_PATH = Path(__file__).parent / "artifacts" / "model.joblib"

app = FastAPI(
    title="MLOps Demo API",
    description="API demo for a ML predictive model",
    version="1.0.0"
)

model = joblib.load(MODEL_PATH)

CLASS_NAMES = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}


class PredictionRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)


@app.get("/")
def root():
    return {
        "message": "MLOps demo API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": "RandomForestClassifier",
        "model_version": "1.0.0",
        "task": "iris_classification",
        "classes": CLASS_NAMES
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    features = np.array([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]])

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    return {
        "prediction": prediction,
        "class_name": CLASS_NAMES[prediction],
        "confidence": float(np.max(probabilities)),
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(probabilities))
        },
        "model_version": "1.0.0"
    }