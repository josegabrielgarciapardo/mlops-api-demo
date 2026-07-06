from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
from pathlib import Path


MODEL_PATH = Path(__file__).parent / "artifacts" / "model.joblib"

app = FastAPI(
    title="MLOps Demo API",
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


@app.get("/health")
def health():
    return {"status": "ok"}


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


@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MLOps Demo</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 700px;
                margin: 40px auto;
                background: #f7f7f7;
            }
            .card {
                background: white;
                padding: 24px;
                border-radius: 12px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            }
            input {
                width: 100%;
                padding: 10px;
                margin: 8px 0 16px 0;
                box-sizing: border-box;
            }
            button {
                padding: 12px 20px;
                cursor: pointer;
                font-weight: bold;
            }
            pre {
                background: #111;
                color: #0f0;
                padding: 16px;
                border-radius: 8px;
                overflow-x: auto;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>MLOps Demo: Iris Classifier</h1>
            <p>
                Esta interfaz consume internamente el endpoint
                <code>POST /predict</code>.
            </p>

            <label>Sepal length</label>
            <input id="sepal_length" type="number" step="0.1" value="5.1">

            <label>Sepal width</label>
            <input id="sepal_width" type="number" step="0.1" value="3.5">

            <label>Petal length</label>
            <input id="petal_length" type="number" step="0.1" value="1.4">

            <label>Petal width</label>
            <input id="petal_width" type="number" step="0.1" value="0.2">

            <button onclick="predict()">Predict</button>

            <h2>Result</h2>
            <pre id="result">Waiting for prediction...</pre>
        </div>

        <script>
            async function predict() {
                const payload = {
                    sepal_length: parseFloat(document.getElementById("sepal_length").value),
                    sepal_width: parseFloat(document.getElementById("sepal_width").value),
                    petal_length: parseFloat(document.getElementById("petal_length").value),
                    petal_width: parseFloat(document.getElementById("petal_width").value)
                };

                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(payload)
                });

                const result = await response.json();
                document.getElementById("result").textContent =
                    JSON.stringify(result, null, 2);
            }
        </script>
    </body>
    </html>
    """