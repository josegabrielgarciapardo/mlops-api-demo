import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/predict"


def predict(sepal_length, sepal_width, petal_length, petal_width):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    response = requests.post(API_URL, json=payload, timeout=10)
    response.raise_for_status()

    result = response.json()

    return {
        "class_name": result["class_name"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"]
    }


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Sepal length"),
        gr.Number(label="Sepal width"),
        gr.Number(label="Petal length"),
        gr.Number(label="Petal width"),
    ],
    outputs=gr.JSON(label="Prediction result"),
    title="MLOps Demo: Iris Classifier",
    description="Interfaz web que consume una API de inferencia"
)

demo.launch()