import joblib
import numpy as np
from pathlib import Path


sepal_length = 2.1
sepal_width = 2.5
petal_length = 2.4
petal_width = 1.2

features = np.array([[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]])


MODEL_PATH = "C:/Users/joseg/Desktop/MLOps/app/artifacts/model.joblib"
model = joblib.load(MODEL_PATH)

prediction = int(model.predict(features)[0])
probabilities = model.predict_proba(features)[0]

print(prediction)
print(probabilities)
breakpoint