from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the model
model = joblib.load("5_activity (1).joblib")

# Class labels
label_names = {
    0: 'standing',
    1: 'walking',
    2: 'running',
    3: 'falling',
    4: 'sitting'
}

# Create FastAPI instance
app = FastAPI(
    title="Elderly Activity Classifier",
    description="API to classify elderly person's activity based on 6 accelerometer features.",
    version="1.0"
)

# Define request body structure
class SensorInput(BaseModel):
    Feature_1: float
    Feature_2: float
    Feature_3: float
    Feature_4: float
    Feature_5: float
    Feature_6: float

# Define the prediction endpoint
@app.post("/predict")
def predict_action(data: SensorInput):
    try:
        input_df = pd.DataFrame([[data.Feature_1, data.Feature_2, data.Feature_3,
                                  data.Feature_4, data.Feature_5, data.Feature_6]],
                                columns=['Feature_1', 'Feature_2', 'Feature_3',
                                         'Feature_4', 'Feature_5', 'Feature_6'])

        prediction = model.predict(input_df)
        predicted_label = label_names[prediction[0]]

        return {"prediction": predicted_label}

    except Exception as e:
        return {"error": str(e)}
