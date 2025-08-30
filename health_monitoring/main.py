from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load("Smart_Watch.joblib")

columns = ['Body_Temperature', 'Heart_Rate', 'SPO2']

app = FastAPI(
    title="Vital Signs Prediction API",
    description="API to predict health status based on vital signs",
    version="1.0"
)

class VitalSigns(BaseModel):
    body_temperature: float
    heart_rate: float
    spo2: float

@app.post("/predict")
def predict(vitals: VitalSigns):
    input_data = pd.DataFrame([[vitals.body_temperature, vitals.heart_rate, vitals.spo2]], columns=columns)
    prediction = model.predict(input_data)
    return {"predicted_class": prediction[0]}
