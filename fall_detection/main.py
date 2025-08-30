from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import mediapipe as mp
import io

app = FastAPI()

interpreter = tf.lite.Interpreter(model_path="model_float16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True)

def predict(img: Image.Image) -> str:
    img_np = np.array(img)

    results = pose_detector.process(img_np)
    if not results.pose_landmarks:
        return "No human has been discovered to classify his status."

    img = img.resize((408, 612))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    prediction = output_data[0][0]
    label = "NOT FALL" if prediction > 0.5 else "FALL"
    return f"Prediction: {label} ({prediction:.2f})"

@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = predict(image)
        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
