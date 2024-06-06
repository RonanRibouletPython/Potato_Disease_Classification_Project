from fastapi import FastAPI, UploadFile, File
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Load the tensorflow model
PROD_MODEL_PATH = "..\\model_creation\\saved_models\\best_model.keras"
BETA_MODEL_PATH = "..\\model_creation\\saved_models\\beta_model.keras"
PROD_MODEL = tf.keras.models.load_model(PROD_MODEL_PATH)
BETA_MODEL = tf.keras.models.load_model(BETA_MODEL_PATH)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

app = FastAPI()

@app.get("/ping")
async def read_root():
    return "Hello World"

def read_file_as_img(data)-> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))

    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
        
    # Convert the uploaded file into a numpy array for the prediction
    img = read_file_as_img(await file.read())

    # Predict the 
    img_batch = np.expand_dims(img, 0)
    predictions = PROD_MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return{
        "predicted_class": predicted_class,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
