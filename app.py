from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json

app = FastAPI(title="Fruit Freshness API")

# Allow frontend/backend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model("fruit_model.keras")

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse dictionary
index_to_class = {v: k for k, v in class_indices.items()}


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.get("/")
def root():
    return {
        "message": "Fruit Freshness API Running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents))

    img = preprocess_image(image)

    prediction = model.predict(img)

    class_index = int(np.argmax(prediction))

    confidence = float(np.max(prediction) * 100)

    label = index_to_class[class_index]

    # Split label
    if label.startswith("fresh"):
        status = "Fresh"
        fruit = label.replace("fresh", "").capitalize()
    else:
        status = "Rotten"
        fruit = label.replace("rotten", "").capitalize()

    return {
        "fruit": fruit,
        "status": status,
        "label": label,
        "confidence": round(confidence, 2)
    }