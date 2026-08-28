
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import os

app = FastAPI(
    title="OncoLens AI",
    description="AI-assisted pathology screening backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "oncolens_model.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)


@app.get("/")
def home():
    return {
        "project": "OncoLens AI",
        "status": "Backend running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")
    except Exception:
        return {
            "error": "Invalid image file"
        }

    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    probability = float(
        model.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    if probability >= 0.5:
        prediction = "Suspicious"
    else:
        prediction = "Non-Suspicious"

    return {
        "filename": file.filename,
        "prediction": prediction,
        "cancer_probability": round(probability, 4),
        "cancer_probability_percent": round(
            probability * 100, 2
        ),
        "model": "EfficientNetB0",
        "note": "AI-assisted screening. Final interpretation requires qualified pathologist review."
    }
