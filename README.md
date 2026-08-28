# OncoLens AI Backend

AI-assisted pathology screening backend.

## Model
EfficientNetB0

## API Endpoints

GET /
GET /health
POST /predict

## Run

pip install -r requirements.txt

uvicorn app:app --host 0.0.0.0 --port 8000

## Prediction

Upload an H&E biopsy image to `/predict`.

The API returns:
- prediction
- cancer probability
- model information

This system is an AI-assisted screening prototype and does not replace a qualified pathologist.
