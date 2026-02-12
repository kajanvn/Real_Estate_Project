from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd
from pathlib import Path
from fastapi.responses import HTMLResponse

from src.serving.model_loader import load_production_model
from src.serving.schema import HouseFeatures


# --------------------------------------------------
# Lifespan (startup / shutdown)
# --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Production model once at startup
    load_production_model()
    yield
    # (Optional cleanup later)


app = FastAPI(
    title="Real Estate Price Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# UI Route (LOCAL + CLOUD SAFE)
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """
    Serves index.html for browser UI testing
    """
    html_path = BASE_DIR / "index.html"
    return html_path.read_text(encoding="utf-8")


# --------------------------------------------------
# Health Check
# --------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------
@app.post("/predict")
def predict_price(features: HouseFeatures):
    model = load_production_model()

    # Convert request → DataFrame (must match model signature)
    input_df = pd.DataFrame([features.model_dump()])

    prediction = model.predict(input_df)

    return {
        "predicted_price": float(prediction[0])
    }


# --------------------------------------------------
# (Optional) Model Info – VALIDATION ENDPOINT
# --------------------------------------------------
@app.get("/model-info")
def model_info():
    model = load_production_model()
    meta = model.metadata

    return {
        "model_name": meta.run_id,
        "signature": str(meta.signature),
        "artifact_path": meta.artifact_path,
    }
