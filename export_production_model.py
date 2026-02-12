import mlflow
from mlflow.artifacts import download_artifacts
from pathlib import Path
import shutil

MODEL_URI = "models:/RealEstateModel@production"
TARGET_DIR = Path("model/production_model")

# Clean old export
if TARGET_DIR.exists():
    shutil.rmtree(TARGET_DIR)

# Download model artifacts
download_artifacts(
    artifact_uri=MODEL_URI,
    dst_path=str(TARGET_DIR)
)

print("✅ Production model exported to model/production_model")