import mlflow.pyfunc
from pathlib import Path

_model = None

def load_production_model():
    global _model
    if _model is None:
        model_path = Path(__file__).resolve().parents[2] / "model" / "production_model"
        _model = mlflow.pyfunc.load_model(str(model_path))
    return _model




'''import mlflow.pyfunc

_model = None

def load_production_model():
    global _model
    if _model is None:
        _model = mlflow.pyfunc.load_model(
            "models:/RealEstateModel@production"
        )
    return _model'''

