import mlflow.pyfunc

_model = None

def load_production_model():
    global _model
    if _model is None:
        _model = mlflow.pyfunc.load_model(
            "models:/RealEstateModel@production"
        )
    return _model
