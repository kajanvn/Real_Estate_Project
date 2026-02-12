import mlflow
import os

def configure_mlflow():
    """
    Configure MLflow using environment variables only.
    Safe for local, Docker, and cloud.
    """

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    # else:
    # MLflow will automatically create ./mlruns
