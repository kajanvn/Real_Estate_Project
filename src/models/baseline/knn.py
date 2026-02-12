"""
knn.py

Baseline K-Nearest Neighbors Regressor.

Purpose:
- Capture local, non-linear relationships
- Compare against linear regression
- Understand bias–variance trade-off

Artifacts logged to MLflow:
- Metrics (RMSE, MAE, R2)
- Residual plot
- Prediction vs Actual plot
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)


import mlflow
import mlflow.sklearn

from src.config.config import (
    RANDOM_STATE,
    TEST_SIZE,
)
from src.common.utils import create_temp_artifact_dir
from src.preprocessing.preprocess import preprocess_data
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



def train_knn_regressor(
    df: pd.DataFrame,
    experiment_name: str,
    n_neighbors: int = 5,
    weights: str = "distance"
) -> None:
    """
    Train and evaluate a KNN regression baseline model.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
    experiment_name : str
        MLflow experiment name
    n_neighbors : int
        Number of neighbors
    weights : str
        Weight function used in prediction
    """

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------
    X, y, fe_report = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # ---------------------------------------------------------
    # Model training
    # ---------------------------------------------------------
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights
        ))
    ])
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------
    y_pred = model.predict(X_test)

    # ---------------------------------------------------------
    # Metrics (future-proof)
    # ---------------------------------------------------------
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # ---------------------------------------------------------
    # MLflow logging
    # ---------------------------------------------------------
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="KNN_Regressor"):

        # ---------------- Parameters ----------------
        mlflow.log_param("model_type", "KNN")
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("weights", weights)
        mlflow.log_param("scaling", "StandardScaler")

        # ---------------- Metrics -------------------
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # -----------------------------------------------------
        # Artifact directory
        # -----------------------------------------------------
        artifact_dir = create_temp_artifact_dir(
            base_dir="baseline_models",
            prefix="knn"
        )

        # -----------------------------------------------------
        # Residual plot
        # -----------------------------------------------------
        residuals = y_test - y_pred

        plt.figure(figsize=(7, 5))
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot – KNN Regressor")
        plt.tight_layout()
        plt.savefig(artifact_dir / "residual_plot.png")
        plt.close()

        # -----------------------------------------------------
        # Prediction vs Actual
        # -----------------------------------------------------
        plt.figure(figsize=(7, 5))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--"
        )
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Prediction vs Actual – KNN Regressor")
        plt.tight_layout()
        plt.savefig(artifact_dir / "prediction_vs_actual.png")
        plt.close()

        # -----------------------------------------------------
        # Log artifacts
        # -----------------------------------------------------
        mlflow.log_artifacts(
            artifact_dir,
            artifact_path="baseline/knn"
        )

        # -----------------------------------------------------
        # Log model
        # -----------------------------------------------------
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )
