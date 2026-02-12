"""
xgboost.py

Baseline XGBoost Regressor.

Purpose:
- Strong gradient-boosted tree baseline
- Handles non-linearity, interactions, and feature importance well
- SHAP-friendly for explainability
- Typically the best-performing model on tabular data

Artifacts logged to MLflow:
- Metrics (RMSE, MAE, R2)
- Residual plot
- Prediction vs Actual plot
"""

import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
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


def train_xgboost_regressor(
    df: pd.DataFrame,
    experiment_name: str,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8
) -> None:
    """
    Train and evaluate an XGBoost regression baseline model.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
    experiment_name : str
        MLflow experiment name
    n_estimators : int
        Number of boosting rounds
    learning_rate : float
        Boosting learning rate
    max_depth : int
        Maximum depth of trees
    subsample : float
        Row subsampling ratio
    colsample_bytree : float
        Column subsampling ratio
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
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

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

    with mlflow.start_run(run_name="XGBoost"):

        # ---------------- Parameters ----------------
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("subsample", subsample)
        mlflow.log_param("colsample_bytree", colsample_bytree)

        # ---------------- Metrics -------------------
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # -----------------------------------------------------
        # Artifact directory
        # -----------------------------------------------------
        artifact_dir = create_temp_artifact_dir(
            base_dir="baseline_models",
            prefix="xgboost"
        )

        # -----------------------------------------------------
        # Residual plot
        # -----------------------------------------------------
        residuals = y_test - y_pred

        plt.figure(figsize=(7, 5))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot – XGBoost")
        plt.tight_layout()
        plt.savefig(artifact_dir / "residual_plot.png")
        plt.close()

        # -----------------------------------------------------
        # Prediction vs Actual
        # -----------------------------------------------------
        plt.figure(figsize=(7, 5))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--"
        )
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Prediction vs Actual – XGBoost")
        plt.tight_layout()
        plt.savefig(artifact_dir / "prediction_vs_actual.png")
        plt.close()

        # -----------------------------------------------------
        # Log artifacts
        # -----------------------------------------------------
        mlflow.log_artifacts(
            artifact_dir,
            artifact_path="baseline/xgboost"
        )

        # -----------------------------------------------------
        # Log model
        # -----------------------------------------------------
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )
