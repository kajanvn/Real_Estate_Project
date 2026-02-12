"""
adaboost.py

Baseline AdaBoost Regressor.

Purpose:
- Apply adaptive boosting on weak learners
- Observe sensitivity to hard-to-predict samples
- Compare adaptive boosting vs gradient boosting

Artifacts logged to MLflow:
- Metrics (RMSE, MAE, R2)
- Residual plot
- Prediction vs Actual plot
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
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


def train_adaboost_regressor(
    df: pd.DataFrame,
    experiment_name: str,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    base_max_depth: int = 3
) -> None:
    """
    Train and evaluate an AdaBoost regression baseline model.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
    experiment_name : str
        MLflow experiment name
    n_estimators : int
        Number of boosting rounds
    learning_rate : float
        Shrinks contribution of each estimator
    base_max_depth : int
        Depth of base decision tree learner
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
    # Base learner
    # ---------------------------------------------------------
    base_estimator = DecisionTreeRegressor(
        max_depth=base_max_depth,
        random_state=RANDOM_STATE
    )

    # ---------------------------------------------------------
    # Model training
    # ---------------------------------------------------------
    model = AdaBoostRegressor(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE
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

    with mlflow.start_run(run_name="AdaBoost"):

        # ---------------- Parameters ----------------
        mlflow.log_param("model_type", "AdaBoost")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("base_max_depth", base_max_depth)

        # ---------------- Metrics -------------------
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # -----------------------------------------------------
        # Artifact directory
        # -----------------------------------------------------
        artifact_dir = create_temp_artifact_dir(
            base_dir="baseline_models",
            prefix="adaboost"
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
        plt.title("Residual Plot – AdaBoost")
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
        plt.title("Prediction vs Actual – AdaBoost")
        plt.tight_layout()
        plt.savefig(artifact_dir / "prediction_vs_actual.png")
        plt.close()

        # -----------------------------------------------------
        # Log artifacts
        # -----------------------------------------------------
        mlflow.log_artifacts(
            artifact_dir,
            artifact_path="baseline/adaboost"
        )

        # -----------------------------------------------------
        # Log model
        # -----------------------------------------------------
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )
