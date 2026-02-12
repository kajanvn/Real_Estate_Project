"""
decision_tree.py

Baseline Decision Tree Regressor.

Purpose:
- Capture non-linear feature interactions
- Provide interpretable decision rules
- Serve as a precursor to ensemble tree models

Artifacts logged to MLflow:
- Metrics (RMSE, MAE, R2)
- Residual plot
- Prediction vs Actual plot
"""

import pandas as pd
import matplotlib.pyplot as plt

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


def train_decision_tree_regressor(
    df: pd.DataFrame,
    experiment_name: str,
    max_depth: int | None = None,
    min_samples_split: int = 2
) -> None:
    """
    Train and evaluate a Decision Tree regression baseline model.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
    experiment_name : str
        MLflow experiment name
    max_depth : int | None
        Maximum depth of the tree
    min_samples_split : int
        Minimum samples required to split an internal node
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
    model = DecisionTreeRegressor(
        random_state=RANDOM_STATE,
        max_depth=max_depth,
        min_samples_split=min_samples_split
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

    with mlflow.start_run(run_name="Decision_Tree"):

        # ---------------- Parameters ----------------
        mlflow.log_param("model_type", "DecisionTree")
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)

        # ---------------- Metrics -------------------
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # -----------------------------------------------------
        # Artifact directory
        # -----------------------------------------------------
        artifact_dir = create_temp_artifact_dir(
            base_dir="baseline_models",
            prefix="decision_tree"
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
        plt.title("Residual Plot – Decision Tree")
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
        plt.title("Prediction vs Actual – Decision Tree")
        plt.tight_layout()
        plt.savefig(artifact_dir / "prediction_vs_actual.png")
        plt.close()

        # -----------------------------------------------------
        # Log artifacts
        # -----------------------------------------------------
        mlflow.log_artifacts(
            artifact_dir,
            artifact_path="baseline/decision_tree"
        )

        # -----------------------------------------------------
        # Log model
        # -----------------------------------------------------
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )
