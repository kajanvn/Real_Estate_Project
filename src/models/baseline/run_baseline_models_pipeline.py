"""
run_baseline_models_pipeline.py

Pipeline script to run all baseline regression models
on the same dataset and log results to MLflow.

This script DOES NOT select the best model automatically.
Model comparison is done visually in MLflow UI.
"""

import logging

from src.data_preparation.load_data import load_raw_data
from src.config.config import (
    RAW_DATA_FILE,
    EXPERIMENT_BASELINE
)
from src.config.mlflow_config import configure_mlflow
configure_mlflow()


# Baseline models
from src.models.baseline.linear_regression import train_linear_regression
from src.models.baseline.knn import train_knn_regressor
from src.models.baseline.decision_tree import train_decision_tree_regressor
from src.models.baseline.random_forest import train_random_forest_regressor
from src.models.baseline.gradient_boosting import train_gradient_boosting_regressor
from src.models.baseline.adaboost import train_adaboost_regressor

# Optional (only if installed)
try:
    from src.models.baseline.xgboost import train_xgboost_regressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_baseline_models() -> None:
    """
    Execute all baseline models sequentially.
    """

    logging.info("Loading dataset...")
    df = load_raw_data(RAW_DATA_FILE)

    logging.info("Starting baseline model training...")

    # ---------------- Linear Regression ----------------
    logging.info("Training Linear Regression...")
    train_linear_regression(
        df=df,
        experiment_name=EXPERIMENT_BASELINE
    )

    # ---------------- KNN ----------------
    logging.info("Training KNN Regressor...")
    train_knn_regressor(
        df=df,
        experiment_name=EXPERIMENT_BASELINE,
        n_neighbors=5,
        weights="distance"
    )

    # ---------------- Decision Tree ----------------
    logging.info("Training Decision Tree...")
    train_decision_tree_regressor(
        df=df,
        experiment_name=EXPERIMENT_BASELINE
    )

    # ---------------- Random Forest ----------------
    logging.info("Training Random Forest...")
    train_random_forest_regressor(
        df=df,
        experiment_name=EXPERIMENT_BASELINE
    )

    # ---------------- Gradient Boosting ----------------
    logging.info("Training Gradient Boosting...")
    train_gradient_boosting_regressor(
        df=df,
        experiment_name=EXPERIMENT_BASELINE
    )

    # ---------------- AdaBoost ----------------
    logging.info("Training AdaBoost...")
    train_adaboost_regressor(
        df=df,
        experiment_name=EXPERIMENT_BASELINE
    )

    # ---------------- XGBoost (Optional) ----------------
    if XGBOOST_AVAILABLE:
        logging.info("Training XGBoost...")
        train_xgboost_regressor(
            df=df,
            experiment_name=EXPERIMENT_BASELINE
        )
    else:
        logging.warning("XGBoost not available. Skipping.")

    logging.info("All baseline models executed successfully.")


if __name__ == "__main__":
    run_baseline_models()
