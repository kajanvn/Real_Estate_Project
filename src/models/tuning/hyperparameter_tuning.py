"""
hyperparameter_tuning.py

Purpose:
- Tune ONE selected model
- Log metrics, plots, SHAP
- ALWAYS log model artifact
- OPTIONAL model registration (manual promotion via UI)

Versioning rules:
- Run name: tuning_<model>_v1, v2, v3 ...
- Model artifact: model_v1, model_v2 ...
"""

import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from xgboost import XGBRegressor

import shap

from src.config.config import (
    RANDOM_STATE,
    TEST_SIZE,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_TUNING,
    REGISTERED_MODEL_NAME,
    RAW_DATA_FILE,
)
from src.preprocessing.preprocess import preprocess_data
from src.common.utils import create_temp_artifact_dir, cleanup_artifact_dir
from src.data_preparation.load_data import load_raw_data


# ==========================================================
# 🔴 DEVELOPER CONTROLS
# ==========================================================
MODEL_NAME = "xgboost"
# "linear_regression", "knn", "decision_tree",
# "random_forest", "gradient_boosting", "adaboost", "xgboost"

REGISTER_MODEL = False
# ==========================================================


def get_next_run_version(experiment_id: str, model_name: str) -> int:
    """
    Increment version based on number of runs for this model
    """
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.model_name = '{model_name}'"
    )
    return len(runs) + 1


def get_model_and_param_grid(model_name: str):

    if model_name == "linear_regression":
        return LinearRegression(), {}

    if model_name == "knn":
        return KNeighborsRegressor(), {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
        }

    if model_name == "decision_tree":
        return DecisionTreeRegressor(random_state=RANDOM_STATE), {
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
        }

    if model_name == "random_forest":
        return RandomForestRegressor(random_state=RANDOM_STATE), {
            "n_estimators": [200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }

    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=RANDOM_STATE), {
            "n_estimators": [200, 300],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        }

    if model_name == "adaboost":
        return AdaBoostRegressor(random_state=RANDOM_STATE), {
            "n_estimators": [200, 300],
            "learning_rate": [0.05, 0.1],
        }

    if model_name == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ), {
            "n_estimators": [225,250],
            "learning_rate": [0.02,0.025],
            "max_depth": [3, 4],
            "subsample": [0.9,0.95,1],
        }

    raise ValueError(f"Unsupported model: {model_name}")


def run_hyperparameter_tuning(df: pd.DataFrame) -> None:

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_TUNING)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_TUNING)
    version = get_next_run_version(experiment.experiment_id, MODEL_NAME)

    run_name = f"tuning_{MODEL_NAME}_v{version}"
    model_artifact_path = f"model_v{version}"

    X, y, _ = preprocess_data(df)
    
    if "No" in X.columns:
        X = X.drop(columns=["No"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model, param_grid = get_model_and_param_grid(MODEL_NAME)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.set_tags({
            "model_name": MODEL_NAME,
            "model_version": f"v{version}",
            "stage": "tuning",
            "run_name": run_name,
        })

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics({
            "rmse": root_mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        })

        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, best_model.predict(X_train))

        mlflow.sklearn.log_model(
            best_model,
            artifact_path=model_artifact_path,
            signature=signature,
            input_example=X_train.iloc[:5],
        )

        artifact_dir = create_temp_artifact_dir("tuning", MODEL_NAME)

        # Prediction vs Actual
        plt.figure(figsize=(7, 5))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], "r--")
        plt.title(f"Prediction vs Actual – {run_name}")
        plt.tight_layout()
        plt.savefig(artifact_dir / "prediction_vs_actual.png")
        plt.close()

        if hasattr(best_model, "feature_importances_"):
            pd.Series(
                best_model.feature_importances_,
                index=X.columns
            ).sort_values().plot(kind="barh", figsize=(8, 6))
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.savefig(artifact_dir / "feature_importance.png")
            plt.close()

            explainer = shap.Explainer(best_model, X_train)
            shap_values = explainer(X_test)

            shap.summary_plot(shap_values, X_test, show=False)
            plt.tight_layout()
            plt.savefig(artifact_dir / "shap_summary.png")
            plt.close()

            for feature in X.columns:
                shap.plots.scatter(shap_values[:, feature], show=False)
                plt.tight_layout()
                plt.savefig(artifact_dir / f"shap_{feature}.png")
                plt.close()

        mlflow.log_artifacts(artifact_dir, artifact_path=f"tuning/{run_name}")
        cleanup_artifact_dir(artifact_dir)

        if REGISTER_MODEL:
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/{model_artifact_path}",
                name=REGISTERED_MODEL_NAME,
            )


if __name__ == "__main__":
    df = load_raw_data(RAW_DATA_FILE)
    run_hyperparameter_tuning(df)
