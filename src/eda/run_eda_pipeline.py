"""
run_eda_pipeline.py

Purpose:
- Orchestrate the complete EDA workflow
- Load raw data (immutable)
- Run all EDA layers
- Log reports and plots to MLflow
- Ensure artifacts are versioned per run
"""

import json
import mlflow
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


from src.config.config import (
    RAW_DATA_FILE,
    TARGET_COLUMN,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_EDA,
)
from src.common.utils import create_temp_artifact_dir, cleanup_artifact_dir
from src.data_preparation.load_data import load_raw_data
from src.eda.data_profile import generate_data_profile
from src.eda.data_quality import generate_data_quality_report
from src.eda.univariate_analysis import run_univariate_analysis
from src.eda.bivariate_analysis import run_bivariate_analysis
from src.eda.multivariate_analysis import run_multivariate_analysis
from src.preprocessing.preprocess import preprocess_data


print("you are into run_eda_pipeline")

print("EDA Registry URI:", mlflow.get_registry_uri())

print("ACTIVE tracking URI:", mlflow.get_tracking_uri())
#print("ACTIVE experiment:", mlflow.get_experiment_by_name(EXPERIMENT_EDA))


def run_eda_pipeline() -> None:
    """
    Execute full EDA pipeline and log all outputs to MLflow.
    """

    # ---------------------------------------------------------
    # MLflow setup
    # ---------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_EDA)

    # ---------------------------------------------------------
    # Load raw data
    # ---------------------------------------------------------
    df = load_raw_data(RAW_DATA_FILE)

    # ---------------------------------------------------------
    # Create temporary artifact directory
    # ---------------------------------------------------------
    artifact_dir = create_temp_artifact_dir(
        base_dir="eda",
        prefix="eda"
    )

    with mlflow.start_run(run_name="EDA_Run"):

        # -----------------------------------------------------
        # Data profiling report
        # -----------------------------------------------------
        data_profile = generate_data_profile(df)
        profile_path = artifact_dir / "data_profile.json"
        with open(profile_path, "w") as f:
            json.dump(data_profile, f, indent=4)

        # -----------------------------------------------------
        # Data quality report
        # -----------------------------------------------------
        data_quality = generate_data_quality_report(df)
        quality_path = artifact_dir / "data_quality_report.json"
        with open(quality_path, "w") as f:
            json.dump(data_quality, f, indent=4)

        # -----------------------------------------------------
        # Univariate analysis (plots)
        # -----------------------------------------------------
        run_univariate_analysis(df, artifact_dir)

        # -----------------------------------------------------
        # Bivariate analysis (feature vs target)
        # -----------------------------------------------------
        run_bivariate_analysis(
            df=df,
            target_column=TARGET_COLUMN,
            artifact_dir=artifact_dir
        )

        # -----------------------------------------------------
        # Multivariate analysis
        # -----------------------------------------------------
        run_multivariate_analysis(df, artifact_dir)

        # -----------------------------------------------------
        # Log artifacts to MLflow
        # Note: Convert to absolute path for MLflow
        # -----------------------------------------------------
        mlflow.log_artifacts(str(artifact_dir.resolve()), artifact_path="eda")

        X_before = df.drop(columns=[TARGET_COLUMN])
        # Apply preprocessing (includes outlier capping)
        X_after, y, fe_report = preprocess_data(df)

        plot_path = Path("eda_lat_long_before_after_capping.png")

        plot_before_after_outlier_capping_vertical(
        X_before=X_before,
        X_after=X_after,
        bounds=fe_report["outlier_detection"],
        output_path=plot_path
        )

        mlflow.log_artifact(
        str(plot_path.resolve()),
        artifact_path="eda"
        )

        plot_path.unlink(missing_ok=True)

        # ---------------------------------------------------------
        # Cleanup local artifacts
        # ---------------------------------------------------------
        cleanup_artifact_dir(artifact_dir)

def plot_before_after_outlier_capping_vertical(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    bounds: dict,
    output_path: Path
) -> None:
    """
    Create a single EDA report with two vertically stacked plots:
    - Left: Before capping (raw column names)
    - Right: After capping (processed column names)
    """

    # 🔑 Map processed → raw column names
    RAW_COLUMN_MAP = {
        "X5_latitude": "X5 latitude",
        "X6_longitude": "X6 longitude",
    }

    features = list(bounds.keys())  # processed names

    fig, axes = plt.subplots(
        nrows=len(features),
        ncols=2,
        figsize=(12, 8)
    )

    for row, feature in enumerate(features):
        raw_feature = RAW_COLUMN_MAP[feature]

        lower = bounds[feature]["lower_bound"]
        upper = bounds[feature]["upper_bound"]

        # -------- BEFORE (RAW DATA) --------
        axes[row, 0].boxplot(X_before[raw_feature], vert=False)
        axes[row, 0].axvline(lower, color="red", linestyle="--", label="Lower bound")
        axes[row, 0].axvline(upper, color="red", linestyle="--", label="Upper bound")
        axes[row, 0].set_title(f"{raw_feature} — Before Capping")
        axes[row, 0].legend()

        # -------- AFTER (PROCESSED DATA) --------
        axes[row, 1].boxplot(X_after[feature], vert=False)
        axes[row, 1].axvline(lower, color="green", linestyle="--", label="Lower bound")
        axes[row, 1].axvline(upper, color="green", linestyle="--", label="Upper bound")
        axes[row, 1].set_title(f"{feature} — After Capping")
        axes[row, 1].legend()

    plt.suptitle(
        "Outlier Capping Validation (Latitude & Longitude)",
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path)
    plt.close()




if __name__ == "__main__":
    run_eda_pipeline()
