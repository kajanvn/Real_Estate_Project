import pandas as pd
from typing import Tuple, Dict

from src.preprocessing.outliers import (
    detect_outliers_iqr,
    cap_outliers,
)

OUTLIER_COLUMNS = ["X3_distance_to_MRT","X5_latitude", "X6_longitude"]


def apply_feature_engineering(
    X: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply feature engineering steps:
    - Outlier detection
    - Outlier treatment (capping)

    Returns transformed data + engineering report.
    """

    # -------- Outlier detection --------
    outlier_report = detect_outliers_iqr(X,columns=OUTLIER_COLUMNS)

    # -------- Outlier treatment --------
    X_capped = cap_outliers(X, outlier_report)


    post_treatment_stats = {}

    for col in outlier_report:
        post_treatment_stats[col] = {
            "min_after": float(X_capped[col].min()),
            "max_after": float(X_capped[col].max()),
        }

    fe_report = {
        "outlier_detection": outlier_report,
        "post_treatment_stats": post_treatment_stats,
        "treated_columns": list(outlier_report.keys()),
        "target_outliers_treated": False
    }

    return X_capped, fe_report
