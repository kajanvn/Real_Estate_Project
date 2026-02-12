import pandas as pd
from typing import Dict, Tuple


def detect_outliers_iqr(df, columns=None):
    """
    Detect outliers using IQR method for selected columns only.
    """
    bounds = {}

    if columns is None:
        columns = df.select_dtypes(include="number").columns

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        bounds[col] = {
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "outlier_percentage": float(
                ((df[col] < lower) | (df[col] > upper)).mean() * 100
            ),
        }

    return bounds



def cap_outliers(
    df: pd.DataFrame,
    bounds: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Cap outliers using provided bounds.
    """
    df_capped = df.copy()

    for col, b in bounds.items():
        df_capped[col] = df_capped[col].clip(
            lower=b["lower_bound"],
            upper=b["upper_bound"]
        )

    return df_capped
