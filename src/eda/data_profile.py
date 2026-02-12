"""
data_profile.py

Purpose:
- Provide a structural and statistical overview of the dataset
- NO plots
- NO data modification
- Output is a JSON-serializable report

This is the first layer of EDA.
"""

import pandas as pd
from typing import Dict, Any


def generate_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive data profiling report.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset (immutable)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing schema, statistics, and structure info
    """

    profile = {}

    # ------------------------------------------------------------------
    # Basic dataset information
    # ------------------------------------------------------------------
    profile["dataset_shape"] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
    }

    # ------------------------------------------------------------------
    # Column-level schema
    # ------------------------------------------------------------------
    profile["columns"] = {}

    for col in df.columns:
        profile["columns"][col] = {
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].notnull().sum()),
            "null_count": int(df[col].isnull().sum()),
            "unique_values": int(df[col].nunique()),
        }

        # Numeric statistics (only if applicable)
        if pd.api.types.is_numeric_dtype(df[col]):
            profile["columns"][col].update(
                {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()),
                }
            )

    # ------------------------------------------------------------------
    # Global statistics
    # ------------------------------------------------------------------
    profile["global_statistics"] = {
        "total_missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(
            df.memory_usage(deep=True).sum() / (1024 * 1024), 3
        ),
    }

    return profile
