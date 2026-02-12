"""
data_quality.py

Purpose:
- Assess data quality issues that affect modeling
- Identify missing data patterns
- Detect duplicates, constant features, skewness
- NO data modification

This is the second layer of EDA.
"""

import pandas as pd
from typing import Dict, Any


def generate_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a data quality report.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset (immutable)

    Returns
    -------
    Dict[str, Any]
        Dictionary summarizing data quality issues
    """

    report: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Missing value analysis
    # ------------------------------------------------------------------
    missing_counts = df.isnull().sum()
    missing_percent = (missing_counts / len(df)) * 100

    report["missing_values"] = {
        col: {
            "missing_count": int(missing_counts[col]),
            "missing_percentage": round(float(missing_percent[col]), 3),
        }
        for col in df.columns
        if missing_counts[col] > 0
    }

    # ------------------------------------------------------------------
    # Duplicate analysis
    # ------------------------------------------------------------------
    report["duplicate_rows"] = {
        "count": int(df.duplicated().sum()),
        "percentage": round(
            (df.duplicated().sum() / len(df)) * 100, 3
        ),
    }

    # ------------------------------------------------------------------
    # Constant / near-constant feature detection
    # ------------------------------------------------------------------
    constant_features = []
    near_constant_features = []

    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)

        if df[col].nunique() == 1:
            constant_features.append(col)
        elif unique_ratio < 0.01:
            near_constant_features.append(col)

    report["constant_features"] = constant_features
    report["near_constant_features"] = near_constant_features

    # ------------------------------------------------------------------
    # Distribution skewness (numeric only)
    # ------------------------------------------------------------------
    skewness = {}
    kurtosis = {}

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        skewness[col] = round(float(df[col].skew()), 4)
        kurtosis[col] = round(float(df[col].kurtosis()), 4)

    report["distribution_shape"] = {
        "skewness": skewness,
        "kurtosis": kurtosis,
    }

    # ------------------------------------------------------------------
    # Potential data quality flags
    # ------------------------------------------------------------------
    flags = []

    if report["duplicate_rows"]["count"] > 0:
        flags.append("Dataset contains duplicate rows")

    if len(constant_features) > 0:
        flags.append("Constant features detected")

    if len(report["missing_values"]) > 0:
        flags.append("Missing values present")

    report["quality_flags"] = flags

    return report
