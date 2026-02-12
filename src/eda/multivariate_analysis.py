"""
multivariate_analysis.py

Purpose:
- Analyze relationships among multiple features
- Detect multicollinearity
- Generate correlation heatmap
- NO data modification
- Save plots to artifact directory

This is the fifth and final EDA layer.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_multivariate_analysis(
    df: pd.DataFrame,
    artifact_dir: Path
) -> None:
    """
    Generate multivariate analysis plots.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset (immutable)
    artifact_dir : Path
        Directory where plots will be saved
    """

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        raise ValueError(
            "At least two numeric features are required for multivariate analysis"
        )

    # Create output directory
    multi_dir = artifact_dir / "multivariate"
    multi_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Correlation Heatmap
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    corr_matrix = numeric_df.corr()

    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(multi_dir / "correlation_heatmap.png")
    plt.close()
