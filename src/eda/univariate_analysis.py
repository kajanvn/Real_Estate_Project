"""
univariate_analysis.py

Purpose:
- Perform univariate analysis on numeric features
- Generate:
    - Histogram + KDE
    - Boxplot (outlier visualization)
- NO data modification
- Save plots to a provided artifact directory

This is the third layer of EDA.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_univariate_analysis(
    df: pd.DataFrame,
    artifact_dir: Path
) -> None:
    """
    Generate univariate plots for all numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset (immutable)
    artifact_dir : Path
        Directory where plots will be saved
    """

    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for univariate analysis")

    # Create sub-directories
    hist_dir = artifact_dir / "histograms"
    box_dir = artifact_dir / "boxplots"

    hist_dir.mkdir(parents=True, exist_ok=True)
    box_dir.mkdir(parents=True, exist_ok=True)

    for col in numeric_cols:
        # -----------------------------
        # Histogram + KDE
        # -----------------------------
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(hist_dir / f"{col}_hist_kde.png")
        plt.close()

        # -----------------------------
        # Boxplot (Outlier visualization)
        # -----------------------------
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col} ")
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(box_dir / f"{col}_boxplot.png")
        plt.close()
