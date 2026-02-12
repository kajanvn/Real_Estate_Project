"""
bivariate_analysis.py

Purpose:
- Analyze relationship between each numeric feature and the target
- Generate:
    - Scatter plots
    - Linear trend line
    - Correlation value annotation
- NO data modification
- Save plots to artifact directory

This is the fourth layer of EDA.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_bivariate_analysis(
    df: pd.DataFrame,
    target_column: str,
    artifact_dir: Path
) -> None:
    """
    Generate bivariate plots (feature vs target).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset (immutable)
    target_column : str
        Name of the target variable
    artifact_dir : Path
        Directory where plots will be saved
    """

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    numeric_cols = df.select_dtypes(include="number").columns

    # Remove target from feature list
    feature_cols = [c for c in numeric_cols if c != target_column]

    if len(feature_cols) == 0:
        raise ValueError("No numeric features available for bivariate analysis")

    # Create output directory
    bivariate_dir = artifact_dir / "feature_vs_target"
    bivariate_dir.mkdir(parents=True, exist_ok=True)

    for col in feature_cols:
        correlation = df[col].corr(df[target_column])

        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=df[col], y=df[target_column], alpha=0.7)
        sns.regplot(
            x=df[col],
            y=df[target_column],
            scatter=False,
            color="red",
            line_kws={"linewidth": 2},
        )

        plt.title(
            f"{col} vs {target_column}\n"
            f"Correlation: {round(correlation, 3)}"
        )
        plt.xlabel(col)
        plt.ylabel(target_column)
        plt.tight_layout()
        plt.savefig(bivariate_dir / f"{col}_vs_{target_column}.png")
        plt.close()
