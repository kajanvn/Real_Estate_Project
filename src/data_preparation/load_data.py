from pathlib import Path
import pandas as pd


def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Load raw CSV data without mutating it.
    """

    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataset is empty")

    return df
