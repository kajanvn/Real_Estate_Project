"""
preprocess.py

Purpose:
- Apply minimal, explicit preprocessing for modeling
- Driven by EDA findings
- Operates fully in-memory (NO CSV persistence)
- Deterministic and reproducible

Rules:
- No feature selection here
- No model-specific logic
- No file I/O
"""

import pandas as pd
from typing import Tuple

from src.config.config import TARGET_COLUMN
from src.preprocessing.feature_engineering import apply_feature_engineering



def preprocess_data(df):
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
        # -------- Column standardization (API-safe names) --------
    X = X.rename(columns={
        "X1 transaction date": "X1_transaction_date",
        "X2 house age": "X2_house_age",
        "X3 distance to the nearest MRT station": "X3_distance_to_MRT",
        "X4 number of convenience stores": "X4_number_of_convenience_stores",
        "X5 latitude": "X5_latitude",
        "X6 longitude": "X6_longitude",
    })


    # -------- Imputation --------
    numeric_cols = X.select_dtypes(include="number").columns
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    # -------- Feature Engineering --------
    X, fe_report = apply_feature_engineering(X)

    return X, y, fe_report
