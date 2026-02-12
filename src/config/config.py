"""
config.py

Centralized configuration for the Real Estate MLOps project.

Rules:
- NO business logic here
- ONLY constants, paths, and global settings
- Any change in data location or target column should be done here
"""

from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------------------
# Project root (resolved dynamically)
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# -------------------------------------------------------------------
# Data paths
# -------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "Real_estate.csv"

# -------------------------------------------------------------------
# Target & modeling configuration
# -------------------------------------------------------------------
TARGET_COLUMN = "Y house price of unit area"

# Columns that should never be used as model features (if any)
EXCLUDE_COLUMNS = []

# -------------------------------------------------------------------
# MLflow configuration
# -------------------------------------------------------------------
#MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_ARTIFACT_URI = os.getenv("MLFLOW_ARTIFACT_URI")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000")

#MLFLOW_TRACKING_URI = "http://localhost:5000" # 👈 default for local dev)


print("MLFLOW_ARTIFACT_URI: ",MLFLOW_ARTIFACT_URI)
print("MLFLOW_TRACKING_URI: ",MLFLOW_TRACKING_URI)

EXPERIMENT_EDA = "EDA"
EXPERIMENT_BASELINE = "Baseline_Models"
EXPERIMENT_TUNING = "Hyperparameter_Tuning"

REGISTERED_MODEL_NAME = "RealEstateModel"

# -------------------------------------------------------------------
# General settings
# -------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
