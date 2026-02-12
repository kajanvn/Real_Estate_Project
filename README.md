# Real Estate Price Prediction Project

## Overview
This project predicts real estate prices using machine learning models. It covers data preprocessing, EDA, model training, hyperparameter tuning, and model serving via an API. MLflow is used for experiment tracking and model management.

## Features
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Multiple regression models (Random Forest, XGBoost, etc.)
- Hyperparameter tuning with GridSearchCV
- MLflow experiment tracking and model registry
- SHAP-based model explainability
- REST API for model serving (FastAPI)

## Project Structure
```
├── Dockerfile
├── requirements.txt
├── README.md
├── data/
│   └── raw/Real_estate.csv
├── src/
│   ├── common/
│   ├── config/
│   ├── data_preparation/
│   ├── eda/
│   ├── models/
│   │   └── tuning/hyperparameter_tuning.py
│   ├── preprocessing/
│   └── serving/
├── services/
│   └── api/main.py
├── mlartifacts/
├── mlruns/
└── artifacts/
```

## Setup Instructions
1. **Clone the repository:**
	 ```bash
	 git clone <repo-url>
	 cd Real_Estate_Project
	 ```
2. **Install dependencies:**
	 ```bash
	 pip install -r requirements.txt
	 ```
3. **Configure MLflow tracking:**
	 - Set the MLflow tracking URI in `src/config/config.py` if needed.

4. **Prepare data:**
	 - Place your raw data in `data/raw/Real_estate.csv`.

## Usage
- **Run EDA:**
	```bash
	python -m src.eda.run_eda_pipeline
	```
- **Run hyperparameter tuning:**
	```bash
	python -m src.models.tuning.hyperparameter_tuning
	```
- **Start API server:**
	```bash
	uvicorn services.api.main:app --reload
	```

## MLflow Tracking
- MLflow UI can be started with:
	```bash
	mlflow ui
	```
	Then visit http://localhost:5000

## Notes
- All model artifacts and experiment logs are stored in `mlartifacts/` and `mlruns/`.
- SHAP plots and feature importance are logged as MLflow artifacts.
- The model to tune can be set in `hyperparameter_tuning.py` (`MODEL_NAME`).

## Requirements
See `requirements.txt` for all dependencies.

## License
MIT License

