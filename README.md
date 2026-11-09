# ModelServe

A machine learning project that predict customer churn using a trained sckit-learn pipeline and a FASTAPI web API

## Project Overview

ModelServe accept customer information, validate the input, and return a churn prediction with an estimated probability.

## Features

- Customer Churn prediction using Logistics Regression
- Numerical and categorical preprocessing
- Feature scaling and one-hot encoding
- REST API bulit with FASTAPI
- Input validation with pydantic
- Automated API tests with pytest
- Saved Model pipeline loaded for predictions

## Tech Stack

python, pandas, scikit-learn, FASTAPI, Pydantic, Joblib, Pytest

## Dataset

The project uses the Telco Customer Churn dataset from Kaggle.

Dataset source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Download the dataset and place it at `data/raw/Telco-Customer-Churn.csv`.

## Run Locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the API from the project root:

```bash
python -m uvicorn app.api:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

## Run Tests

```bash
python -m pytest -v
```

## Current Limitations

- Predictions are based on the dataset used for training.
- Model performance depends on data quality and representativeness.
- The API has not yet been containerized or deployed to a public server