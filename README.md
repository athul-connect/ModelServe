# ModelServe - Customer Churn Prediction API

A machine learning apllication that predict customer churn using a trained sckit-learn pipeline and a FASTAPI web API

Live API : https://modelserve.onrender.com
API Documentation : https://modelserve.onrender.com/docs
Health Check : https://modelserve.onrender.com/health

## Project Overview

ModelServe accept Customer information,validates the input, and retuen a churn prediction with an estimated probability.

The project demonstrates the workflow of taking a trained machine learning model and making it accessible through a containerized, cloud-hosted API.

## Features

- Customer Churn prediction using Logistics Regression
- Numerical and categorical preprocessing
- Feature scaling and one-hot encoding
- REST API bulit with FASTAPI
- Input validation with pydantic
- Automated API tests with pytest
- Docker containerization
- Github Actions CI for automates testing and Docker image buliding
- Container health-check smoke testing in CI
- Cloud deployment on Render

## Machine Learning Workflow

1. Load and inspect the Telco Customer Churn dataset.
2. Clean the data and prepare the target variable.
3. Separate features from the target and prepare training and test data.
4. Detect numerical and categorical features.
5. Apply scaling and one-hot encoding through a preprocessing pipeline.
6. Train a Logistic Regression classifier.
7. Save the complete trained pipeline using Joblib.
8. Load the saved pipeline in the API for predictions.

## Tech Stack

python, pandas, scikit-learn, FASTAPI, Pydantic, Joblib, Pytest, Docker, GitHub Actions, Render

## Dataset

The project uses the Telco Customer Churn dataset from Kaggle.

Dataset source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Download the dataset and place it at `data/raw/Telco-Customer-Churn.csv`.

The raw dataset isn't included in this repository.

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
## Run With DOcker

Bulid the image from the project root

```bash
docker build -t modelserve:latest .
```
Run the Container:

```bash
docker run --rm -p 8000:8000 modelserve:latest
```
Open the API documentation:

http://127.0.0.1:8000/docs

## Continuous Integration

GitHub Actions runs the project's automated checks when changes are pushed or pull requests are opened.

The CI workflow includes:

- Installing Python dependencies
- Running pytest
- Building the Docker image
- Starting a test container
- Checking the API health endpoint
- Cleaning up the test container

Workflow configuration: .github/workflows/ci.yml

## Deployment

The API is deployed on Render using Docker.

- Live service: https://modelserve.onrender.com
- Health endpoint: https://modelserve.onrender.com/health
- Interactive API documentation: https://modelserve.onrender.com/docs

The free hosting instance may spin down after inactivity, so the first request after a period of inactivity can take longer.

## Current Limitations

- Predictions depend on the quality and representativeness of the       training dataset.
- The model's real-world performance should be evaluated on appropriate unseen data.
- The application is a portfolio project and has not been established as a production-grade, business-critical prediction system.
- The free hosting tier may introduce cold-start latency.

## Author 

Athul Sasikumar

Github :  https://github.com/athul-connect/ModelServe