Automated Property Value Prediction

A machine learning application that predicts house values from property and geographic characteristics. The project combines a trained machine learning pipeline with a FastAPI backend and a React + TypeScript frontend.

Overview

The project follows a complete machine learning workflow:

Load and prepare housing data.

Preprocess the input features.

Train and evaluate five regression models.

Select the best-performing model.

Save the trained model and preprocessing pipeline.

Expose the prediction pipeline through FastAPI.

Connect the API to a React dashboard.

Validate user input before requesting a prediction.

The current best model is Random Forest.

Features

Property value prediction

Data preprocessing and outlier handling

Comparison of five regression models

Saved model and preprocessing pipeline

FastAPI prediction endpoint

React + TypeScript dashboard

Tailwind CSS styling

Glassmorphism interface

Input validation

Loading and error states

Reset functionality

Model performance display

INR-formatted prediction display

Machine Learning Models

Linear Regression

Ridge

Random Forest

Gradient Boosting

XGBoost

Model Results

Model

R² Score

Linear Regression

64.70%

Ridge

64.70%

Random Forest

80.37%

Gradient Boosting

77.68%

XGBoost

77.56%

The documented Random Forest RMSE is 0.507.

Input Features

Median Income

House Age

Average Rooms

Average Bedrooms

Population

Average Occupancy

Latitude

Longitude

The original API feature names are MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, and Longitude.

Technology Stack

Machine Learning

Python, pandas, NumPy, scikit-learn, XGBoost, joblib

Backend

FastAPI, Pydantic, Uvicorn

Frontend

React, TypeScript, Vite, Tailwind CSS, shadcn/ui

Project Structure

HousePricePrediction/
├── backend/
│   └── main.py
├── frontend/
│   ├── src/
│   │   ├── components/ui/
│   │   │   └── glass-button.tsx
│   │   ├── App.tsx
│   │   ├── index.css
│   │   └── main.tsx
│   └── package.json
├── models/
│   ├── best_model.pkl
│   └── preprocessing_pipeline.pkl
├── notebooks/
├── src/
├── data/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore

Installation

Python

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Frontend

cd frontend
npm install

Running the Application

Backend

From the project root:

python backend\main.py

API:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs

Frontend

In another terminal:

cd frontend
npm run dev

Open the Vite URL shown in the terminal, normally http://localhost:5173.

API

POST /predict

Example request:

{
  "MedInc": 5.0,
  "HouseAge": 20.0,
  "AveRooms": 5.5,
  "AveBedrms": 1.0,
  "Population": 1000.0,
  "AveOccup": 3.0,
  "Latitude": 34.05,
  "Longitude": -118.25
}

Response:

{
  "predicted_price_usd": 157605.05
}

The frontend converts the returned USD value to an INR-formatted display value.

Validation

The dashboard rejects missing or non-numeric values and checks sensible ranges for income, age, rooms, bedrooms, population, occupancy, latitude, and longitude before calling the API.

Prediction Flow

User input
   ↓
React dashboard
   ↓
Input validation
   ↓
POST /predict
   ↓
FastAPI
   ↓
predict_from_values()
   ↓
Preprocessing pipeline
   ↓
Saved best model
   ↓
Prediction
   ↓
React dashboard
   ↓
INR display

Limitations

Prediction quality depends on the training data and available features.

The result is a model estimate, not a professional property valuation.

The local frontend depends on the FastAPI server being available.

INR display uses the configured USD-to-INR conversion rather than a live exchange-rate service.

Future Improvements

Deploy the frontend and backend.

Use a live currency exchange-rate source.

Add automated API and prediction tests.

Add richer model evaluation visualizations.

Improve input guidance and accessibility.

Add monitoring and structured logging.

Author

Praksham Kushwaha