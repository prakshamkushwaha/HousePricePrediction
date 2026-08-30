PROJECT_DOCUMENTATION.md


Project Documentation — Automated Property Value Prediction
1. Introduction
Automated Property Value Prediction is a machine learning application for estimating house values from property and geographic characteristics.

The project combines a Python machine learning pipeline, a FastAPI backend, and a React + TypeScript frontend.

2. Problem Statement
The project uses property and location characteristics to estimate house values with regression models. The objective is to compare several models, select the strongest one based on evaluation results, and make that model available through a web application.

3. Objectives
Prepare housing data for machine learning.

Apply the project's preprocessing workflow consistently.

Train and compare five regression models.

Evaluate model performance.

Persist the selected model and preprocessing pipeline.

Provide a reusable prediction function.

Expose prediction through a REST API.

Provide a web dashboard for predictions.

Validate user input before API requests.

4. Dataset and Features
The application accepts eight features:

API Feature	Dashboard Label
MedInc	Median Income
HouseAge	House Age
AveRooms	Average Rooms
AveBedrms	Average Bedrooms
Population	Population
AveOccup	Average Occupancy
Latitude	Latitude
Longitude	Longitude
5. Data Preprocessing
The project contains a dedicated preprocessing module. It includes the custom OutlierCapper class and the preprocessing workflow used by the prediction pipeline.

The fitted preprocessing pipeline is persisted as:

models/preprocessing_pipeline.pkl
The saved pipeline is reused during prediction so new inputs receive the same preprocessing treatment used during model development.

6. Machine Learning Models
Five regression models are evaluated:

Linear Regression

Ridge

Random Forest

Gradient Boosting

XGBoost

7. Model Evaluation
The current dashboard reports these R² results:

Model	R²
Linear Regression	64.70%
Ridge	64.70%
Random Forest	80.37%
Gradient Boosting	77.68%
XGBoost	77.56%
The documented Random Forest RMSE is 0.507.

8. Best Model
Random Forest is currently selected because it has the highest displayed R² score of 80.37% among the five evaluated models.

9. Model Persistence
The trained artifacts are stored in:

models/
├── best_model.pkl
└── preprocessing_pipeline.pkl
The prediction application loads these saved artifacts rather than retraining a model for every request.

10. Prediction Pipeline
The prediction flow is:

Input values
    ↓
predict_from_values()
    ↓
Saved preprocessing pipeline
    ↓
Saved best model
    ↓
Predicted value
The FastAPI endpoint calls the existing predict_from_values() function instead of implementing a second prediction process.

11. FastAPI Backend
The backend is implemented in:

backend/main.py
It exposes:

POST /predict
The request is validated using Pydantic and contains the eight model features.

The response has the form:

{
  "predicted_price_usd": 0
}
Interactive documentation is available at:

http://127.0.0.1:8000/docs
when the backend is running.

12. React Frontend
The frontend is a Vite-based React + TypeScript application.

The dashboard includes:

Property input form

Prediction display

Predict button

Reset button

KPI cards

Model performance comparison

Loading state

Error state

Input validation

INR-formatted display

Tailwind CSS is used for styling and the existing GlassButton component is used for the main prediction action.

13. Frontend Validation
Before calling the API, the frontend checks that values are present and numeric.

It also validates:

Median Income > 0

House Age >= 0

Average Rooms > 0

Average Bedrooms > 0

Population > 0

Average Occupancy > 0

Latitude between -90 and 90

Longitude between -180 and 180

Invalid input prevents the API request and shows an error message.

14. API Communication
The frontend sends a JSON POST request to:

http://127.0.0.1:8000/predict
Example:

{
  "MedInc": 5,
  "HouseAge": 20,
  "AveRooms": 5.5,
  "AveBedrms": 1,
  "Population": 1000,
  "AveOccup": 3,
  "Latitude": 34.05,
  "Longitude": -118.25
}
The backend returns the prediction in USD. The frontend formats the returned number for INR display.

15. Currency Display
The API field remains predicted_price_usd. Currency conversion is handled at the presentation layer in the frontend.

The current dashboard uses a configured USD-to-INR conversion value rather than a live exchange-rate service.

16. Project Architecture
flowchart TD
    A[User] --> B[React + TypeScript Dashboard]
    B --> C[Input Validation]
    C --> D[POST /predict]
    D --> E[FastAPI]
    E --> F[predict_from_values]
    F --> G[Preprocessing Pipeline]
    G --> H[Saved Best Model]
    H --> I[USD Prediction]
    I --> B
    B --> J[INR Display]
17. Project Structure
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
└── README.md
18. Installation and Setup
Create the Python environment:

python -m venv .venv
.venv\Scripts\Activate.ps1
Install Python dependencies:

pip install -r requirements.txt
Install frontend dependencies:

cd frontend
npm install
19. Running the Application
Start the backend from the project root:

python backend\main.py
Then start the frontend in another terminal:

cd frontend
npm run dev
Both services need to be running for live frontend predictions.

20. Testing
Important application checks include:

Empty fields are rejected.

Non-numeric input is rejected.

Invalid latitude and longitude values are rejected.

Valid values produce a real API request.

The loading state appears during prediction.

The returned prediction is displayed.

Reset clears the inputs and previous prediction.

A stopped backend produces a clear frontend error.

The FastAPI /docs page can be used to test the API directly.

21. Results
Current documented results:

Best Model: Random Forest

R²: 80.37%

RMSE: 0.507

Model comparison:

Linear Regression     64.70%
Ridge                  64.70%
Random Forest          80.37%
Gradient Boosting      77.68%
XGBoost                77.56%
22. Limitations
The model's performance depends on the training dataset and the features available to it. The result should therefore be treated as a machine learning estimate rather than a formal property valuation.

The local development frontend also depends on the FastAPI backend being available.

The INR display is a presentation-layer conversion and does not represent a live financial exchange rate.

23. Future Improvements
Possible improvements include:

Deployment of frontend and backend.

Live currency exchange-rate integration.

Automated unit and integration testing.

Additional model evaluation visualizations.

Improved accessibility and input guidance.

Monitoring and structured logging.

More detailed prediction explanations.

These are future improvements rather than existing features.

24. Conclusion
The project connects a complete machine learning workflow to a usable web application.

Five regression models are evaluated, with Random Forest currently producing the strongest displayed R² result. The selected model and preprocessing pipeline are persisted and reused for prediction. FastAPI exposes the existing prediction function through a simple HTTP endpoint, while the React dashboard provides validation, prediction, reset functionality, and model-performance information.