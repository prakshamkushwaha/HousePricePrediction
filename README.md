# House Price Prediction using Machine Learning

## Project Overview
This project focuses on predicting house prices using Machine Learning techniques. Given features of a property (such as location, size, number of rooms, and other relevant attributes), the goal is to build a model that can estimate its market price. The project is structured to support the full ML lifecycle — from raw data ingestion and preprocessing, through model training and evaluation, to eventual result visualization.

## Project Objectives
- Analyze and understand housing data to identify factors that influence price
- Clean and preprocess raw housing data into a model-ready format
- Build and evaluate machine learning models for price prediction
- Provide clear visual reports and insights derived from the data and models
- Maintain a clean, modular, and reproducible project structure

## Key Features
- Organized project structure separating raw data, processed data, source code, and reports
- Centralized logging configured in `main.py` for consistent output tracking
- Dedicated folders for exploratory analysis (`notebooks/`), reusable code (`src/`), and generated visuals (`reports/figures/`)
- Placeholder `dashboard/` folder reserved for future result visualization
- Environment-based configuration support via `.env`

## Machine Learning Workflow
The intended end-to-end workflow for this project is as follows:
1. **Data Collection** — Gather raw housing data and store it in `data/raw/`
2. **Data Preprocessing** — Clean, transform, and engineer features; save output to `data/processed/`
3. **Exploratory Data Analysis (EDA)** — Analyze patterns and relationships using notebooks in `notebooks/`
4. **Model Training** — Train regression models on the processed data (to be implemented in `src/`)
5. **Model Evaluation** — Assess model performance using appropriate regression metrics
6. **Reporting** — Generate figures and summaries of findings in `reports/figures/`
7. **Visualization / Dashboard** — Present results through a dashboard (planned, in `dashboard/`)

> Note: This workflow describes the planned pipeline. See **Current Project Status** below for what has actually been implemented so far.

## Technology Stack
- **Language:** Python
- **Data Handling:** pandas, numpy
- **Machine Learning:** scikit-learn
- **Visualization:** matplotlib, seaborn
- **Notebooks:** Jupyter
- **Model Persistence:** joblib
- **Configuration:** python-dotenv

## Project Folder Structure
```
HousePricePrediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── dashboard/
├── models/
├── reports/
│   └── figures/
├── tests/
├── requirements.txt
├── README.md
├── .gitignore
└── main.py
```

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd HousePricePrediction
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Project
Run the main entry point:
```bash
python main.py
```
At this stage, running `main.py` displays a welcome message and initializes logging. It serves as the foundation that future data loading, preprocessing, training, and evaluation logic will be built on top of.

## Current Project Status
This project is currently in the **setup phase**. The folder structure, base configuration files (`requirements.txt`, `.gitignore`), and the `main.py` entry point (with logging and a welcome message) are in place.

**Not yet implemented:**
- Data collection / raw dataset
- Data preprocessing and feature engineering
- Model training
- Model evaluation
- Dashboard / visualization layer

## Future Improvements
- Source and add the raw housing dataset to `data/raw/`
- Implement data cleaning and preprocessing pipeline
- Perform exploratory data analysis in `notebooks/`
- Implement and train regression models in `src/`
- Evaluate models using metrics such as RMSE and R²
- Generate and store visual reports in `reports/figures/`
- Build out the `dashboard/` for interactive result visualization
- Add unit tests in `tests/`
