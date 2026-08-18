"""
backend/main.py

Purpose
-------
A minimal FastAPI backend for the House Price Prediction project.
"""

import logging
import sys
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from src.predict import predict_from_values  # noqa: E402


logger = logging.getLogger(__name__)

app = FastAPI(
    title="House Price Prediction API",
    description="Predicts California house prices from the existing trained model.",
    version="0.1.0",
)

# Allow the React/Vite frontend to communicate with FastAPI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    """Request body for POST /predict — exactly the 8 model features."""

    MedInc: float = Field(
        ...,
        description="Median income (tens of thousands of $)",
    )
    HouseAge: float = Field(
        ...,
        description="Median house age (years)",
    )
    AveRooms: float = Field(
        ...,
        description="Average rooms per household",
    )
    AveBedrms: float = Field(
        ...,
        description="Average bedrooms per household",
    )
    Population: float = Field(
        ...,
        description="Block group population",
    )
    AveOccup: float = Field(
        ...,
        description="Average occupants per household",
    )
    Latitude: float = Field(
        ...,
        description="Block group latitude",
    )
    Longitude: float = Field(
        ...,
        description="Block group longitude",
    )


class PredictionResponse(BaseModel):
    """Response body for POST /predict."""

    predicted_price_usd: float


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """
    Predict a house price from the given feature values.

    Calls the existing predict_from_values() function unchanged.
    """
    try:
        predicted_price = predict_from_values(
            MedInc=payload.MedInc,
            HouseAge=payload.HouseAge,
            AveRooms=payload.AveRooms,
            AveBedrms=payload.AveBedrms,
            Population=payload.Population,
            AveOccup=payload.AveOccup,
            Latitude=payload.Latitude,
            Longitude=payload.Longitude,
        )

    except (TypeError, ValueError) as validation_error:
        logger.warning(
            "predict: invalid input: %s",
            validation_error,
        )
        raise HTTPException(
            status_code=422,
            detail=str(validation_error),
        ) from validation_error

    except FileNotFoundError as missing_artifacts_error:
        logger.error(
            "predict: saved model/pipeline not found: %s",
            missing_artifacts_error,
        )
        raise HTTPException(
            status_code=503,
            detail=str(missing_artifacts_error),
        ) from missing_artifacts_error

    except RuntimeError as prediction_error:
        logger.error(
            "predict: prediction failed: %s",
            prediction_error,
        )
        raise HTTPException(
            status_code=500,
            detail=str(prediction_error),
        ) from prediction_error

    return PredictionResponse(
        predicted_price_usd=predicted_price
    )


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )