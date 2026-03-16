# house-pricing

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
import os

app = FastAPI(
    title="🏠 House Pricing Prediction API",
    description="Predict house prices using a trained ML model. Supports single and batch predictions.",
    version="1.0.0",
)

# ----------------------------
# Load model (or use dummy for demo)
# ----------------------------
MODEL_PATH = "model/house_model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()


# ----------------------------
# Schemas
# ----------------------------
class HouseInput(BaseModel):
    area_sqft: float = Field(..., gt=0, description="Total area in square feet")
    bedrooms: int = Field(..., ge=1, le=20, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=1, le=20, description="Number of bathrooms")
    floors: int = Field(..., ge=1, le=10, description="Number of floors")
    house_age_years: int = Field(..., ge=0, description="Age of the house in years")
    garage: bool = Field(..., description="Has garage or not")
    garden: bool = Field(..., description="Has garden or not")
    location_type: str = Field(..., description="Location type: urban, suburban, or rural")
    distance_to_city_km: float = Field(..., ge=0, description="Distance to city center in km")
    nearby_schools: int = Field(..., ge=0, description="Number of nearby schools")

    class Config:
        json_schema_extra = {
            "example": {
                "area_sqft": 1800,
                "bedrooms": 3,
                "bathrooms": 2,
                "floors": 2,
                "house_age_years": 10,
                "garage": True,
                "garden": True,
                "location_type": "suburban",
                "distance_to_city_km": 12.5,
                "nearby_schools": 3
            }
        }

class PredictionResponse(BaseModel):
    predicted_price_usd: float
    price_range_low: float
    price_range_high: float
    confidence_level: str
    input_summary: dict

class BatchRequest(BaseModel):
    houses: List[HouseInput]

class BatchResponse(BaseModel):
    total_houses: int
    predictions: List[PredictionResponse]


# ----------------------------
# Helper: Feature Engineering
# ----------------------------
LOCATION_MAP = {"urban": 2, "suburban": 1, "rural": 0}

def extract_features(data: HouseInput) -> np.ndarray:
    location_score = LOCATION_MAP.get(data.location_type.lower(), 1)
    return np.array([
        data.area_sqft,
        data.bedrooms,
        data.bathrooms,
        data.floors,
        data.house_age_years,
        int(data.garage),
        int(data.garden),
        location_score,
        data.distance_to_city_km,
        data.nearby_schools
    ])

def rule_based_prediction(data: HouseInput) -> float:
    """Fallback rule-based estimator when no model is loaded."""
    location_score = LOCATION_MAP.get(data.location_type.lower(), 1)
    base = data.area_sqft * 120
    base += data.bedrooms * 8000
    base += data.bathrooms * 6000
    base += data.floors * 4000
    base -= data.house_age_years * 500
    base += int(data.garage) * 12000
    base += int(data.garden) * 8000
    base += location_score * 20000
    base -= data.distance_to_city_km * 1500
    base += data.nearby_schools * 3000
    return max(base, 50000)

def build_response(data: HouseInput, predicted_price: float) -> PredictionResponse:
    margin = predicted_price * 0.08
    if predicted_price > 500000:
        confidence = "High"
    elif predicted_price > 200000:
        confidence = "Medium"
    else:
        confidence = "Low"

    return PredictionResponse(
        predicted_price_usd=round(predicted_price, 2),
        price_range_low=round(predicted_price - margin, 2),
        price_range_high=round(predicted_price + margin, 2),
        confidence_level=confidence,
        input_summary={
            "area_sqft": data.area_sqft,
            "bedrooms": data.bedrooms,
            "bathrooms": data.bathrooms,
            "location_type": data.location_type,
            "house_age_years": data.house_age_years,
        }
    )


# ----------------------------
# Routes
# ----------------------------
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "api": "House Pricing Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "single_prediction": "/predict",
            "batch_prediction": "/predict/batch",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/model/info", tags=["Model"])
def model_info():
    return {
        "model_type": type(model).__name__ if model else "Rule-based fallback",
        "model_loaded": model is not None,
        "features": [
            "area_sqft", "bedrooms", "bathrooms", "floors",
            "house_age_years", "garage", "garden",
            "location_type", "distance_to_city_km", "nearby_schools"
        ],
        "output": "Predicted house price in USD",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: HouseInput):
    """
    Predict the price of a single house based on its features.
    """
    try:
        if model:
            features = extract_features(data).reshape(1, -1)
            price = float(model.predict(features)[0])
        else:
            price = rule_based_prediction(data)

        return build_response(data, price)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(request: BatchRequest):
    """
    Predict house prices for a batch of houses in one request.
    Maximum 100 houses per batch.
    """
    if len(request.houses) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100 houses.")

    try:
        results = []
        for house in request.houses:
            if model:
                features = extract_features(house).reshape(1, -1)
                price = float(model.predict(features)[0])
            else:
                price = rule_based_prediction(house)
            results.append(build_response(house, price))

        return BatchResponse(
            total_houses=len(results),
            predictions=results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
