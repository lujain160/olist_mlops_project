from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import pandas as pd

from src.predict import make_predictions
from src.validation import validate_input_data

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Olist Shipping Delay API",
    description="MLOps service for predicting shipping delays with data validation and logging",
    version="1.0.0",
)


class OrderRequest(BaseModel):
    order_purchase_timestamp: str
    estimated_delivery_time_days: float = Field(..., ge=0, le=365)
    purchase_hour: int = Field(..., ge=0, le=23)
    purchase_dayofweek: int = Field(..., ge=0, le=6)


class BatchRequest(BaseModel):
    orders: List[OrderRequest]


class PredictionResponse(BaseModel):
    status: str
    prediction: int
    probability: Optional[float] = None
    model_version: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "service": "Olist Shipping Delay API",
        "version": app.version,
    }


@app.get("/model-info")
def model_info():
    return {
        "name": "OlistDeliveryModel",
        "version": "latest",
        "source": "mlflow-registry",
        "status": "ready",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(order: OrderRequest):
    """Predict a single order's delivery delay probability."""
    try:
        input_df = pd.DataFrame([order.model_dump()])

        validation_output = validate_input_data(input_df)
        if not validation_output["success"]:
            logger.warning(f"Data validation failed for request: {order.model_dump()}")
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Input data failed validation checks.",
                    "details": validation_output["results"],
                },
            )

        predictions, probabilities = make_predictions(input_df)

        if isinstance(predictions, dict) and "error" in predictions:
            raise HTTPException(status_code=400, detail=predictions["error"])

        prediction_value = int(predictions[0])
        probability_value = float(probabilities[0]) if probabilities else 0.0

        return {
            "status": "success",
            "prediction": prediction_value,
            "probability": probability_value,
            "model_version": "latest",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Service error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict/batch")
def batch_predict_endpoint(batch: BatchRequest):
    """Predict a batch of orders in one request."""
    try:
        if not batch.orders:
            raise HTTPException(
                status_code=400, detail="At least one order is required."
            )

        input_df = pd.DataFrame([order.model_dump() for order in batch.orders])

        validation_output = validate_input_data(input_df)
        if not validation_output["success"]:
            logger.warning("Batch validation failed")
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "One or more records failed validation checks.",
                    "details": validation_output["results"],
                },
            )

        predictions, probabilities = make_predictions(input_df)

        if isinstance(predictions, dict) and "error" in predictions:
            raise HTTPException(status_code=400, detail=predictions["error"])

        response_predictions = [int(item) for item in predictions]
        response_probabilities = (
            [float(item) for item in probabilities] if probabilities else []
        )

        return {
            "status": "success",
            "predictions": response_predictions,
            "probabilities": response_probabilities,
            "model_version": "latest",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
