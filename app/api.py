from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from typing import List, Union
from src.predict import make_predictions
import logging
from src.validation import validate_input_data

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Olist Shipping Delay API",
    description="MLOps service for predicting shipping delays with data validation and logging",
    version="1.0.0",
)


# Pydantic Schemas for Request and Response Validation
class OrderRequest(BaseModel):
    # Add or adjust the exact fields required by your Olist dataset features
    order_id: str
    price: float
    freight_value: float


class PredictionResponse(BaseModel):
    status: str
    prediction: Union[List, int, float]
    probabilities: Union[List, float]
    model_version: str


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    return {
        "model_name": "OlistDeliveryModel",
        "version": "v1",
        "description": "Inference pipeline for predicting Olist shipping delays",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(order: OrderRequest):
    """
    API endpoint to receive input data, validate it using Great Expectations,
    and return the prediction, probabilities, and model version.
    """
    try:
        # Convert Pydantic model to DataFrame for validation and prediction
        input_df = pd.DataFrame([order.dict()])

        validation_output = validate_input_data(input_df)
        if not validation_output["success"]:
            logger.warning(f"Data validation failed for request: {order}")
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Input data failed validation checks.",
                    "details": validation_output["results"]["statistics"],
                },
            )

        # Make predictions using your inference module
        prediction, probabilities = make_predictions(input_df)
        model_version = "v1"  # Can be dynamically retrieved from artifacts or config

        return {
            "status": "success",
            "prediction": (
                prediction.tolist() if hasattr(prediction, "tolist") else prediction
            ),
            "probabilities": (
                probabilities.tolist()
                if hasattr(probabilities, "tolist")
                else probabilities
            ),
            "model_version": model_version,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Service error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict/batch")
def batch_predict_endpoint(orders: List[OrderRequest]):
    """Batch prediction endpoint for multiple orders at once."""
    try:
        input_df = pd.DataFrame([o.dict() for o in orders])
        prediction, probabilities = make_predictions(input_df)

        return {
            "status": "success",
            "predictions": (
                prediction.tolist() if hasattr(prediction, "tolist") else prediction
            ),
            "model_version": "v1",
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
