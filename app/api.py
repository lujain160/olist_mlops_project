from fastapi import FastAPI, HTTPException
import pandas as pd
from src.predict import make_predictions
import logging
from src.validation import validate_input_data

logger = logging.getLogger(__name__)
app = FastAPI(
    title="Olist Shipping Delay API",
    description="MLOps service for predicting shipping delays with data validation and logging",
    version="1.0.0",
)


@app.post("/predict")
def predict_endpoint(data: dict):
    """
    API endpoint to receive input data, validate it, and return the prediction.
    Handles bad inputs and service crashes gracefully.
    """
    try:
        if not data:
            raise HTTPException(status_code=400, detail="Input data cannot be empty.")

        input_df = pd.DataFrame([data])

        validation_output = validate_input_data(input_df)
        if not validation_output["success"]:
            logger.warning(f"Data vaidation failed for request: {data}")
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Input data failed validation checks.",
                    "details": validation_output["results"]["statistics"],
                },
            )

        prediction, probabilities = make_predictions(input_df)

        return {
            "status": "success",
            "prediction": (
                prediction.tolist() if hasattr(prediction, "tolist") else prediction
            ),
            "probabilities": probabilities,
        }
    except HTTPException as he:
        raise he

    except Exception as e:
        logging.error(f"Service error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
