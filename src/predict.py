from pathlib import Path
import joblib
import pandas as pd
from src.config import load_config
from src.validation import validate_input_data
import logging
import time
import mlflow.sklearn

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "prediction.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_artifacts():
    """Load the saved model and preprocessor with path verification."""
    config = load_config()
    paths = config.get("Paths", {})

    mlflow.set_tracking_uri("http://localhost:5000")

    model_uri = "models:/OlistDeliveryModel/latest"
    logger.info(f"Loading model directly from MLflow Registry: {model_uri}")

    try:
        model = mlflow.sklearn.load_model(model_uri)
        model_version = "v1.0.0"
    except Exception as e:
        logger.error(f"Failed to load model from MLflow Registry: {e}")
        raise e

    model_dir = Path(paths.get("model_dir", "models"))
    preprocessor_path = model_dir / "preprocessor.joblib"
    preprocessor = (
        joblib.load(preprocessor_path) if preprocessor_path.exists() else None
    )

    return model, preprocessor, model_version


def make_predictions(input_data: pd.DataFrame):
    """Make predictions and extract probabilities if available."""
    start_time = time.time()
    try:
        if input_data is None or input_data.empty:
            raise ValueError("Input data is empty or None.")

        validation_output = validate_input_data(input_data)
        if not validation_output["success"]:
            logger.error("Data validation FAILED. Rejecting prediction request.")
            return {
                "error": "Bad input data: Data failed validation check.",
                "validation_details": validation_output["results"],
            }, None

        logger.info("Data validation passed successfully.")
        model, preprocessor, model_version = load_artifacts()

        # Apply transform only if preprocessor exists
        if preprocessor is not None:
            processed_data = preprocessor.transform(input_data)
        else:
            processed_data = input_data

        predictions = model.predict(processed_data)

        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(processed_data)[:, 1].tolist()
        latency = (time.time() - start_time) * 1000
        logger.info(
            f"Prediction successful | Model Version: {model_version} | "
            f"Input Shape: {input_data.shape} | Predictions: {predictions.tolist()} | "
            f"Latency: {latency:.2f}ms"
        )
        return predictions, probabilities

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return {"error": str(e)}, None


if __name__ == "__main__":
    print("Predict script is ready.")
