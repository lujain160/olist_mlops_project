import logging
import os
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from src.config import load_config
from src.monitoring import monitor
from src.validation import validate_input_data

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "prediction.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_latest_model_version(model_name: str) -> str:
    """Return the latest registered version for a model name."""
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No registered model versions found for '{model_name}'.")
    return max(int(v.version) for v in versions)


def load_artifacts():
    """Load the model and preprocessor from MLflow registry or fallback local files."""
    config = load_config()
    paths = config.get("Paths", {})

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    model_name = os.getenv("MLFLOW_MODEL_NAME", "OlistDeliveryModel")
    model_version = os.getenv("MLFLOW_MODEL_VERSION", "latest")

    model_dir = Path(paths.get("model_dir", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        if model_version == "latest":
            model_version = get_latest_model_version(model_name)

        model_uri = f"models:/{model_name}/{model_version}"
        logger.info(f"Loading model from MLflow Registry: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model version {model_version} from MLflow Registry.")
    except Exception as exc:
        logger.warning(
            f"Failed to load model from MLflow Registry: {exc}. Falling back to local model artifact."
        )
        local_model_path = model_dir / "final_model.joblib"
        if not local_model_path.exists():
            raise FileNotFoundError(
                f"Could not load model from registry or local artifact: {local_model_path}"
            )
        model = joblib.load(local_model_path)
        model_version = "local"

    preprocessor_path = model_dir / "preprocessor.joblib"
    preprocessor = (
        joblib.load(preprocessor_path) if preprocessor_path.exists() else None
    )

    return model, preprocessor, str(model_version)


def make_predictions(input_data: pd.DataFrame):
    """Make predictions and return probabilities for each observed order."""
    start_time = time.time()
    try:
        if input_data is None or input_data.empty:
            raise ValueError("Input data is empty or None.")

        validation_output = validate_input_data(input_data)
        if not validation_output["success"]:
            logger.error("Data validation FAILED. Rejecting prediction request.")
            latency = (time.time() - start_time) * 1000
            monitor.track_request(latency=latency, is_error=True)
            return (
                {
                    "error": "Bad input data: Data failed validation check.",
                    "validation_details": validation_output["results"],
                },
                None,
                "unknown",
            )

        logger.info("Data validation passed successfully.")
        model, preprocessor, model_version = load_artifacts()

        if preprocessor is not None:
            processed_data = preprocessor.transform(input_data)
        else:
            processed_data = input_data

        predictions = model.predict(processed_data)
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(processed_data)[:, 1].tolist()

        latency = (time.time() - start_time) * 1000
        monitor.track_request(latency=latency, is_error=False)

        if hasattr(predictions, "__len__") and len(predictions) > 0:
            pred_val = int(predictions[0])
            prob_val = float(probabilities[0]) if probabilities else 0.0
        else:
            pred_val = int(predictions)
            prob_val = float(probabilities) if probabilities else 0.0

        monitor.log_predictions_for_evaluation(
            input_data=input_data.iloc[0].to_dict() if not input_data.empty else {},
            prediction=pred_val,
            probability=prob_val,
        )

        logger.info(
            f"Prediction successful | Model Version: {model_version} | "
            f"Input Shape: {input_data.shape} | Predictions: {predictions.tolist()} | "
            f"Latency: {latency:.2f}ms"
        )
        return predictions, probabilities, model_version

    except Exception as exc:
        latency = (time.time() - start_time) * 1000
        monitor.track_request(latency=latency, is_error=True)
        logger.error(f"Error during prediction: {str(exc)}", exc_info=True)
        return {"error": str(exc)}, None, "unknown"


if __name__ == "__main__":
    print("Predict script is ready.")
