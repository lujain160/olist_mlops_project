import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.config import load_config
from src.data import load_data
import logging

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "evaluation.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def evaluate_model():
    try:
        config = load_config()
        models_dir = Path(config["Paths"]["model_dir"])
        model_path = models_dir / "final_model.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Please run train.py first."
            )
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        _, _, _, _, X_test, y_test = load_data()

        logger.info("Evaluating the model on test data....")
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info("\nConfusion Matrix:\n")
        logger.info(conf_matrix)
        logger.info("\nClassification Report:\n")
        logger.info(class_report)

        return acc, conf_matrix, class_report
    except Exception as e:
        logger.error(f"Error occurred during model evaluation: {str(e)}", exc_info=True)
        raise e


if __name__ == "__main__":
    evaluate_model()
