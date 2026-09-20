import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from src.config import load_config
from src.data import load_data
from src.validation import validate_dataframe
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model():
    try:
        logger.info("Loading configuration...")
        config = load_config()

        random_state = config["model_params"].get("random_state", 42)
        n_estimators = config["model_params"].get("n_estimators", 100)

        models_dir = Path(config["Paths"]["model_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading datasets....")
        X_train, y_train, X_val, y_val, X_test, y_test = load_data()

        logger.info("Validating training data...")

        validate_dataframe(
            df=X_train, required_columns=list(X_train.columns), target_col=None
        )

        logger.info("Random Forest......")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

        logger.info("Training the model")
        model.fit(X_train, y_train)

        model_path = models_dir / "final_model.joblib"
        logger.info(f"Saving trained model to {model_path}....")
        joblib.dump(model, model_path)

        logger.info(f"Model successfully saved at {model_path}")
        return model

    except Exception as e:
        logger.error(f"Error occurred during model training: {str(e)}", exc_info=True)
        raise e


if __name__ == "__main__":
    train_model()
