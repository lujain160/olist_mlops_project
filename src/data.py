import yaml
import pandas as pd
from pathlib import Path
from src.config import load_config
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data():
    try:
        logger.info("Loading configuration for the data paths...")
        config = load_config()
        paths = config["Paths"]
        target_col = config["model_params"]["target_col"]

        logger.info("Loading feature datasets...")
        X_train = pd.read_csv(paths["train_feature"])
        X_val = pd.read_csv(paths["validation_feature"])
        X_test = pd.read_csv(paths["test_feature"])

        logger.info("Loading label datasets...")
        y_train = pd.read_csv(paths["train_labels"])[target_col]
        y_val = pd.read_csv(paths["validation_labels"])[target_col]
        y_test = pd.read_csv(paths["test_labels"])[target_col]

        logger.info("All datasets loaded successfully.")
        return X_train, y_train, X_val, y_val, X_test, y_test
    except Exception as e:
        logger.error("Error occurred while loading datasets: {str(e)}", exc_info=True)
        raise e
