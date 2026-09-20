import pandas as pd
from pathlib import Path
import joblib
from src.config import load_config
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_preprocessors(model_dir: str = "models"):
    try:

        config = load_config()
        paths = config.get("Paths", {})

        model_dir = Path(paths.get("model_dir", "models"))
        preprocessor_path = Path(model_dir) / "preprocessor.joblib"

        if preprocessor_path.exists():
            logger.info(f"Loading preprocessor from {preprocessor_path}")
            return joblib.load(preprocessor_path)

        logger.warning(
            f"Preprocessor not found at {preprocessor_path}. Proceeding without it."
        )
        return None

    except Exception as e:
        logger.error(
            f"Error occurred while loading preprocessor: {str(e)}", exc_info=True
        )
        return None


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a valid pandas DataFrame.")
        if df.empty:
            logger.warning("Warning: Provided DataFrame for cleaning is empty.")
            return df

        logger.info("Starting data cleaning process...")
        df_clean = df.copy()

        logger.info("Data cleaned successfully.")
        return df_clean

    except Exception as e:
        logger.error(f"Error occurred during data cleaning: {str(e)}", exc_info=True)
        raise e


def preprocessor_features(df: pd.DataFrame, preprocessor=None) -> pd.DataFrame:
    try:
        df_processed = clean_data(df)

        if preprocessor is None:
            preprocessor = load_preprocessors()

        if preprocessor is not None:
            logger.info("Applying fitted preprocessor transformation on data...")

            df_processed = preprocessor.transform(df_processed)
            logger.info("Preprocessing transformation applied successfully.")
        else:
            logger.info("No preprocessor applied; returning cleaned dataframe.")
        return df_processed

    except Exception as e:
        logger.error(
            f"Error occurred during feature preprocessing: {str(e)}", exc_info=True
        )
        raise e
