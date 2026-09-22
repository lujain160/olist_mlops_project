import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal and datetime-based features from order timestamps.

    Args:
        df (pd.DataFrame): Raw orders dataframe.

    Returns:
        pd.DataFrame: Dataframe with added temporal features.
    """
    try:
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a valid pandas DataFrame.")
        if df.empty:
            logger.warning("Warning: Provided DataFrame is empty.")
            return df

        logger.info("Extracting temporal features...")
        df_feat = df.copy()

        # List of date columns to convert
        date_columns = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]

        # Convert existing columns to datetime objects safely
        for col in date_columns:
            if col in df_feat.columns:
                df_feat[col] = pd.to_datetime(df_feat[col], errors="coerce")

        # Feature 1: Expected delivery duration in days
        if (
            "order_estimated_delivery_date" in df_feat.columns
            and "order_purchase_timestamp" in df_feat.columns
        ):
            df_feat["estimated_delivery_time_days"] = (
                df_feat["order_estimated_delivery_date"]
                - df_feat["order_purchase_timestamp"]
            ).dt.total_seconds() / 86400.0

        # Feature 2: Purchase hour and day of week to capture ordering time patterns
        if "order_purchase_timestamp" in df_feat.columns:
            df_feat["purchase_hour"] = df_feat["order_purchase_timestamp"].dt.hour
            df_feat["purchase_dayofweek"] = df_feat[
                "order_purchase_timestamp"
            ].dt.dayofweek

        logger.info("Temporal features extracted successfully.")
        return df_feat

    except Exception as e:
        logger.error(
            f"Error occurred while extracting temporal features: {str(e)}",
            exc_info=True,
        )
        raise e


def create_target_variable(
    df: pd.DataFrame, target_col: str = "is_late"
) -> pd.DataFrame:
    """
    Create the binary target variable for delivery delay prediction.

    Args:
        df (pd.DataFrame): Dataframe containing order dates.
        target_col (str): Name of the target column to create.

    Returns:
        pd.DataFrame: Dataframe including the target column.
    """
    try:
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a valid pandas DataFrame.")
        if df.empty:
            logger.warning("Warning: Provided DataFrame is empty.")
            return df

        logger.info(f"Creating target variable '{target_col}'...")
        df_feat = df.copy()

        # FIXED: Corrected column name to match actual data
        required_cols = [
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
        missing_cols = [col for col in required_cols if col not in df_feat.columns]

        if missing_cols:
            logger.error(
                f"Missing columns required for target creation: {missing_cols}"
            )
            raise KeyError(f"The following columns are missing: {missing_cols}")

        df_feat[target_col] = (
            (
                df_feat["order_delivered_customer_date"]
                > df_feat["order_estimated_delivery_date"]
            )
            .fillna(0)
            .astype(int)
        )

        logger.info("Target variable created successfully.")
        return df_feat

    except Exception as e:
        logger.error(
            f"Error occurred while creating target variable: {str(e)}", exc_info=True
        )
        raise e
