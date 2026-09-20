import pandas as pd
import logging
import great_expectations as ge

# Set up logging configuration to track validation status
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_dataframe(
    df: pd.DataFrame, required_columns: list, target_col: str = None
) -> bool:
    """
    Validates a pandas DataFrame before feeding it into a machine learning model.

    Parameters:
    - df: The DataFrame to be checked.
    - required_columns: A list of mandatory columns that must exist.
    - target_col: The target column name (optional) to validate its values.
    """
    logger.info("Starting data validation process...")

    # 1. Check for the existence of mandatory columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Critical Error: The following required columns are missing from the data: {missing_cols}"
        )
    logger.info("✓ All required columns verified successfully.")

    # 2. Check for missing values (NaN / Null)
    missing_values_count = df.isnull().sum().sum()
    if missing_values_count > 0:
        logger.warning(
            f"Warning: Found {missing_values_count} missing values in the dataset."
        )
    else:
        logger.info("✓ Dataset is clean and completely free of missing values.")

    # 3. Validate target column range and values (if provided)
    if target_col and target_col in df.columns:
        unique_values = df[target_col].unique()
        logger.info(
            f"Discovered values for target column '{target_col}': {unique_values}"
        )

        # Example: Check if the target column is binary (0 or 1) for classification tasks
        invalid_targets = [val for val in unique_values if val not in [0, 1, 0.0, 1.0]]
        if invalid_targets:
            logger.warning(
                f"Warning: Unexpected values found in target column: {invalid_targets}"
            )

    logger.info(
        "Data validation completed successfully and the dataset is ready for training!"
    )
    return True


def validate_input_data(df: pd.DataFrame) -> dict:
    """
    Validates incoming dataframe using Great Expectations.
    Checks column types, missing rates, category bounds, and value ranges.
    """
    logger.info("Starting data validation using Great Expectations...")

    ge_df = ge.from_pandas(df)

    required_cols = [
        "order_purchase_timestamp",
        "estimated_delivery_time_days",
        "purchase_hour",
        "purchase_dayofweek",
    ]
    for col in required_cols:
        ge_df.expect_column_to_exist(col)

    ge_df.expect_column_values_to_be_of_type("purchase_hour", "int64")
    ge_df.expect_column_values_to_be_of_type("purchase_dayofweek", "int64")

    ge_df.expect_column_values_to_be_between("purchase_hour", min_value=0, max_value=23)
    ge_df.expect_column_values_to_be_between(
        "purchase_dayofweek", min_value=0, max_value=6
    )
    ge_df.expect_column_values_to_be_between(
        "estimated_delivery_time_days", min_value=0, max_value=365
    )
    ge_df.expect_column_values_to_not_be_null("purchase_hour")

    validation_result = ge_df.validate()
    is_success = validation_result.success

    logger.info(
        f"Validation completed. Status: {'SUCCESS' if is_success else 'FAILED'}"
    )
    return {"success": is_success, "results": validation_result.to_json_dict()}
