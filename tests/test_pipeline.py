import pytest
import pandas as pd
import numpy as np
from src.feature import extract_temporal_features, create_target_variable
from src.preprocess import clean_data
from src.validation import validate_dataframe


def test_extract_temporal_features():
    data = {
        "order_purchase_timestamp": ["2026-01-01 10:00:00"],
        "order_estimated_delivery_date": ["2026-01-05 10:00:00"],
    }

    df = pd.DataFrame(data)
    df_feat = extract_temporal_features(df)

    assert "purchase_hour" in df_feat.columns
    assert "purchase_dayofweek" in df_feat.columns
    assert "estimated_delivery_time_days" in df_feat.columns
    assert df_feat["purchase_hour"].iloc[0] == 10
    assert df_feat["estimated_delivery_time_days"].iloc[0] == 4.0


def test_create_target_variable():
    data = {
        "order_delivery_customer_date": ["2026-01-06 10:00:00", "2026-01-04 10:00:00"],
        "order_estimated_delivery_date": ["2026-01-05 10:00:00", "2026-01-05 10:00:00"],
    }
    df = pd.DataFrame(data)
    df_target = create_target_variable(df, target_col="is_late")

    assert "is_late" in df_target.columns
    assert df_target["is_late"].iloc[0] == 1
    assert df_target["is_late"].iloc[1] == 0


def test_clean_data():
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    cleaned_df = clean_data(df)
    assert not cleaned_df.empty
    assert list(cleaned_df.columns) == ["col1", "col2"]


def test_validate_dataframe_schema():
    df = pd.DataFrame(
        {"feature_a": [1, 2, 3], "feature_b": [4, 5, 6], "target": [0, 1, 0]}
    )
    required_cols = ["feature_a", "feature_b"]

    retult = validate_dataframe(df, required_columns=required_cols, target_col="target")
    assert retult is True


def test_validate_dataframe_missing_columns():
    df = pd.DataFrame({"feature_a": [1, 2, 3]})
    required_cols = ["feature_a", "missing_col"]

    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns=required_cols)


def test_model_prediction_structure():
    from sklearn.ensemble import RandomForestClassifier

    X_dummy = np.array([[10, 2, 4.0], [12, 3, 2.0]])
    y_dummy = np.array([0, 1])

    model = RandomForestClassifier(random_state=42)
    model.fit(X_dummy, y_dummy)

    preds = model.predict(X_dummy)
    assert len(preds) == 2
    assert set(preds).issubset({0, 1})
