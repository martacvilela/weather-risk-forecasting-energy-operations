from __future__ import annotations

import joblib
import pandas as pd

from .features import get_feature_columns, prepare_modeling_table


def load_model(path: str):
    return joblib.load(path)


def predict_weather_risk(model, raw_weather_df: pd.DataFrame) -> pd.DataFrame:
    features_df = prepare_modeling_table(raw_weather_df)
    X = features_df[get_feature_columns()]
    predictions = model.predict(X)
    result = raw_weather_df.copy()
    result["predicted_is_unsafe"] = predictions
    result["predicted_risk_label"] = pd.Series(predictions).map({0: "safe", 1: "unsafe"})
    if hasattr(model, "predict_proba"):
        result["unsafe_probability"] = model.predict_proba(X)[:, 1]
    return result
