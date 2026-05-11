from __future__ import annotations

import pandas as pd

from .config import OPERATIONAL_THRESHOLDS, TARGET_COLUMN


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date")

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        df[col] = df[col].interpolate(method="linear", limit_direction="both")
        df[col] = df[col].fillna(df[col].median())

    return df


def create_operational_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t = OPERATIONAL_THRESHOLDS
    unsafe_rules = (
        (df["wind_speed_max_kmh"] >= t["wind_speed_max_kmh"])
        | (df["wind_gust_max_kmh"] >= t["wind_gust_max_kmh"])
        | (df["precipitation_mm"] >= t["precipitation_mm"])
        | (df["temperature_min_c"] <= t["temperature_min_c"])
        | (df["temperature_max_c"] >= t["temperature_max_c"])
    )
    df[TARGET_COLUMN] = unsafe_rules.astype(int)
    df["risk_label"] = df[TARGET_COLUMN].map({0: "safe", 1: "unsafe"})
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["temp_range_c"] = df["temperature_max_c"] - df["temperature_min_c"]
    df["wind_gust_factor"] = df["wind_gust_max_kmh"] / df["wind_speed_mean_kmh"].clip(lower=1)
    df["rain_flag"] = (df["precipitation_mm"] > 0.1).astype(int)
    df["heavy_rain_flag"] = (df["precipitation_mm"] >= 8).astype(int)
    df["wind_rain_interaction"] = df["wind_speed_max_kmh"] * df["precipitation_mm"]
    df["thermal_stress_index"] = abs(df["temperature_max_c"] - 24) + abs(df["temperature_min_c"] - 8)
    return df


def prepare_modeling_table(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_weather_data(df)
    df = create_operational_target(df)
    df = engineer_features(df)
    return df


def get_feature_columns() -> list[str]:
    return [
        "temperature_mean_c",
        "temperature_min_c",
        "temperature_max_c",
        "wind_speed_mean_kmh",
        "wind_speed_max_kmh",
        "wind_gust_max_kmh",
        "precipitation_mm",
        "humidity_pct",
        "pressure_hpa",
        "month",
        "day_of_year",
        "temp_range_c",
        "wind_gust_factor",
        "rain_flag",
        "heavy_rain_flag",
        "wind_rain_interaction",
        "thermal_stress_index",
    ]
