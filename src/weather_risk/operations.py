from __future__ import annotations

import pandas as pd

from .config import OPERATIONAL_THRESHOLDS


def identify_risk_drivers(row: pd.Series) -> list[str]:
    """Return human-readable operational risk drivers for one weather scenario."""
    t = OPERATIONAL_THRESHOLDS
    drivers: list[str] = []

    if row["wind_speed_max_kmh"] >= t["wind_speed_max_kmh"]:
        drivers.append("High sustained wind")
    if row["wind_gust_max_kmh"] >= t["wind_gust_max_kmh"]:
        drivers.append("Extreme wind gusts")
    if row["precipitation_mm"] >= t["precipitation_mm"]:
        drivers.append("Heavy precipitation")
    if row["temperature_min_c"] <= t["temperature_min_c"]:
        drivers.append("Freezing conditions")
    if row["temperature_max_c"] >= t["temperature_max_c"]:
        drivers.append("Heat stress")

    return drivers or ["No threshold breach detected"]


def recommend_action(unsafe_probability: float | None, predicted_is_unsafe: int) -> str:
    """Map a model output to a decision-support recommendation."""
    if unsafe_probability is None:
        return "Review operational thresholds and site-specific constraints."
    if predicted_is_unsafe == 1 and unsafe_probability >= 0.75:
        return "Postpone weather-sensitive work or require senior HSE review."
    if predicted_is_unsafe == 1:
        return "Flag for planner review before confirming execution window."
    if unsafe_probability >= 0.35:
        return "Proceed with caution and monitor updated forecasts."
    return "Conditions look suitable for standard outdoor operations."


def build_decision_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a stakeholder-friendly table with risk drivers and recommendations."""
    result = df.copy()
    result["risk_drivers"] = result.apply(lambda row: ", ".join(identify_risk_drivers(row)), axis=1)
    return result
