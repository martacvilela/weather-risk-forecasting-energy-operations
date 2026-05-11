from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE


def generate_synthetic_weather_data(n_days: int = 1460, start_date: str = "2020-01-01") -> pd.DataFrame:
    """Generate realistic daily weather data for an energy operations use case.

    The data is synthetic but designed to mimic seasonal patterns, operationally
    relevant extreme events, and noisy measurements found in real weather feeds.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    day_of_year = dates.dayofyear.to_numpy()

    seasonal_temp = 16 + 11 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temperature_mean_c = seasonal_temp + rng.normal(0, 3.0, n_days)
    temperature_min_c = temperature_mean_c - rng.uniform(4, 9, n_days)
    temperature_max_c = temperature_mean_c + rng.uniform(5, 11, n_days)

    seasonal_wind = 18 + 5 * np.sin(2 * np.pi * (day_of_year + 30) / 365)
    wind_speed_mean_kmh = np.clip(seasonal_wind + rng.normal(0, 5, n_days), 2, None)
    wind_speed_max_kmh = wind_speed_mean_kmh + rng.gamma(shape=2.2, scale=4.0, size=n_days)
    wind_gust_max_kmh = wind_speed_max_kmh + rng.gamma(shape=2.0, scale=5.0, size=n_days)

    rain_probability = 0.18 + 0.12 * np.cos(2 * np.pi * (day_of_year - 20) / 365)
    rain_event = rng.binomial(1, np.clip(rain_probability, 0.05, 0.45), n_days)
    precipitation_mm = rain_event * rng.gamma(shape=1.6, scale=5.0, size=n_days)

    humidity_pct = np.clip(55 + 0.9 * precipitation_mm + rng.normal(0, 12, n_days), 20, 100)
    pressure_hpa = 1015 - 0.15 * wind_speed_max_kmh - 0.5 * precipitation_mm + rng.normal(0, 5, n_days)

    # Inject a small number of compound high-risk events, common in operational planning.
    event_idx = rng.choice(n_days, size=max(12, n_days // 55), replace=False)
    wind_gust_max_kmh[event_idx] += rng.uniform(20, 45, len(event_idx))
    precipitation_mm[event_idx] += rng.uniform(10, 35, len(event_idx))
    pressure_hpa[event_idx] -= rng.uniform(8, 18, len(event_idx))

    df = pd.DataFrame(
        {
            "date": dates,
            "temperature_mean_c": temperature_mean_c,
            "temperature_min_c": temperature_min_c,
            "temperature_max_c": temperature_max_c,
            "wind_speed_mean_kmh": wind_speed_mean_kmh,
            "wind_speed_max_kmh": wind_speed_max_kmh,
            "wind_gust_max_kmh": wind_gust_max_kmh,
            "precipitation_mm": precipitation_mm,
            "humidity_pct": humidity_pct,
            "pressure_hpa": pressure_hpa,
        }
    )

    missing_cols = ["temperature_mean_c", "wind_speed_mean_kmh", "humidity_pct", "pressure_hpa"]
    for col in missing_cols:
        missing_mask = rng.random(n_days) < 0.015
        df.loc[missing_mask, col] = np.nan

    return df


def save_raw_dataset(path: str | None = None) -> pd.DataFrame:
    from .config import DATA_RAW_DIR

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_weather_data()
    output_path = DATA_RAW_DIR / "synthetic_weather_operations.csv" if path is None else path
    df.to_csv(output_path, index=False)
    return df
