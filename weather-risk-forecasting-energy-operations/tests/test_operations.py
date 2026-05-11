import pandas as pd

from weather_risk.operations import identify_risk_drivers, recommend_action


def test_identify_risk_drivers_detects_multiple_threshold_breaches():
    row = pd.Series(
        {
            "wind_speed_max_kmh": 50,
            "wind_gust_max_kmh": 70,
            "precipitation_mm": 12,
            "temperature_min_c": 4,
            "temperature_max_c": 30,
        }
    )
    drivers = identify_risk_drivers(row)
    assert "High sustained wind" in drivers
    assert "Extreme wind gusts" in drivers
    assert "Heavy precipitation" in drivers


def test_recommend_action_high_probability_unsafe():
    assert "Postpone" in recommend_action(0.9, predicted_is_unsafe=1)
