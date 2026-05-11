from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from weather_risk.data import generate_synthetic_weather_data
from weather_risk.features import prepare_modeling_table, get_feature_columns
from weather_risk.modeling import train_and_evaluate, select_best_model

st.set_page_config(page_title="Weather Risk Forecasting", layout="wide")
st.title("Weather Risk Forecasting for Energy Operations")
st.write(
    "Prototype decision-support tool that classifies daily weather conditions as safe or unsafe "
    "for outdoor energy operations based on wind, precipitation and temperature risk drivers."
)

raw_df = generate_synthetic_weather_data(n_days=1460)
df = prepare_modeling_table(raw_df)
models, metrics = train_and_evaluate(df)
best_name = select_best_model(metrics)
best_model = models[best_name]

st.subheader("Model summary")
st.write(f"Best model by F1 score: **{best_name}**")
st.dataframe(pd.DataFrame(metrics).T[["accuracy", "precision", "recall", "f1"]])

st.subheader("Try a daily weather scenario")
col1, col2, col3 = st.columns(3)
with col1:
    wind_speed_max = st.slider("Max wind speed (km/h)", 0, 100, 35)
    wind_gust_max = st.slider("Max wind gust (km/h)", 0, 130, 50)
with col2:
    precipitation = st.slider("Precipitation (mm)", 0.0, 60.0, 4.0)
    humidity = st.slider("Humidity (%)", 10, 100, 65)
with col3:
    temp_min = st.slider("Min temperature (°C)", -15, 25, 6)
    temp_max = st.slider("Max temperature (°C)", 0, 50, 28)

scenario = pd.DataFrame(
    {
        "date": [pd.Timestamp("2026-01-01")],
        "temperature_mean_c": [(temp_min + temp_max) / 2],
        "temperature_min_c": [temp_min],
        "temperature_max_c": [temp_max],
        "wind_speed_mean_kmh": [max(wind_speed_max * 0.65, 1)],
        "wind_speed_max_kmh": [wind_speed_max],
        "wind_gust_max_kmh": [wind_gust_max],
        "precipitation_mm": [precipitation],
        "humidity_pct": [humidity],
        "pressure_hpa": [1013],
    }
)
scenario_model = prepare_modeling_table(scenario)
X = scenario_model[get_feature_columns()]
pred = best_model.predict(X)[0]
prob = best_model.predict_proba(X)[0, 1] if hasattr(best_model, "predict_proba") else None

label = "UNSAFE" if pred == 1 else "SAFE"
st.metric("Predicted operational class", label)
if prob is not None:
    st.metric("Estimated unsafe probability", f"{prob:.1%}")
