from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from weather_risk.data import generate_synthetic_weather_data
from weather_risk.features import get_feature_columns, prepare_modeling_table
from weather_risk.modeling import select_best_model, train_and_evaluate
from weather_risk.operations import identify_risk_drivers, recommend_action


@st.cache_data(show_spinner=False)
def load_training_data() -> pd.DataFrame:
    raw_df = generate_synthetic_weather_data(n_days=1460)
    return prepare_modeling_table(raw_df)


@st.cache_resource(show_spinner=False)
def fit_models(df: pd.DataFrame):
    models, metrics = train_and_evaluate(df)
    best_name = select_best_model(metrics)
    return models, metrics, best_name


st.set_page_config(page_title="Weather Risk Forecasting", layout="wide")
st.title("Weather Risk Forecasting for Energy Operations")
st.caption("Applied ML prototype for weather-sensitive planning in energy and construction operations.")

st.write(
    "This dashboard converts daily weather variables into an operational risk signal. "
    "It is designed as a decision-support prototype: the model flags potentially unsafe days, "
    "explains the main risk drivers and suggests a planning action."
)

df = load_training_data()
models, metrics, best_name = fit_models(df)
best_model = models[best_name]

metric_cols = st.columns(4)
metrics_df = pd.DataFrame(metrics).T[["accuracy", "precision", "recall", "f1"]]
with metric_cols[0]:
    st.metric("Best model", best_name.replace("_", " ").title())
with metric_cols[1]:
    st.metric("F1 score", f"{metrics_df.loc[best_name, 'f1']:.3f}")
with metric_cols[2]:
    st.metric("Recall unsafe", f"{metrics_df.loc[best_name, 'recall']:.3f}")
with metric_cols[3]:
    unsafe_rate = df["is_unsafe"].mean()
    st.metric("Unsafe days in dataset", f"{unsafe_rate:.1%}")

with st.expander("Model comparison"):
    st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)

st.subheader("Scenario tester")
st.write("Adjust a daily weather scenario and inspect the model prediction plus operational interpretation.")

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
pred = int(best_model.predict(X)[0])
prob = float(best_model.predict_proba(X)[0, 1]) if hasattr(best_model, "predict_proba") else None
label = "UNSAFE" if pred == 1 else "SAFE"

result_cols = st.columns(3)
with result_cols[0]:
    st.metric("Predicted class", label)
with result_cols[1]:
    st.metric("Unsafe probability", f"{prob:.1%}" if prob is not None else "Not available")
with result_cols[2]:
    st.metric("Risk drivers", len(identify_risk_drivers(scenario.iloc[0])))

st.info(recommend_action(prob, pred))

st.markdown("**Detected risk drivers**")
st.write(", ".join(identify_risk_drivers(scenario.iloc[0])))

st.markdown("**Scenario input**")
st.dataframe(scenario, use_container_width=True)
