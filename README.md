# Weather Risk Forecasting for Energy Operations

Machine learning prototype for classifying daily weather conditions as **safe** or **unsafe** for outdoor energy operations exposed to wind, rainfall and temperature risk.

This project is designed as a recruiter-facing portfolio case study for junior / graduate roles in **AI, Data Science, Applied AI and industrial analytics**. It connects a Physics background with practical energy-sector decision support: cleaning meteorological data, engineering operational risk features, training supervised classification models and translating predictions into an interpretable operational output.

---

## Business context

Energy infrastructure projects often depend on weather-sensitive outdoor operations: field inspections, construction work, lifting operations, access to remote assets, maintenance windows and site mobilisation. Even simple weather thresholds can affect safety, cost and schedule.

This repository demonstrates how a lightweight ML system could support planning teams by flagging days with elevated operational weather risk.

The first version uses a **synthetic but realistic daily weather dataset** so that the full pipeline is reproducible without external APIs. The code is structured so the synthetic source can be replaced by public weather feeds or company-specific operational datasets.

---

## Objective

Predict whether a day is operationally:

- **Safe**: weather conditions are within predefined operating limits.
- **Unsafe**: at least one key weather variable exceeds an operational threshold.

The target is defined using transparent industrial-style rules:

| Risk driver | Unsafe if |
|---|---:|
| Maximum wind speed | >= 45 km/h |
| Maximum wind gust | >= 65 km/h |
| Daily precipitation | >= 8 mm |
| Minimum temperature | <= -3 °C |
| Maximum temperature | >= 38 °C |

In a real deployment, these thresholds would be calibrated with HSE requirements, equipment limits, site-specific procedures and historical downtime records.

---

## Methods

The project includes:

- Synthetic weather data generation with seasonality and extreme-event injection.
- Data cleaning and missing-value handling.
- Feature engineering for operational risk.
- Binary classification models:
  - Logistic Regression
  - Random Forest
  - XGBoost, when installed
- Time-aware train/test split to avoid random temporal leakage.
- Model evaluation using accuracy, precision, recall, F1 and confusion matrix.
- Visualisation of class balance, monthly unsafe-day rate, model metrics and feature importance.
- Optional Streamlit prototype for scenario testing.

---

## Repository structure

```text
weather-risk-forecasting-energy-operations/
├── data/
│   ├── raw/                         # Synthetic raw weather data
│   └── processed/                   # Modeling table and metrics
├── docs/
│   └── career_texts.md              # CV, LinkedIn and positioning copy
├── models/                          # Saved best model
├── notebooks/
│   └── 01_weather_risk_modeling.ipynb
├── reports/
│   └── figures/                     # Evaluation and storytelling charts
├── src/
│   └── weather_risk/
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── modeling.py
│       ├── predict.py
│       ├── train.py
│       └── visualization.py
├── streamlit_app/
│   └── app.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## How to run

```bash
git clone <your-repo-url>
cd weather-risk-forecasting-energy-operations
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
PYTHONPATH=src python -m weather_risk.train
```

Optional Streamlit app:

```bash
streamlit run streamlit_app/app.py
```

---

## Current baseline results

On the initial synthetic dataset, the best model is typically Random Forest or XGBoost. Example metrics from one reproducible run:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.921 | 0.739 | 0.932 | 0.824 |
| Random Forest | 0.995 | 1.000 | 0.973 | 0.986 |
| XGBoost | 0.995 | 1.000 | 0.973 | 0.986 |

These strong scores are expected because the initial target is generated from explicit operational rules and the model receives the relevant weather variables. The value of the project is not claiming a production-ready forecast model, but demonstrating the full applied AI workflow: problem framing, data preparation, feature engineering, model comparison, evaluation and operational storytelling.

---

## Example outputs

After training, charts are saved under `reports/figures/`:

- `target_distribution.png`
- `monthly_unsafe_rate.png`
- `model_metrics_comparison.png`
- `confusion_matrix_random_forest.png`
- `feature_importance_random_forest.png`

---

## Why this project is relevant for AI / Data Science roles

This is a compact but realistic example of applied machine learning in an industrial setting. It shows the ability to:

- Convert a physical / operational problem into a supervised ML task.
- Define a target variable that stakeholders can understand.
- Build reproducible Python pipelines instead of isolated notebook code.
- Compare interpretable and non-linear models.
- Use metrics that reflect operational consequences, especially recall for unsafe conditions.
- Communicate results through charts, README storytelling and deployment-oriented structure.

---

## Future improvements

- Replace synthetic data with historical observations from a public weather API or station dataset.
- Add forecast data and evaluate true day-ahead predictions.
- Calibrate thresholds by asset type, site location and operation type.
- Add probability calibration and cost-sensitive decision thresholds.
- Integrate weather alerts and automatic reporting.
- Add unit tests and CI workflow.

