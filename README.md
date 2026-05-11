# Weather Risk Forecasting for Energy Operations

Applied machine learning prototype for classifying daily weather conditions as **safe** or **unsafe** for outdoor energy and construction operations exposed to wind, precipitation and temperature risk.

The project is designed as a recruiter-facing portfolio case study for junior / graduate roles in **AI, Data Science, Applied ML and industrial analytics**. It connects a Physics background with practical energy-sector decision support: meteorological data processing, feature engineering, supervised classification, model evaluation and operational recommendations.

---

## Business context

Energy and infrastructure projects often depend on weather-sensitive outdoor activities: field inspections, construction work, lifting operations, access to remote assets, maintenance windows and site mobilisation. Adverse weather can affect safety, cost and schedule.

This repository demonstrates how a lightweight ML pipeline can support planning teams by identifying days with elevated operational weather risk and translating model outputs into a stakeholder-friendly recommendation.

The current version uses a **synthetic but realistic daily weather dataset** so that the complete workflow is reproducible without external APIs. The structure is ready to be extended with public weather feeds or company-specific operational datasets.

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

## What the project demonstrates

- End-to-end Python ML workflow using a modular `src/` structure.
- Synthetic weather data generation with seasonality, noise and injected extreme events.
- Data cleaning, missing-value treatment and time-aware train/test split.
- Feature engineering for operational weather risk.
- Binary classification with Logistic Regression, Random Forest and optional XGBoost.
- Evaluation using accuracy, precision, recall, F1 and confusion matrix.
- Operational decision layer with risk drivers and recommended planner actions.
- Streamlit dashboard for scenario testing.
- Unit tests and GitHub Actions CI.

---

## Repository structure

```text
weather-risk-forecasting-energy-operations/
├── .github/workflows/ci.yml          # Automated tests and training smoke test
├── data/
│   ├── raw/                          # Synthetic raw weather data
│   └── processed/                    # Modeling table and metrics
├── docs/
│   ├── career_texts.md               # CV / LinkedIn positioning copy
│   └── model_card.md                 # Intended use, limitations and next steps
├── models/                           # Saved best model
├── notebooks/
│   └── 01_weather_risk_modeling.ipynb
├── reports/figures/                  # Evaluation and storytelling charts
├── scripts/
│   └── run_training.py
├── src/weather_risk/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── modeling.py
│   ├── operations.py                 # Risk-driver interpretation and recommendations
│   ├── predict.py
│   ├── train.py
│   └── visualization.py
├── streamlit_app/app.py
├── tests/
├── Makefile
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## How to run locally

```bash
git clone <your-repo-url>
cd weather-risk-forecasting-energy-operations
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest
PYTHONPATH=src python -m weather_risk.train
```

Or with `make`:

```bash
make install
make train
make test
```

Launch the Streamlit demo:

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

These strong scores are expected because the current target is generated from explicit operational rules and the model receives the relevant weather variables. The value of the project is not claiming a production-ready forecast model, but demonstrating a complete applied AI workflow: problem framing, data preparation, feature engineering, model comparison, evaluation, explainability and operational storytelling.

---

## Example outputs

Training saves charts under `reports/figures/`:

- `target_distribution.png`
- `monthly_unsafe_rate.png`
- `model_metrics_comparison.png`
- `confusion_matrix_random_forest.png`
- `feature_importance_random_forest.png`

---

## Why this is relevant for AI / Data Science roles

This project shows the ability to:

- Convert a physical / operational problem into a supervised ML task.
- Define a target variable that stakeholders can understand.
- Build reproducible Python pipelines instead of isolated notebook code.
- Compare interpretable and non-linear models.
- Use metrics that reflect operational consequences, especially recall for unsafe conditions.
- Communicate results through charts, documentation and a dashboard prototype.
- Add basic software-engineering practices: tests, CI, modular code and clear project structure.

---

## Limitations and future improvements

- Replace synthetic data with historical station observations or forecast API data.
- Evaluate true day-ahead predictions rather than same-day weather classification.
- Calibrate thresholds by asset type, site location and operation type.
- Add probability calibration and cost-sensitive decision thresholds.
- Add operation-specific profiles, such as lifting operations, inspections or offshore access.
- Integrate automatic reporting and alert generation.
