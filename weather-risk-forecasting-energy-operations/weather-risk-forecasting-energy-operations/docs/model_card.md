# Model Card: Weather Risk Classifier

## Intended use

This model is a portfolio prototype for classifying daily weather conditions as safe or unsafe for outdoor energy operations. It is designed for decision-support storytelling, not autonomous operational approval.

## Inputs

Daily meteorological variables including temperature, wind speed, wind gusts, precipitation, humidity and pressure. Additional engineered features describe seasonality, rain flags, gust factors and thermal stress.

## Output

A binary prediction:

- `0`: safe
- `1`: unsafe

When supported by the estimator, the pipeline also returns an estimated probability of unsafe conditions.

## Target definition

The target is generated using transparent operational thresholds for wind, wind gusts, precipitation and temperature. These thresholds are intentionally simple so that the full ML workflow remains explainable.

## Limitations

- The current dataset is synthetic and should not be interpreted as real site data.
- Strong model scores are expected because the target is rule-generated from weather variables.
- A production version would require observed weather data, forecast horizons, asset-specific thresholds, cost-sensitive validation and domain review.

## Recommended next steps

- Replace synthetic data with station or forecast API data.
- Add probability calibration.
- Evaluate day-ahead and multi-day forecast performance.
- Add operation-specific risk profiles, such as lifting, inspection or offshore access.
