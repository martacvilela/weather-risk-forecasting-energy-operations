from __future__ import annotations

import pandas as pd

from .config import DATA_PROCESSED_DIR, DATA_RAW_DIR, FIGURES_DIR
from .data import save_raw_dataset
from .features import get_feature_columns, prepare_modeling_table
from .modeling import save_metrics, save_model, select_best_model, time_aware_train_test_split, train_and_evaluate
from .visualization import (
    save_confusion_matrix,
    save_feature_importance,
    save_metrics_comparison,
    save_target_distribution_plot,
    save_weather_risk_timeseries,
)


def main() -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_RAW_DIR / "synthetic_weather_operations.csv"
    if not raw_path.exists():
        raw_df = save_raw_dataset()
    else:
        raw_df = pd.read_csv(raw_path)

    df = prepare_modeling_table(raw_df)
    df.to_csv(DATA_PROCESSED_DIR / "modeling_table.csv", index=False)

    fitted_models, metrics = train_and_evaluate(df)
    best_name = select_best_model(metrics, criterion="f1")
    save_model(fitted_models[best_name], "best_weather_risk_model")
    save_metrics(metrics, DATA_PROCESSED_DIR / "metrics.json")

    X_train, X_test, y_train, y_test = time_aware_train_test_split(df)
    save_target_distribution_plot(df)
    save_weather_risk_timeseries(df)
    save_metrics_comparison(metrics)
    save_confusion_matrix(fitted_models[best_name], X_test, y_test, best_name)
    save_feature_importance(fitted_models[best_name], get_feature_columns(), best_name)

    print(f"Best model: {best_name}")
    print(metrics)


if __name__ == "__main__":
    main()
