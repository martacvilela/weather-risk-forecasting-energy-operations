from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from .config import FIGURES_DIR, TARGET_COLUMN


def save_target_distribution_plot(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    counts = df["risk_label"].value_counts().reindex(["safe", "unsafe"])
    ax = counts.plot(kind="bar", figsize=(7, 4))
    ax.set_title("Operational Weather Risk Distribution")
    ax.set_xlabel("Risk class")
    ax.set_ylabel("Number of days")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=160)
    plt.close()


def save_weather_risk_timeseries(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    monthly = df.set_index("date").resample("ME")[TARGET_COLUMN].mean()
    ax = monthly.plot(figsize=(10, 4))
    ax.set_title("Monthly Share of Unsafe Days")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unsafe day rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "monthly_unsafe_rate.png", dpi=160)
    plt.close()


def save_metrics_comparison(metrics: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, values in metrics.items():
        for metric_name in ["accuracy", "precision", "recall", "f1"]:
            rows.append({"model": model_name, "metric": metric_name, "value": values[metric_name]})
    metrics_df = pd.DataFrame(rows)
    pivot = metrics_df.pivot(index="model", columns="metric", values="value")
    ax = pivot[["accuracy", "precision", "recall", "f1"]].plot(kind="bar", figsize=(9, 5))
    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_metrics_comparison.png", dpi=160)
    plt.close()


def save_confusion_matrix(model, X_test, y_test, model_name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["safe", "unsafe"],
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{model_name}.png", dpi=160)
    plt.close()


def save_feature_importance(model, feature_columns: list[str], model_name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if not hasattr(estimator, "feature_importances_"):
        return
    importance = pd.Series(estimator.feature_importances_, index=feature_columns).sort_values().tail(12)
    ax = importance.plot(kind="barh", figsize=(8, 5))
    ax.set_title(f"Top Feature Importances - {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"feature_importance_{model_name}.png", dpi=160)
    plt.close()
