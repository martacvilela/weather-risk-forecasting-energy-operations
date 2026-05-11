from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_DIR, RANDOM_STATE, TARGET_COLUMN
from .features import get_feature_columns


def build_models() -> dict:
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    except Exception:
        pass

    return models


def time_aware_train_test_split(df: pd.DataFrame, test_size: float = 0.25):
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    features = get_feature_columns()
    return train_df[features], test_df[features], train_df[TARGET_COLUMN], test_df[TARGET_COLUMN]


def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def train_and_evaluate(df: pd.DataFrame) -> tuple[dict, dict]:
    X_train, X_test, y_train, y_test = time_aware_train_test_split(df)
    models = build_models()
    fitted_models = {}
    metrics = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        metrics[name] = evaluate_classifier(model, X_test, y_test)

    return fitted_models, metrics


def select_best_model(metrics: dict, criterion: str = "f1") -> str:
    return max(metrics, key=lambda name: metrics[name][criterion])


def save_model(model, name: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def save_metrics(metrics: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
