from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42
TARGET_COLUMN = "is_unsafe"

OPERATIONAL_THRESHOLDS = {
    "wind_speed_max_kmh": 45,
    "wind_gust_max_kmh": 65,
    "precipitation_mm": 8,
    "temperature_min_c": -3,
    "temperature_max_c": 38,
}
