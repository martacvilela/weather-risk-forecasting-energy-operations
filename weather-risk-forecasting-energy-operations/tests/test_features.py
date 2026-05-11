from weather_risk.data import generate_synthetic_weather_data
from weather_risk.features import prepare_modeling_table, get_feature_columns


def test_prepare_modeling_table_creates_target_and_features():
    raw_df = generate_synthetic_weather_data(n_days=100)
    df = prepare_modeling_table(raw_df)
    assert "is_unsafe" in df.columns
    assert "risk_label" in df.columns
    assert set(get_feature_columns()).issubset(df.columns)
    assert df[get_feature_columns()].isna().sum().sum() == 0
