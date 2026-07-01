import pandas as pd

from src.config import EngineConfig


def build_features(df: pd.DataFrame, config: EngineConfig) -> pd.DataFrame:
    result = []

    for sensor_id, group in df.groupby("sensor_id"):
        group = group.sort_values("timestamp").copy()

        group["temperature_c"] = pd.to_numeric(group["temperature_c"], errors="coerce")
        group["humidity_rh"] = pd.to_numeric(group["humidity_rh"], errors="coerce")

        group["temp_min_c"] = pd.to_numeric(group["temp_min_c"], errors="coerce").fillna(config.default_temp_min_c)
        group["temp_max_c"] = pd.to_numeric(group["temp_max_c"], errors="coerce").fillna(config.default_temp_max_c)
        group["humidity_min_rh"] = pd.to_numeric(group["humidity_min_rh"], errors="coerce").fillna(config.default_humidity_min_rh)
        group["humidity_max_rh"] = pd.to_numeric(group["humidity_max_rh"], errors="coerce").fillna(config.default_humidity_max_rh)

        group["temp_smooth"] = group["temperature_c"].rolling(
            window=config.rolling_window,
            min_periods=1
        ).mean()

        group["humidity_smooth"] = group["humidity_rh"].rolling(
            window=config.rolling_window,
            min_periods=1
        ).mean()

        group["time_diff_min"] = group["timestamp"].diff().dt.total_seconds() / 60
        group["is_large_gap"] = group["time_diff_min"] > config.max_sample_gap_min

        group["temp_diff"] = group["temp_smooth"].diff()
        group["humidity_diff"] = group["humidity_smooth"].diff()

        group["temp_slope_c_per_min"] = group["temp_diff"] / group["time_diff_min"]
        group["humidity_slope_rh_per_min"] = group["humidity_diff"] / group["time_diff_min"]

        group["is_temp_high"] = group["temperature_c"] > group["temp_max_c"]
        group["is_temp_low"] = group["temperature_c"] < group["temp_min_c"]
        group["is_humidity_high"] = group["humidity_rh"] > group["humidity_max_rh"]
        group["is_humidity_low"] = group["humidity_rh"] < group["humidity_min_rh"]

        group["temp_high_deviation_c"] = group["temperature_c"] - group["temp_max_c"]
        group["temp_low_deviation_c"] = group["temp_min_c"] - group["temperature_c"]
        group["humidity_high_deviation_rh"] = group["humidity_rh"] - group["humidity_max_rh"]
        group["humidity_low_deviation_rh"] = group["humidity_min_rh"] - group["humidity_rh"]

        group["is_temp_rising"] = (
            group["temp_slope_c_per_min"] > config.slope_threshold_c_per_min
        ) & (~group["is_large_gap"].fillna(False))

        group["is_temp_falling"] = (
            group["temp_slope_c_per_min"] < -config.slope_threshold_c_per_min
        ) & (~group["is_large_gap"].fillna(False))

        group["date"] = group["timestamp"].dt.date
        group["hour"] = group["timestamp"].dt.hour

        result.append(group)

    return pd.concat(result, ignore_index=True)
