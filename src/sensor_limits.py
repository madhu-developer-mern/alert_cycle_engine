import os

import pandas as pd

from src.config import EngineConfig


def normalize_sensor_id(value) -> str:
    return str(value).strip().upper()


def load_sensor_limit_history(limits_file: str) -> pd.DataFrame:
    if not limits_file or not os.path.exists(limits_file):
        return pd.DataFrame()

    limits = pd.read_csv(limits_file)
    limits["sensor_id"] = limits["sensor_id"].apply(normalize_sensor_id)
    limits["effective_from"] = pd.to_datetime(limits["effective_from"], errors="coerce")

    numeric_cols = [
        "temp_min_c",
        "temp_max_c",
        "humidity_min_rh",
        "humidity_max_rh",
    ]

    for col in numeric_cols:
        limits[col] = pd.to_numeric(limits[col], errors="coerce")

    limits = limits.dropna(subset=["sensor_id", "effective_from"])
    limits = limits.sort_values(["sensor_id", "effective_from"]).reset_index(drop=True)

    return limits


def apply_sensor_limits(sensor_df: pd.DataFrame, limits_file: str, config: EngineConfig) -> pd.DataFrame:
    df = sensor_df.copy()
    df["sensor_id"] = df["sensor_id"].apply(normalize_sensor_id)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    limits = load_sensor_limit_history(limits_file)

    if limits.empty:
        df["group_name"] = "DEFAULT"
        df["temp_min_c"] = config.default_temp_min_c
        df["temp_max_c"] = config.default_temp_max_c
        df["humidity_min_rh"] = config.default_humidity_min_rh
        df["humidity_max_rh"] = config.default_humidity_max_rh
        df["email"] = ""
        df["mobile"] = ""
        df["limit_source"] = "DEFAULT_CONFIG"
        return df

    output_parts = []

    for sensor_id, group in df.groupby("sensor_id"):
        group = group.sort_values("timestamp").copy()
        sensor_limits = limits[limits["sensor_id"] == sensor_id].copy()

        if sensor_limits.empty:
            group["group_name"] = "DEFAULT"
            group["temp_min_c"] = config.default_temp_min_c
            group["temp_max_c"] = config.default_temp_max_c
            group["humidity_min_rh"] = config.default_humidity_min_rh
            group["humidity_max_rh"] = config.default_humidity_max_rh
            group["email"] = ""
            group["mobile"] = ""
            group["limit_source"] = "DEFAULT_CONFIG"
            output_parts.append(group)
            continue

        merged = pd.merge_asof(
            group.sort_values("timestamp"),
            sensor_limits.sort_values("effective_from"),
            left_on="timestamp",
            right_on="effective_from",
            direction="backward",
        )

        if "sensor_id_x" in merged.columns:
            merged["sensor_id"] = merged["sensor_id_x"]
            merged = merged.drop(columns=[c for c in ["sensor_id_x", "sensor_id_y"] if c in merged.columns])

        merged["group_name"] = merged["group_name"].fillna("DEFAULT")
        merged["temp_min_c"] = merged["temp_min_c"].fillna(config.default_temp_min_c)
        merged["temp_max_c"] = merged["temp_max_c"].fillna(config.default_temp_max_c)
        merged["humidity_min_rh"] = merged["humidity_min_rh"].fillna(config.default_humidity_min_rh)
        merged["humidity_max_rh"] = merged["humidity_max_rh"].fillna(config.default_humidity_max_rh)
        merged["email"] = merged["email"].fillna("")
        merged["mobile"] = merged["mobile"].fillna("")
        merged["limit_source"] = "SENSOR_LIMIT_HISTORY"

        output_parts.append(merged)

    result = pd.concat(output_parts, ignore_index=True)
    result = result.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    missing = result[result["limit_source"] == "DEFAULT_CONFIG"]["sensor_id"].unique()

    if len(missing) > 0:
        print("Sensors using default limits:", list(missing))

    return result
