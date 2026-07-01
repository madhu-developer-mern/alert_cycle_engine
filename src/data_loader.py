import glob
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


TIME_KEYWORDS = [
    "gw_utc_time", "timestamp", "datetime", "date_time", "created_at",
    "recorded_at", "received_at", "time", "date", "ts"
]

TEMP_KEYWORDS = [
    "tem", "temp", "temperature", "temperature_c", "temp_c", "celsius"
]

HUMIDITY_KEYWORDS = [
    "hum", "humidity", "humidity_rh", "rh", "relative_humidity"
]

SENSOR_KEYWORDS = [
    "s_id", "sensor_id", "sensorid", "sensor", "device_id", "deviceid",
    "device", "tracker_id", "trackerid", "logger_id"
]


def normalize_column_name(value) -> str:
    value = str(value).strip().lower()
    value = value.replace("%", "percent").replace("°", "")
    value = re.sub(r"[\s\-\/\.\(\)\[\]:]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def make_unique_columns(columns: List[str]) -> List[str]:
    seen = {}
    result = []

    for col in columns:
        col = normalize_column_name(col)

        if not col or col == "nan":
            col = "unnamed"

        if col not in seen:
            seen[col] = 0
            result.append(col)
        else:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")

    return result


def find_column(columns: List[str], keywords: List[str]) -> Optional[str]:
    for col in columns:
        normalized = normalize_column_name(col)

        for key in keywords:
            key = normalize_column_name(key)

            if normalized == key:
                return col

            if key in normalized:
                return col

    return None


def parse_timestamp(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() > 0:
        median_value = numeric.dropna().median()

        # Epoch milliseconds
        if median_value > 10_000_000_000:
            parsed = pd.to_datetime(numeric, unit="ms", errors="coerce")
            if parsed.notna().sum() > 0:
                return parsed

        # Epoch seconds
        if median_value > 1_000_000_000:
            parsed = pd.to_datetime(numeric, unit="s", errors="coerce")
            if parsed.notna().sum() > 0:
                return parsed

        # Excel serial datetime
        if 20_000 < median_value < 80_000:
            parsed = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
            if parsed.notna().sum() > 0:
                return parsed

    return pd.to_datetime(series, errors="coerce")


def find_header_row(raw: pd.DataFrame) -> Optional[int]:
    max_rows = min(30, len(raw))
    best_row = None
    best_score = -1

    for row_index in range(max_rows):
        columns = make_unique_columns(raw.iloc[row_index].tolist())

        time_col = find_column(columns, TIME_KEYWORDS)
        temp_col = find_column(columns, TEMP_KEYWORDS)
        hum_col = find_column(columns, HUMIDITY_KEYWORDS)
        sensor_col = find_column(columns, SENSOR_KEYWORDS)

        score = 0
        if time_col:
            score += 4
        if temp_col:
            score += 4
        if hum_col:
            score += 2
        if sensor_col:
            score += 2

        if score > best_score:
            best_score = score
            best_row = row_index

        if time_col and temp_col:
            return row_index

    if best_score >= 6:
        return best_row

    return None


def read_sheet_auto_header(file_path: str, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    if raw.empty:
        return pd.DataFrame()

    header_row = find_header_row(raw)

    if header_row is None:
        return pd.DataFrame()

    columns = make_unique_columns(raw.iloc[header_row].tolist())

    df = raw.iloc[header_row + 1:].copy()
    df.columns = columns
    df = df.dropna(how="all")

    return df


def load_sensor_data(data_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(data_dir, "*.xlsx")) + glob.glob(os.path.join(data_dir, "*.xls"))

    if not files:
        raise FileNotFoundError(f"No Excel files found in {data_dir}")

    frames = []

    for file_path in files:
        workbook: Dict[str, pd.DataFrame] = pd.read_excel(file_path, sheet_name=None, header=None)

        for sheet_name in workbook.keys():
            df = read_sheet_auto_header(file_path, sheet_name)

            if df.empty:
                print(f"Skipping {os.path.basename(file_path)} / {sheet_name}: header not detected")
                continue

            time_col = find_column(list(df.columns), TIME_KEYWORDS)
            temp_col = find_column(list(df.columns), TEMP_KEYWORDS)
            hum_col = find_column(list(df.columns), HUMIDITY_KEYWORDS)
            sensor_col = find_column(list(df.columns), SENSOR_KEYWORDS)

            print("\nDetected columns")
            print("File:", os.path.basename(file_path))
            print("Sheet:", sheet_name)
            print("Time:", time_col)
            print("Temperature:", temp_col)
            print("Humidity:", hum_col)
            print("Sensor:", sensor_col)

            if time_col is None or temp_col is None:
                print("Skipping because timestamp or temperature column missing")
                print("Available columns:", list(df.columns))
                continue

            clean = pd.DataFrame()
            clean["timestamp"] = parse_timestamp(df[time_col])

            # Raw sensor values are stored as scaled integers.
            # Example: tem=2910 means 29.10°C, hum=5400 means 54.00%RH.
            clean["temperature_raw"] = pd.to_numeric(df[temp_col], errors="coerce")
            clean["temperature_c"] = clean["temperature_raw"] / 100.0

            if hum_col:
                clean["humidity_raw"] = pd.to_numeric(df[hum_col], errors="coerce")
                clean["humidity_rh"] = clean["humidity_raw"] / 100.0
            else:
                clean["humidity_raw"] = np.nan
                clean["humidity_rh"] = np.nan

            if sensor_col:
                clean["sensor_id"] = df[sensor_col].astype(str).str.strip().str.upper()
            else:
                clean["sensor_id"] = os.path.basename(file_path).split("_")[0].upper()

            clean["source_file"] = os.path.basename(file_path)
            clean["sheet_name"] = sheet_name

            before = len(clean)
            clean = clean.dropna(subset=["timestamp", "temperature_c"])
            after = len(clean)

            print(f"Valid rows: {after}/{before}")

            if not clean.empty:
                frames.append(clean)

    if not frames:
        raise ValueError("No valid sensor data found. Check Excel columns.")

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    return result
