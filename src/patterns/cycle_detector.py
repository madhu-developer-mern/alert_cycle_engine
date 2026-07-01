import pandas as pd

from src.config import EngineConfig


def _segments_from_boolean(group: pd.DataFrame, flag_col: str):
    indices = group.index[group[flag_col] == True].tolist()

    if not indices:
        return []

    segments = []
    current = [indices[0]]

    for idx in indices[1:]:
        large_gap = False

        if "is_large_gap" in group.columns:
            value = group.loc[idx, "is_large_gap"]
            large_gap = bool(value) if pd.notna(value) else False

        if idx == current[-1] + 1 and not large_gap:
            current.append(idx)
        else:
            segments.append(current)
            current = [idx]

    segments.append(current)
    return segments


def _common_fields(group: pd.DataFrame, idx: int) -> dict:
    return {
        "sensor_id": group.loc[idx].get("sensor_id", ""),
        "group_name": group.loc[idx].get("group_name", ""),
        "email": group.loc[idx].get("email", ""),
        "mobile": group.loc[idx].get("mobile", ""),
        "temp_min_c": group.loc[idx].get("temp_min_c", None),
        "temp_max_c": group.loc[idx].get("temp_max_c", None),
        "humidity_min_rh": group.loc[idx].get("humidity_min_rh", None),
        "humidity_max_rh": group.loc[idx].get("humidity_max_rh", None),
        "limit_source": group.loc[idx].get("limit_source", ""),
    }


def detect_temperature_breaches(df: pd.DataFrame, config: EngineConfig) -> pd.DataFrame:
    events = []

    rules = [
        ("is_temp_high", "HIGH_TEMPERATURE_BREACH", "temp_high_deviation_c", "temp_max_c", "max"),
        ("is_temp_low", "LOW_TEMPERATURE_BREACH", "temp_low_deviation_c", "temp_min_c", "min"),
    ]

    for sensor_id, group in df.groupby("sensor_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)

        for flag_col, cycle_type, deviation_col, limit_col, value_mode in rules:
            segments = _segments_from_boolean(group, flag_col)

            for segment in segments:
                start_idx = segment[0]
                end_idx = segment[-1]
                rows = group.loc[segment]

                start_time = group.loc[start_idx, "timestamp"]
                end_time = group.loc[end_idx, "timestamp"]
                duration_min = max((end_time - start_time).total_seconds() / 60, 0)

                worst_temp = rows["temperature_c"].max() if value_mode == "max" else rows["temperature_c"].min()
                deviation = max(rows[deviation_col].max(), 0)
                limit_value = rows[limit_col].iloc[0]

                event = _common_fields(group, start_idx)
                event.update({
                    "cycle_type": cycle_type,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_min": round(float(duration_min), 2),
                    "worst_temperature_c": round(float(worst_temp), 2),
                    "temperature_limit_c": round(float(limit_value), 2),
                    "temperature_deviation_c": round(float(deviation), 2),
                })

                events.append(event)

    return pd.DataFrame(events)


def detect_humidity_breaches(df: pd.DataFrame, config: EngineConfig) -> pd.DataFrame:
    events = []

    rules = [
        ("is_humidity_high", "HIGH_HUMIDITY_BREACH", "humidity_high_deviation_rh", "humidity_max_rh", "max"),
        ("is_humidity_low", "LOW_HUMIDITY_BREACH", "humidity_low_deviation_rh", "humidity_min_rh", "min"),
    ]

    for sensor_id, group in df.groupby("sensor_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)

        for flag_col, cycle_type, deviation_col, limit_col, value_mode in rules:
            segments = _segments_from_boolean(group, flag_col)

            for segment in segments:
                start_idx = segment[0]
                end_idx = segment[-1]
                rows = group.loc[segment]

                start_time = group.loc[start_idx, "timestamp"]
                end_time = group.loc[end_idx, "timestamp"]
                duration_min = max((end_time - start_time).total_seconds() / 60, 0)

                worst_humidity = rows["humidity_rh"].max() if value_mode == "max" else rows["humidity_rh"].min()
                deviation = max(rows[deviation_col].max(), 0)
                limit_value = rows[limit_col].iloc[0]

                event = _common_fields(group, start_idx)
                event.update({
                    "cycle_type": cycle_type,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_min": round(float(duration_min), 2),
                    "worst_humidity_rh": round(float(worst_humidity), 2),
                    "humidity_limit_rh": round(float(limit_value), 2),
                    "humidity_deviation_rh": round(float(deviation), 2),
                })

                events.append(event)

    return pd.DataFrame(events)


def detect_cooling_cycles(df: pd.DataFrame, config: EngineConfig) -> pd.DataFrame:
    events = []

    for sensor_id, group in df.groupby("sensor_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        segments = _segments_from_boolean(group, "is_temp_falling")

        for segment in segments:
            if len(segment) < config.min_cooling_points:
                continue

            start_idx = max(segment[0] - 1, 0)
            peak_window_start = max(start_idx - 3, 0)
            peak_window = group.loc[peak_window_start:start_idx]

            peak_idx = peak_window["temp_smooth"].idxmax()
            low_idx = group.loc[segment]["temp_smooth"].idxmin()

            peak_temp = group.loc[peak_idx, "temp_smooth"]
            low_temp = group.loc[low_idx, "temp_smooth"]
            temp_drop = peak_temp - low_temp

            if temp_drop < config.min_temp_drop_c:
                continue

            start_time = group.loc[peak_idx, "timestamp"]
            end_time = group.loc[low_idx, "timestamp"]
            duration_min = (end_time - start_time).total_seconds() / 60

            if duration_min <= 0:
                continue

            if duration_min > config.max_cooling_cycle_duration_min:
                continue

            humidity_start = group.loc[peak_idx, "humidity_smooth"]
            humidity_end = group.loc[low_idx, "humidity_smooth"]

            event = _common_fields(group, peak_idx)
            event.update({
                "cycle_type": "COOLING_CYCLE",
                "start_time": start_time,
                "end_time": end_time,
                "duration_min": round(float(duration_min), 2),
                "peak_temperature_c": round(float(peak_temp), 2),
                "lowest_temperature_c": round(float(low_temp), 2),
                "temperature_drop_c": round(float(temp_drop), 2),
                "cooling_rate_c_per_min": round(float(temp_drop / duration_min), 4),
                "humidity_start_rh": round(float(humidity_start), 2) if pd.notna(humidity_start) else None,
                "humidity_end_rh": round(float(humidity_end), 2) if pd.notna(humidity_end) else None,
                "humidity_change_rh": round(float(humidity_end - humidity_start), 2) if pd.notna(humidity_start) and pd.notna(humidity_end) else None,
            })

            events.append(event)

    return pd.DataFrame(events)


def detect_data_gap_events(df: pd.DataFrame, config: EngineConfig) -> pd.DataFrame:
    events = []

    for sensor_id, group in df.groupby("sensor_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        gap_rows = group[group["is_large_gap"] == True]

        for idx, row in gap_rows.iterrows():
            if idx == 0:
                continue

            event = _common_fields(group, idx)
            event.update({
                "cycle_type": "DATA_GAP",
                "start_time": group.loc[idx - 1, "timestamp"],
                "end_time": row["timestamp"],
                "duration_min": round(float(row["time_diff_min"]), 2),
                "reason": "Sensor data gap detected.",
            })

            events.append(event)

    return pd.DataFrame(events)


def detect_possible_defrost_cycles(df: pd.DataFrame, config: EngineConfig) -> pd.DataFrame:
    events = []

    for sensor_id, group in df.groupby("sensor_id"):
        group = group.sort_values("timestamp").reset_index(drop=True).copy()

        group["temp_rise_window"] = group["temp_smooth"] - group["temp_smooth"].shift(3)
        group["future_temp_drop"] = group["temp_smooth"] - group["temp_smooth"].shift(-3)

        candidates = group[
            (group["temp_rise_window"] >= config.defrost_temp_rise_c)
            & (group["future_temp_drop"] >= 0.2)
            & (~group["is_large_gap"].fillna(False))
        ]

        for idx in candidates.index:
            start_idx = max(idx - 3, 0)
            end_idx = min(idx + 6, len(group) - 1)

            start_time = group.loc[start_idx, "timestamp"]
            end_time = group.loc[end_idx, "timestamp"]
            duration_min = max((end_time - start_time).total_seconds() / 60, 0)

            if duration_min > config.max_cooling_cycle_duration_min:
                continue

            temp_before = group.loc[start_idx, "temp_smooth"]
            temp_peak = group.loc[start_idx:end_idx, "temp_smooth"].max()
            temp_rise = temp_peak - temp_before

            event = _common_fields(group, start_idx)
            event.update({
                "cycle_type": "POSSIBLE_DEFROST_CYCLE",
                "start_time": start_time,
                "end_time": end_time,
                "duration_min": round(float(duration_min), 2),
                "temperature_rise_c": round(float(temp_rise), 2),
                "reason": "Temperature rose quickly and later recovered.",
            })

            events.append(event)

    result = pd.DataFrame(events)

    if result.empty:
        return result

    return result.drop_duplicates(subset=["sensor_id", "start_time", "cycle_type"])


def detect_all_events(df: pd.DataFrame, config: EngineConfig) -> pd.DataFrame:
    all_events = []

    for detector in [
        detect_temperature_breaches,
        detect_humidity_breaches,
        detect_cooling_cycles,
        detect_possible_defrost_cycles,
        detect_data_gap_events,
    ]:
        detected = detector(df, config)

        if not detected.empty:
            all_events.append(detected)

    if not all_events:
        return pd.DataFrame()

    events = pd.concat(all_events, ignore_index=True, sort=False)
    events = events.sort_values(["sensor_id", "start_time", "cycle_type"]).reset_index(drop=True)

    return events
