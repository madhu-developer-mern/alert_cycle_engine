from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st


LIMITS_FILE = Path("config/sensor_limit_history.csv")
PROCESSED_HISTORY_FILE = Path("outputs/02_processed_sensor_features.csv")
HEALTH_FILE = Path("outputs/06_sensor_health_summary.csv")


DEFAULT_LIMIT = {
    "sensor_id": "UNKNOWN",
    "group_name": "CUSTOM",
    "temp_min_c": 2.0,
    "temp_max_c": 8.0,
    "humidity_min_rh": 40.0,
    "humidity_max_rh": 85.0,
    "email": "",
    "mobile": "",
}


def load_limits() -> pd.DataFrame:
    if not LIMITS_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(LIMITS_FILE)
    df["sensor_id"] = df["sensor_id"].astype(str).str.strip().str.upper()
    df["effective_from"] = pd.to_datetime(df["effective_from"], errors="coerce")

    for col in ["temp_min_c", "temp_max_c", "humidity_min_rh", "humidity_max_rh"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["sensor_id"])


@st.cache_data
def load_history() -> pd.DataFrame:
    if not PROCESSED_HISTORY_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(PROCESSED_HISTORY_FILE)
    df["sensor_id"] = df["sensor_id"].astype(str).str.strip().str.upper()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    df["humidity_rh"] = pd.to_numeric(df["humidity_rh"], errors="coerce")

    return df.dropna(subset=["sensor_id", "timestamp", "temperature_c"])


def get_active_limit(sensor_id: str, timestamp: pd.Timestamp, limits_df: pd.DataFrame) -> dict:
    sensor_id = str(sensor_id).strip().upper()

    if limits_df.empty:
        limit = DEFAULT_LIMIT.copy()
        limit["sensor_id"] = sensor_id
        return limit

    sensor_limits = limits_df[
        (limits_df["sensor_id"] == sensor_id)
        & (limits_df["effective_from"] <= timestamp)
    ].sort_values("effective_from")

    if sensor_limits.empty:
        limit = DEFAULT_LIMIT.copy()
        limit["sensor_id"] = sensor_id
        return limit

    return sensor_limits.iloc[-1].to_dict()


def get_sensor_baseline(sensor_id: str, history_df: pd.DataFrame) -> dict:
    sensor_id = str(sensor_id).strip().upper()

    if history_df.empty:
        return {}

    df = history_df[history_df["sensor_id"] == sensor_id].copy()

    if df.empty:
        return {}

    baseline = {
        "records": len(df),
        "first_timestamp": df["timestamp"].min(),
        "last_timestamp": df["timestamp"].max(),

        "temp_mean": df["temperature_c"].mean(),
        "temp_std": df["temperature_c"].std(),
        "temp_min": df["temperature_c"].min(),
        "temp_max": df["temperature_c"].max(),
        "temp_p05": df["temperature_c"].quantile(0.05),
        "temp_p95": df["temperature_c"].quantile(0.95),

        "hum_mean": df["humidity_rh"].mean(),
        "hum_std": df["humidity_rh"].std(),
        "hum_min": df["humidity_rh"].min(),
        "hum_max": df["humidity_rh"].max(),
        "hum_p05": df["humidity_rh"].quantile(0.05),
        "hum_p95": df["humidity_rh"].quantile(0.95),
    }

    return baseline


def severity_from_score(score: int) -> str:
    if score >= 75:
        return "CRITICAL"
    if score >= 50:
        return "WARNING"
    return "INFO"


def score_threshold_event(event_type: str, duration_min: float, deviation: float) -> int:
    if event_type in ["HIGH_TEMPERATURE_BREACH", "LOW_TEMPERATURE_BREACH"]:
        score = 35
        score += min(int(duration_min * 1.5), 35)
        score += min(int(deviation * 10), 30)
        return min(score, 100)

    if event_type in ["HIGH_HUMIDITY_BREACH", "LOW_HUMIDITY_BREACH"]:
        score = 25
        score += min(int(duration_min), 30)
        score += min(int(deviation * 2), 30)
        return min(score, 100)

    return 0


def get_channels(event_type: str, severity: str, duration_min: float, deviation: float) -> str:
    channels = []

    if severity in ["WARNING", "CRITICAL"]:
        channels.append("DASHBOARD")

    if event_type in ["HIGH_TEMPERATURE_BREACH", "LOW_TEMPERATURE_BREACH"]:
        if duration_min >= 15:
            channels.append("EMAIL")

        if deviation >= 3 and duration_min >= 30:
            channels.append("SMS")

    if event_type in ["HIGH_HUMIDITY_BREACH", "LOW_HUMIDITY_BREACH"]:
        if duration_min >= 20:
            channels.append("EMAIL")

    return ",".join(channels)


def get_recommendation(event_type: str, abnormal_reason: str = "") -> str:
    if event_type == "NORMAL":
        return "Reading is within threshold and historical behavior looks normal."

    if event_type == "HIGH_TEMPERATURE_BREACH":
        return "Temperature crossed maximum limit. Check cooling performance, door opening, product loading, airflow blockage, and threshold setting."

    if event_type == "LOW_TEMPERATURE_BREACH":
        return "Temperature dropped below minimum limit. Check thermostat/setpoint, freezing risk, sensor placement, and cooling control."

    if event_type == "HIGH_HUMIDITY_BREACH":
        return "Humidity crossed maximum limit. Check condensation, wet product loading, door opening, drainage, and air circulation."

    if event_type == "LOW_HUMIDITY_BREACH":
        return "Humidity dropped below minimum limit. Check dry-zone risk, excessive dehumidification, airflow direction, and product moisture-loss risk."

    if event_type == "HISTORICAL_ABNORMAL_PATTERN":
        return f"Current value is unusual compared with old data. {abnormal_reason} Check sensor placement, environment change, product load, or calibration."

    if event_type == "POSSIBLE_COOLING_ACTIVITY":
        return "Temperature is falling compared with previous reading. Possible cooling activity detected. Monitor recovery time and cooling rate."

    if event_type == "POSSIBLE_DEFROST_OR_LOADING":
        return "Temperature increased quickly compared with previous reading. Possible defrost, loading, door opening, or disturbance pattern."

    return "Monitor and compare with historical behavior."


def check_thresholds(
    sensor_id: str,
    group_name: str,
    temperature_c: float,
    humidity_rh: float,
    duration_min: float,
    temp_min_c: float,
    temp_max_c: float,
    humidity_min_rh: float,
    humidity_max_rh: float,
) -> pd.DataFrame:
    events = []

    if temperature_c > temp_max_c:
        deviation = temperature_c - temp_max_c
        event_type = "HIGH_TEMPERATURE_BREACH"
        score = score_threshold_event(event_type, duration_min, deviation)
        severity = severity_from_score(score)
        channels = get_channels(event_type, severity, duration_min, deviation)

        events.append({
            "decision": "ALERT",
            "sensor_id": sensor_id,
            "group_name": group_name,
            "event_type": event_type,
            "value": round(temperature_c, 2),
            "limit": temp_max_c,
            "deviation": round(deviation, 2),
            "duration_min": duration_min,
            "severity": severity,
            "alert_score": score,
            "channels": channels,
            "send_alert": bool(channels),
            "recommendation": get_recommendation(event_type),
        })

    elif temperature_c < temp_min_c:
        deviation = temp_min_c - temperature_c
        event_type = "LOW_TEMPERATURE_BREACH"
        score = score_threshold_event(event_type, duration_min, deviation)
        severity = severity_from_score(score)
        channels = get_channels(event_type, severity, duration_min, deviation)

        events.append({
            "decision": "ALERT",
            "sensor_id": sensor_id,
            "group_name": group_name,
            "event_type": event_type,
            "value": round(temperature_c, 2),
            "limit": temp_min_c,
            "deviation": round(deviation, 2),
            "duration_min": duration_min,
            "severity": severity,
            "alert_score": score,
            "channels": channels,
            "send_alert": bool(channels),
            "recommendation": get_recommendation(event_type),
        })

    if humidity_rh > humidity_max_rh:
        deviation = humidity_rh - humidity_max_rh
        event_type = "HIGH_HUMIDITY_BREACH"
        score = score_threshold_event(event_type, duration_min, deviation)
        severity = severity_from_score(score)
        channels = get_channels(event_type, severity, duration_min, deviation)

        events.append({
            "decision": "ALERT",
            "sensor_id": sensor_id,
            "group_name": group_name,
            "event_type": event_type,
            "value": round(humidity_rh, 2),
            "limit": humidity_max_rh,
            "deviation": round(deviation, 2),
            "duration_min": duration_min,
            "severity": severity,
            "alert_score": score,
            "channels": channels,
            "send_alert": bool(channels),
            "recommendation": get_recommendation(event_type),
        })

    elif humidity_rh < humidity_min_rh:
        deviation = humidity_min_rh - humidity_rh
        event_type = "LOW_HUMIDITY_BREACH"
        score = score_threshold_event(event_type, duration_min, deviation)
        severity = severity_from_score(score)
        channels = get_channels(event_type, severity, duration_min, deviation)

        events.append({
            "decision": "ALERT",
            "sensor_id": sensor_id,
            "group_name": group_name,
            "event_type": event_type,
            "value": round(humidity_rh, 2),
            "limit": humidity_min_rh,
            "deviation": round(deviation, 2),
            "duration_min": duration_min,
            "severity": severity,
            "alert_score": score,
            "channels": channels,
            "send_alert": bool(channels),
            "recommendation": get_recommendation(event_type),
        })

    return pd.DataFrame(events)


def check_historical_abnormality(
    sensor_id: str,
    group_name: str,
    temperature_c: float,
    humidity_rh: float,
    baseline: dict,
) -> pd.DataFrame:
    if not baseline:
        return pd.DataFrame([{
            "decision": "NO_HISTORY",
            "sensor_id": sensor_id,
            "group_name": group_name,
            "event_type": "NO_HISTORICAL_DATA",
            "severity": "INFO",
            "alert_score": 0,
            "send_alert": False,
            "recommendation": "No historical data found for this sensor. Only threshold check can be used.",
        }])

    rows = []

    temp_std = baseline.get("temp_std", 0)
    hum_std = baseline.get("hum_std", 0)

    temp_z = 0
    hum_z = 0

    if pd.notna(temp_std) and temp_std > 0:
        temp_z = abs((temperature_c - baseline["temp_mean"]) / temp_std)

    if pd.notna(hum_std) and hum_std > 0:
        hum_z = abs((humidity_rh - baseline["hum_mean"]) / hum_std)

    temp_outside_p95 = temperature_c > baseline["temp_p95"] or temperature_c < baseline["temp_p05"]
    hum_outside_p95 = humidity_rh > baseline["hum_p95"] or humidity_rh < baseline["hum_p05"]

    reasons = []

    if temp_z >= 3:
        reasons.append(f"Temperature z-score is {temp_z:.2f}, which is very unusual.")

    if hum_z >= 3:
        reasons.append(f"Humidity z-score is {hum_z:.2f}, which is very unusual.")

    if temp_outside_p95:
        reasons.append(
            f"Temperature is outside historical 5–95% range "
            f"({baseline['temp_p05']:.2f} to {baseline['temp_p95']:.2f}°C)."
        )

    if hum_outside_p95:
        reasons.append(
            f"Humidity is outside historical 5–95% range "
            f"({baseline['hum_p05']:.2f} to {baseline['hum_p95']:.2f}%RH)."
        )

    if reasons:
        score = 50

        if temp_z >= 3 or hum_z >= 3:
            score = 75

        severity = severity_from_score(score)
        reason_text = " ".join(reasons)

        rows.append({
            "decision": "ABNORMAL_PATTERN",
            "sensor_id": sensor_id,
            "group_name": group_name,
            "event_type": "HISTORICAL_ABNORMAL_PATTERN",
            "temperature_c": round(temperature_c, 2),
            "humidity_rh": round(humidity_rh, 2),
            "temp_z_score": round(temp_z, 2),
            "humidity_z_score": round(hum_z, 2),
            "severity": severity,
            "alert_score": score,
            "send_alert": severity == "CRITICAL",
            "channels": "DASHBOARD" if severity == "WARNING" else "DASHBOARD,EMAIL",
            "reason": reason_text,
            "recommendation": get_recommendation("HISTORICAL_ABNORMAL_PATTERN", reason_text),
        })

    else:
        rows.append({
            "decision": "NORMAL_PATTERN",
            "sensor_id": sensor_id,
            "group_name": group_name,
            "event_type": "HISTORICAL_NORMAL_PATTERN",
            "temperature_c": round(temperature_c, 2),
            "humidity_rh": round(humidity_rh, 2),
            "temp_z_score": round(temp_z, 2),
            "humidity_z_score": round(hum_z, 2),
            "severity": "INFO",
            "alert_score": 0,
            "send_alert": False,
            "channels": "",
            "reason": "Current value is close to historical behavior.",
            "recommendation": "Pattern looks normal compared with old data.",
        })

    return pd.DataFrame(rows)


def check_previous_reading_pattern(
    sensor_id: str,
    group_name: str,
    current_time: pd.Timestamp,
    temperature_c: float,
    humidity_rh: float,
    previous_time: pd.Timestamp,
    previous_temperature_c: float,
    previous_humidity_rh: float,
) -> pd.DataFrame:
    gap_min = max((current_time - previous_time).total_seconds() / 60, 0.01)

    temp_change = temperature_c - previous_temperature_c
    humidity_change = humidity_rh - previous_humidity_rh
    temp_slope = temp_change / gap_min

    rows = []

    if temp_change <= -0.25:
        event_type = "POSSIBLE_COOLING_ACTIVITY"
        rows.append({
            "decision": "PATTERN",
            "sensor_id": sensor_id,
            "group_name": group_name,
            "event_type": event_type,
            "previous_temperature_c": round(previous_temperature_c, 2),
            "current_temperature_c": round(temperature_c, 2),
            "temperature_change_c": round(temp_change, 2),
            "humidity_change_rh": round(humidity_change, 2),
            "time_gap_min": round(gap_min, 2),
            "slope_c_per_min": round(temp_slope, 4),
            "severity": "INFO",
            "alert_score": 10,
            "send_alert": False,
            "recommendation": get_recommendation(event_type),
        })

    elif temp_change >= 1.0:
        event_type = "POSSIBLE_DEFROST_OR_LOADING"
        rows.append({
            "decision": "PATTERN",
            "sensor_id": sensor_id,
            "group_name": group_name,
            "event_type": event_type,
            "previous_temperature_c": round(previous_temperature_c, 2),
            "current_temperature_c": round(temperature_c, 2),
            "temperature_change_c": round(temp_change, 2),
            "humidity_change_rh": round(humidity_change, 2),
            "time_gap_min": round(gap_min, 2),
            "slope_c_per_min": round(temp_slope, 4),
            "severity": "WARNING",
            "alert_score": 50,
            "send_alert": False,
            "recommendation": get_recommendation(event_type),
        })

    return pd.DataFrame(rows)


def show_single_reading_simulator():
    limits_df = load_limits()
    history_df = load_history()

    st.subheader("Single Reading Simulator with Threshold + Historical Pattern Check")

    if not limits_df.empty:
        sensor_options = sorted(limits_df["sensor_id"].unique().tolist())
    elif not history_df.empty:
        sensor_options = sorted(history_df["sensor_id"].unique().tolist())
    else:
        sensor_options = ["H4200116", "H4200117", "H4200118", "H4200136"]

    col1, col2 = st.columns(2)

    with col1:
        sensor_id = st.selectbox("Sensor ID", sensor_options)
        reading_date = st.date_input("Reading Date")
        reading_time = st.time_input("Reading Time")
        timestamp = pd.Timestamp(datetime.combine(reading_date, reading_time))

    active_limit = get_active_limit(sensor_id, timestamp, limits_df)
    baseline = get_sensor_baseline(sensor_id, history_df)

    st.markdown("### Threshold Settings")

    threshold_mode = st.radio(
        "Threshold Source",
        ["Use sensor configured threshold", "Use custom threshold"],
        horizontal=True,
    )

    if threshold_mode == "Use sensor configured threshold":
        temp_min_default = float(active_limit["temp_min_c"])
        temp_max_default = float(active_limit["temp_max_c"])
        hum_min_default = float(active_limit["humidity_min_rh"])
        hum_max_default = float(active_limit["humidity_max_rh"])
        group_name = str(active_limit.get("group_name", "DEFAULT"))
    else:
        temp_min_default = float(active_limit["temp_min_c"])
        temp_max_default = float(active_limit["temp_max_c"])
        hum_min_default = float(active_limit["humidity_min_rh"])
        hum_max_default = float(active_limit["humidity_max_rh"])
        group_name = "CUSTOM_LIMIT_TEST"

    t1, t2, t3, t4 = st.columns(4)

    temp_min_c = t1.number_input("Temp Min °C", value=temp_min_default, step=0.5)
    temp_max_c = t2.number_input("Temp Max °C", value=temp_max_default, step=0.5)
    humidity_min_rh = t3.number_input("Humidity Min %RH", value=hum_min_default, step=1.0)
    humidity_max_rh = t4.number_input("Humidity Max %RH", value=hum_max_default, step=1.0)

    st.markdown("### Input Reading")

    with col2:
        raw_mode = st.checkbox(
            "I am entering raw sensor values like tem=2910 / hum=5400",
            value=False,
        )

        temperature_input = st.number_input("Temperature", value=29.10, step=0.10)
        humidity_input = st.number_input("Humidity", value=56.00, step=0.10)

        duration_min = st.number_input(
            "Assumed breach duration in minutes",
            min_value=0.0,
            value=15.0,
            step=5.0,
        )

    if raw_mode:
        temperature_c = temperature_input / 100.0
        humidity_rh = humidity_input / 100.0
    else:
        temperature_c = temperature_input
        humidity_rh = humidity_input

    st.markdown("### Optional Previous Reading for Cycle / Pattern Detection")

    use_previous = st.checkbox("Add previous reading")

    previous_pattern_df = pd.DataFrame()

    if use_previous:
        p1, p2 = st.columns(2)

        with p1:
            previous_date = st.date_input("Previous Reading Date")
            previous_time = st.time_input("Previous Reading Time")
            previous_timestamp = pd.Timestamp(datetime.combine(previous_date, previous_time))

        with p2:
            previous_temperature_c = st.number_input("Previous Temperature °C", value=31.00, step=0.10)
            previous_humidity_rh = st.number_input("Previous Humidity %RH", value=55.00, step=0.10)

    if st.button("Analyze Reading", type="primary"):
        threshold_events_df = check_thresholds(
            sensor_id=sensor_id,
            group_name=group_name,
            temperature_c=temperature_c,
            humidity_rh=humidity_rh,
            duration_min=duration_min,
            temp_min_c=temp_min_c,
            temp_max_c=temp_max_c,
            humidity_min_rh=humidity_min_rh,
            humidity_max_rh=humidity_max_rh,
        )

        historical_df = check_historical_abnormality(
            sensor_id=sensor_id,
            group_name=group_name,
            temperature_c=temperature_c,
            humidity_rh=humidity_rh,
            baseline=baseline,
        )

        if use_previous:
            previous_pattern_df = check_previous_reading_pattern(
                sensor_id=sensor_id,
                group_name=group_name,
                current_time=timestamp,
                temperature_c=temperature_c,
                humidity_rh=humidity_rh,
                previous_time=previous_timestamp,
                previous_temperature_c=previous_temperature_c,
                previous_humidity_rh=previous_humidity_rh,
            )

        st.markdown("## Final Decision")

        has_threshold_alert = not threshold_events_df.empty
        has_historical_abnormal = (
            not historical_df.empty
            and "decision" in historical_df.columns
            and historical_df.iloc[0]["decision"] == "ABNORMAL_PATTERN"
        )

        if has_threshold_alert:
            max_severity = threshold_events_df["severity"].iloc[0]
            if "CRITICAL" in threshold_events_df["severity"].values:
                st.error("ALERT: Reading crossed configured threshold.")
            else:
                st.warning("WARNING: Reading crossed configured threshold.")
        elif has_historical_abnormal:
            if historical_df.iloc[0]["severity"] == "CRITICAL":
                st.error("ABNORMAL PATTERN: Reading is unusual compared with old data.")
            else:
                st.warning("ABNORMAL PATTERN: Reading is outside normal historical range.")
        else:
            st.success("NORMAL: Reading is within threshold and historical pattern looks normal.")

        st.markdown("### Active Threshold Used")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Sensor", sensor_id)
        m2.metric("Group", group_name)
        m3.metric("Temp Range", f"{temp_min_c:.2f} to {temp_max_c:.2f} °C")
        m4.metric("Hum Range", f"{humidity_min_rh:.0f} to {humidity_max_rh:.0f} %RH")
        m5.metric("Input", f"{temperature_c:.2f}°C / {humidity_rh:.2f}%RH")

        st.markdown("### Threshold Decision")

        if threshold_events_df.empty:
            normal_row = pd.DataFrame([{
                "decision": "NORMAL",
                "sensor_id": sensor_id,
                "group_name": group_name,
                "event_type": "NORMAL",
                "temperature_c": round(temperature_c, 2),
                "humidity_rh": round(humidity_rh, 2),
                "severity": "INFO",
                "alert_score": 0,
                "send_alert": False,
                "channels": "",
                "recommendation": get_recommendation("NORMAL"),
            }])
            st.dataframe(normal_row, use_container_width=True)
        else:
            st.dataframe(threshold_events_df, use_container_width=True)

        st.markdown("### Historical Pattern Decision")

        if baseline:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Historical Records", f"{baseline['records']}")
            b2.metric("Avg Temp", f"{baseline['temp_mean']:.2f} °C")
            b3.metric("Temp 5–95%", f"{baseline['temp_p05']:.2f} to {baseline['temp_p95']:.2f}")
            b4.metric("Hum 5–95%", f"{baseline['hum_p05']:.2f} to {baseline['hum_p95']:.2f}")

        st.dataframe(historical_df, use_container_width=True)

        if not previous_pattern_df.empty:
            st.markdown("### Previous Reading Pattern")
            st.dataframe(previous_pattern_df, use_container_width=True)


def show_bulk_paste_simulator():
    limits_df = load_limits()
    history_df = load_history()

    st.subheader("Bulk Paste Simulator")

    st.write("Paste CSV data. You can use your own threshold columns also.")

    sample = """sensor_id,timestamp,temperature_c,humidity_rh,duration_min,temp_min_c,temp_max_c,humidity_min_rh,humidity_max_rh
H4200116,2026-07-01 10:00:00,29.5,56,15,20,43,10,90
H4200116,2026-07-01 10:15:00,44.5,57,30,20,43,10,90
H4200136,2026-07-01 10:00:00,-12.2,75,30,-11,40,1,90
"""

    text = st.text_area("Paste readings", value=sample, height=220)

    if st.button("Analyze Bulk Readings"):
        try:
            df = pd.read_csv(StringIO(text))

            required = {"sensor_id", "timestamp", "temperature_c", "humidity_rh", "duration_min"}
            if not required.issubset(set(df.columns)):
                st.error(f"Missing columns. Required minimum columns: {sorted(required)}")
                return

            all_rows = []

            for _, row in df.iterrows():
                sensor_id = str(row["sensor_id"]).strip().upper()
                timestamp = pd.to_datetime(row["timestamp"])
                limit = get_active_limit(sensor_id, timestamp, limits_df)

                temp_min = float(row.get("temp_min_c", limit["temp_min_c"]))
                temp_max = float(row.get("temp_max_c", limit["temp_max_c"]))
                hum_min = float(row.get("humidity_min_rh", limit["humidity_min_rh"]))
                hum_max = float(row.get("humidity_max_rh", limit["humidity_max_rh"]))

                threshold_df = check_thresholds(
                    sensor_id=sensor_id,
                    group_name=str(limit.get("group_name", "DEFAULT")),
                    temperature_c=float(row["temperature_c"]),
                    humidity_rh=float(row["humidity_rh"]),
                    duration_min=float(row["duration_min"]),
                    temp_min_c=temp_min,
                    temp_max_c=temp_max,
                    humidity_min_rh=hum_min,
                    humidity_max_rh=hum_max,
                )

                baseline = get_sensor_baseline(sensor_id, history_df)
                hist_df = check_historical_abnormality(
                    sensor_id=sensor_id,
                    group_name=str(limit.get("group_name", "DEFAULT")),
                    temperature_c=float(row["temperature_c"]),
                    humidity_rh=float(row["humidity_rh"]),
                    baseline=baseline,
                )

                if threshold_df.empty:
                    threshold_df = pd.DataFrame([{
                        "decision": "NORMAL",
                        "sensor_id": sensor_id,
                        "group_name": str(limit.get("group_name", "DEFAULT")),
                        "event_type": "NORMAL",
                        "temperature_c": float(row["temperature_c"]),
                        "humidity_rh": float(row["humidity_rh"]),
                        "severity": "INFO",
                        "alert_score": 0,
                        "send_alert": False,
                        "channels": "",
                        "recommendation": get_recommendation("NORMAL"),
                    }])

                all_rows.append(threshold_df)
                all_rows.append(hist_df)

            result = pd.concat(all_rows, ignore_index=True, sort=False)
            st.dataframe(result, use_container_width=True)

        except Exception as exc:
            st.error(f"Failed to analyze bulk readings: {exc}")


def show_batch_output_dashboard():
    st.subheader("Batch Output Dashboard")

    summary_file = Path("outputs/05_event_summary.csv")
    alerts_file = Path("outputs/04_alerts_to_send.csv")
    health_file = Path("outputs/06_sensor_health_summary.csv")

    if not summary_file.exists():
        st.warning("No batch output found. Run the engine first.")
        st.code(
            "python3 main.py --data-dir data --output-dir outputs --limits-file config/sensor_limit_history.csv",
            language="bash",
        )
        return

    health_df = pd.read_csv(health_file) if health_file.exists() else pd.DataFrame()
    summary_df = pd.read_csv(summary_file)
    alerts_df = pd.read_csv(alerts_file) if alerts_file.exists() else pd.DataFrame()

    st.markdown("### Sensor Health")
    st.dataframe(health_df, use_container_width=True)

    st.markdown("### Event Summary")
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("### Alerts")
    st.dataframe(alerts_df.head(300), use_container_width=True)


def main():
    st.set_page_config(
        page_title="Thinxsense Alert Simulator",
        page_icon="🌡️",
        layout="wide",
    )

    st.title("Thinxsense Alert + Cycle Simulator")

    page = st.sidebar.radio(
        "Choose Page",
        [
            "Single Reading Simulator",
            "Bulk Paste Simulator",
            "Batch Output Dashboard",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.write("Limit config:")
    st.sidebar.code(str(LIMITS_FILE))

    st.sidebar.write("Historical data:")
    st.sidebar.code(str(PROCESSED_HISTORY_FILE))

    if page == "Single Reading Simulator":
        show_single_reading_simulator()
    elif page == "Bulk Paste Simulator":
        show_bulk_paste_simulator()
    else:
        show_batch_output_dashboard()


if __name__ == "__main__":
    main()
