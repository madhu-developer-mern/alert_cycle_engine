import pandas as pd

from src.config import EngineConfig


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def calculate_alert_score(row: pd.Series, config: EngineConfig) -> int:
    cycle_type = row.get("cycle_type", "")
    duration = _safe_float(row.get("duration_min", 0))
    temp_deviation = _safe_float(row.get("temperature_deviation_c", 0))
    humidity_deviation = _safe_float(row.get("humidity_deviation_rh", 0))
    temp_rise = _safe_float(row.get("temperature_rise_c", 0))

    score = 0

    if cycle_type in ["HIGH_TEMPERATURE_BREACH", "LOW_TEMPERATURE_BREACH"]:
        score += 35
        score += min(int(duration * 1.5), 35)
        score += min(int(temp_deviation * 10), 30)

    elif cycle_type in ["HIGH_HUMIDITY_BREACH", "LOW_HUMIDITY_BREACH"]:
        score += 25
        score += min(int(duration), 30)
        score += min(int(humidity_deviation * 2), 30)

    elif cycle_type == "COOLING_CYCLE":
        duration = _safe_float(row.get("duration_min", 0))

        if duration >= config.slow_cooling_critical_min:
            score = 80
        elif duration >= config.slow_cooling_warning_min:
            score = 60
        else:
            score = 10

    elif cycle_type == "POSSIBLE_DEFROST_CYCLE":
        score = 20 + min(int(temp_rise * 10), 30)

    elif cycle_type == "DATA_GAP":
        if duration >= config.data_gap_critical_min:
            score = 80
        elif duration >= config.data_gap_warning_min:
            score = 60
        else:
            score = 20

    return min(score, 100)


def decide_channels(row: pd.Series, config: EngineConfig, severity: str, send_alert: bool) -> dict:
    cycle_type = row.get("cycle_type", "")
    duration = _safe_float(row.get("duration_min", 0))
    temp_deviation = _safe_float(row.get("temperature_deviation_c", 0))

    dashboard_alert = severity in ["WARNING", "CRITICAL"]
    email_alert = False
    sms_alert = False

    if not send_alert:
        return {
            "dashboard_alert": dashboard_alert,
            "email_alert": False,
            "sms_alert": False,
            "alert_channels": "DASHBOARD" if dashboard_alert else "",
        }

    if cycle_type in ["HIGH_TEMPERATURE_BREACH", "LOW_TEMPERATURE_BREACH"]:
        email_alert = duration >= config.temp_email_duration_min
        sms_alert = (
            temp_deviation >= config.sms_temp_deviation_c
            and duration >= config.sms_temp_duration_min
        )

    elif cycle_type in ["HIGH_HUMIDITY_BREACH", "LOW_HUMIDITY_BREACH"]:
        email_alert = duration >= config.humidity_email_duration_min
        sms_alert = False

    elif cycle_type == "DATA_GAP":
        email_alert = severity in ["WARNING", "CRITICAL"]
        sms_alert = severity == "CRITICAL"

    elif cycle_type == "COOLING_CYCLE":
        email_alert = severity == "CRITICAL"
        sms_alert = False

    elif cycle_type == "POSSIBLE_DEFROST_CYCLE":
        email_alert = severity == "CRITICAL"
        sms_alert = False

    channels = []

    if dashboard_alert:
        channels.append("DASHBOARD")

    if email_alert:
        channels.append("EMAIL")

    if sms_alert:
        channels.append("SMS")

    return {
        "dashboard_alert": dashboard_alert,
        "email_alert": email_alert,
        "sms_alert": sms_alert,
        "alert_channels": ",".join(channels),
    }


def apply_decisions(events_df: pd.DataFrame, config: EngineConfig) -> pd.DataFrame:
    if events_df.empty:
        return events_df

    rows = []

    for _, row in events_df.iterrows():
        score = calculate_alert_score(row, config)

        if score >= config.alert_score_critical:
            severity = "CRITICAL"
            send_alert = True
        elif score >= config.alert_score_warning:
            severity = "WARNING"
            send_alert = True
        else:
            severity = "INFO"
            send_alert = False

        channels = decide_channels(row, config, severity, send_alert)

        final = row.to_dict()
        final["alert_score"] = score
        final["severity"] = severity
        final["send_alert"] = send_alert
        final.update(channels)

        rows.append(final)

    return pd.DataFrame(rows)
