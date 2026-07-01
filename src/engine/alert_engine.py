import pandas as pd


def build_alert_message(row: pd.Series) -> str:
    sensor_id = row.get("sensor_id", "UNKNOWN")
    group_name = row.get("group_name", "")
    cycle_type = row.get("cycle_type", "")
    severity = row.get("severity", "")
    start_time = row.get("start_time", "")
    end_time = row.get("end_time", "")
    duration = row.get("duration_min", "")
    score = row.get("alert_score", "")
    channels = row.get("alert_channels", "")
    recommendation = row.get("recommendation", "")

    return (
        f"[{severity}] {cycle_type} detected for {sensor_id} ({group_name}). "
        f"Start: {start_time}, End: {end_time}, Duration: {duration} min, "
        f"Score: {score}, Channels: {channels}. "
        f"Recommendation: {recommendation}"
    )


def prepare_alerts(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    alerts = events_df[
        (events_df["dashboard_alert"] == True)
        | (events_df["email_alert"] == True)
        | (events_df["sms_alert"] == True)
    ].copy()

    if alerts.empty:
        return alerts

    alerts["alert_message"] = alerts.apply(build_alert_message, axis=1)

    return alerts
