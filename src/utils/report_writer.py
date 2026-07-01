import os

import pandas as pd


def save_reports(
    enriched_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    events_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    enriched_df.to_csv(os.path.join(output_dir, "01_enriched_sensor_data_with_limits.csv"), index=False)
    processed_df.to_csv(os.path.join(output_dir, "02_processed_sensor_features.csv"), index=False)
    events_df.to_csv(os.path.join(output_dir, "03_detected_events.csv"), index=False)
    alerts_df.to_csv(os.path.join(output_dir, "04_alerts_to_send.csv"), index=False)

    if not events_df.empty:
        event_summary = (
            events_df.groupby(["sensor_id", "group_name", "cycle_type", "severity"], dropna=False)
            .agg(
                event_count=("cycle_type", "count"),
                dashboard_alerts=("dashboard_alert", "sum"),
                email_alerts=("email_alert", "sum"),
                sms_alerts=("sms_alert", "sum"),
                avg_alert_score=("alert_score", "mean"),
            )
            .reset_index()
        )

        event_summary["avg_alert_score"] = event_summary["avg_alert_score"].round(2)
    else:
        event_summary = pd.DataFrame()

    event_summary.to_csv(os.path.join(output_dir, "05_event_summary.csv"), index=False)

    if not processed_df.empty:
        sensor_summary = (
            processed_df.groupby(["sensor_id", "group_name"], dropna=False)
            .agg(
                total_records=("sensor_id", "count"),
                first_timestamp=("timestamp", "min"),
                last_timestamp=("timestamp", "max"),
                avg_temperature_c=("temperature_c", "mean"),
                min_temperature_c=("temperature_c", "min"),
                max_temperature_c=("temperature_c", "max"),
                avg_humidity_rh=("humidity_rh", "mean"),
                min_humidity_rh=("humidity_rh", "min"),
                max_humidity_rh=("humidity_rh", "max"),
                temp_high_points=("is_temp_high", "sum"),
                temp_low_points=("is_temp_low", "sum"),
                humidity_high_points=("is_humidity_high", "sum"),
                humidity_low_points=("is_humidity_low", "sum"),
                data_gap_points=("is_large_gap", "sum"),
            )
            .reset_index()
        )

        for col in [
            "avg_temperature_c",
            "min_temperature_c",
            "max_temperature_c",
            "avg_humidity_rh",
            "min_humidity_rh",
            "max_humidity_rh",
        ]:
            sensor_summary[col] = sensor_summary[col].round(2)
    else:
        sensor_summary = pd.DataFrame()

    sensor_summary.to_csv(os.path.join(output_dir, "06_sensor_health_summary.csv"), index=False)
