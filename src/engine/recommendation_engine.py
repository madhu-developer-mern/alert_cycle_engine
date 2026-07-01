import pandas as pd


def build_recommendation(row: pd.Series) -> str:
    cycle_type = row.get("cycle_type", "")
    severity = row.get("severity", "INFO")

    if cycle_type == "HIGH_TEMPERATURE_BREACH":
        return (
            "Temperature crossed maximum threshold. Check cooling performance, door opening, "
            "product loading, airflow blockage, and threshold configuration."
        )

    if cycle_type == "LOW_TEMPERATURE_BREACH":
        return (
            "Temperature dropped below minimum threshold. Check thermostat/setpoint, freezing risk, "
            "sensor placement, and cooling control."
        )

    if cycle_type == "HIGH_HUMIDITY_BREACH":
        return (
            "Humidity crossed maximum threshold. Check condensation, door opening, wet product loading, "
            "drainage, and air circulation."
        )

    if cycle_type == "LOW_HUMIDITY_BREACH":
        return (
            "Humidity dropped below minimum threshold. Check dry-zone risk, excessive dehumidification, "
            "airflow direction, and product moisture-loss risk."
        )

    if cycle_type == "COOLING_CYCLE":
        if severity in ["WARNING", "CRITICAL"]:
            return (
                "Cooling cycle is slower than expected. Check evaporator coil, airflow blockage, frost build-up, "
                "door-open duration, and product loading impact."
            )
        return "Cooling cycle looks informational. Continue monitoring cooling duration and cooling rate."

    if cycle_type == "POSSIBLE_DEFROST_CYCLE":
        return (
            "Possible defrost-like pattern detected. Verify defrost schedule, temperature rise, "
            "humidity change, and recovery duration."
        )

    if cycle_type == "DATA_GAP":
        return (
            "Sensor data gap detected. Check gateway connectivity, sensor battery, RSSI/signal strength, "
            "upload interval, and missing packets."
        )

    return "Monitor this event and compare with historical behavior."


def add_recommendations(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return events_df

    events_df = events_df.copy()
    events_df["recommendation"] = events_df.apply(build_recommendation, axis=1)

    return events_df
