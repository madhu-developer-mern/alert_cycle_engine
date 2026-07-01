import argparse

from src.config import EngineConfig
from src.data_loader import load_sensor_data
from src.sensor_limits import apply_sensor_limits
from src.features import build_features
from src.patterns.cycle_detector import detect_all_events
from src.engine.decision_engine import apply_decisions
from src.engine.recommendation_engine import add_recommendations
from src.engine.alert_engine import prepare_alerts
from src.utils.report_writer import save_reports


def main():
    parser = argparse.ArgumentParser(
        description="Thinxsense Dynamic Threshold Alert + Cycle + Recommendation Engine"
    )

    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--limits-file", default="config/sensor_limit_history.csv")

    args = parser.parse_args()

    config = EngineConfig()

    print("Loading raw sensor data...")
    raw_df = load_sensor_data(args.data_dir)
    print(f"Raw records loaded: {len(raw_df)}")

    print("\nApplying sensor-wise threshold limits...")
    enriched_df = apply_sensor_limits(raw_df, args.limits_file, config)
    print(f"Records after threshold mapping: {len(enriched_df)}")

    print("\nBuilding features...")
    processed_df = build_features(enriched_df, config)
    print(f"Processed records: {len(processed_df)}")

    print("\nDetecting breaches, cycles, and data gaps...")
    events_df = detect_all_events(processed_df, config)
    print(f"Events detected: {len(events_df)}")

    print("\nApplying decision engine...")
    events_df = apply_decisions(events_df, config)

    print("\nAdding recommendations...")
    events_df = add_recommendations(events_df)

    print("\nPreparing alert list...")
    alerts_df = prepare_alerts(events_df)
    print(f"Alerts to show/send: {len(alerts_df)}")

    print("\nSaving output reports...")
    save_reports(
        enriched_df=enriched_df,
        processed_df=processed_df,
        events_df=events_df,
        alerts_df=alerts_df,
        output_dir=args.output_dir,
    )

    print("\nEngine completed.")
    print(f"Raw records: {len(raw_df)}")
    print(f"Processed records: {len(processed_df)}")
    print(f"Events detected: {len(events_df)}")
    print(f"Alerts to show/send: {len(alerts_df)}")

    if not alerts_df.empty:
        print("\nFirst 10 alerts:")
        columns = [
            "sensor_id",
            "group_name",
            "cycle_type",
            "severity",
            "alert_score",
            "alert_channels",
            "start_time",
            "end_time",
            "duration_min",
        ]
        existing = [col for col in columns if col in alerts_df.columns]
        print(alerts_df[existing].head(10).to_string(index=False))
    else:
        print("\nNo alerts generated.")


if __name__ == "__main__":
    main()
