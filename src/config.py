from dataclasses import dataclass


@dataclass
class EngineConfig:
    # Default limits used only when sensor-specific limit is missing
    default_temp_min_c: float = 2.0
    default_temp_max_c: float = 8.0
    default_humidity_min_rh: float = 40.0
    default_humidity_max_rh: float = 85.0

    # Data / pattern settings
    rolling_window: int = 3
    max_sample_gap_min: float = 30.0
    slope_threshold_c_per_min: float = 0.01

    # Cooling cycle
    min_cooling_points: int = 2
    min_temp_drop_c: float = 0.25
    max_cooling_cycle_duration_min: float = 240.0
    slow_cooling_warning_min: float = 90.0
    slow_cooling_critical_min: float = 180.0

    # Alert duration rules
    temp_email_duration_min: float = 15.0
    humidity_email_duration_min: float = 20.0

    # SMS rule
    sms_temp_deviation_c: float = 3.0
    sms_temp_duration_min: float = 30.0

    # Data gap
    data_gap_warning_min: float = 60.0
    data_gap_critical_min: float = 180.0

    # Defrost-like pattern
    defrost_temp_rise_c: float = 1.0

    # Alert score
    alert_score_warning: int = 50
    alert_score_critical: int = 75
