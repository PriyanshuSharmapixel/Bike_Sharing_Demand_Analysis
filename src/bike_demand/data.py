"""Read the original files, audit them, and build only prediction-time features."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CATEGORICAL = ["hr", "weekday", "mnth", "season", "holiday", "workingday", "weathersit"]
NUMERIC = ["temp", "atemp", "hum", "windspeed", "elapsed_days", "hour_sin", "hour_cos", "year_sin", "year_cos"]
FEATURES = CATEGORICAL + NUMERIC


def load_data(raw=RAW):
    raw = Path(raw)
    hour = pd.read_csv(raw / "hour.csv", parse_dates=["dteday"])
    day = pd.read_csv(raw / "day.csv", parse_dates=["dteday"])
    hour["timestamp"] = hour.dteday + pd.to_timedelta(hour.hr, unit="h")
    return hour.sort_values("timestamp").reset_index(drop=True), day.sort_values("dteday").reset_index(drop=True)


def audit(hour, day):
    expected = pd.date_range(day.dteday.min(), day.dteday.max() + pd.Timedelta(hours=23), freq="h")
    missing = expected.difference(pd.DatetimeIndex(hour.timestamp))
    checks = {}
    for label, frame, keys in [("hour", hour, ["dteday", "hr"]), ("day", day, ["dteday"])]:
        checks[label] = {
            "rows": len(frame), "missing_cells": int(frame.isna().sum().sum()),
            "duplicate_keys": int(frame.duplicated(keys).sum()),
            "invalid_count_identity": int((frame.casual + frame.registered != frame.cnt).sum()),
            "negative_counts": int((frame[["cnt", "casual", "registered"]] < 0).any(axis=1).sum()),
            "invalid_scaled_weather": int(((frame[["temp", "atemp", "hum", "windspeed"]] < 0) | (frame[["temp", "atemp", "hum", "windspeed"]] > 1)).any(axis=1).sum()),
            "invalid_calendar": int(((frame.yr != frame.dteday.dt.year - 2011) | (frame.mnth != frame.dteday.dt.month) | (frame.weekday != (frame.dteday.dt.dayofweek + 1) % 7) | (frame.workingday != ((~frame.weekday.isin([0, 6])) & frame.holiday.eq(0)).astype(int))).sum()),
            "invalid_categories": int((~frame.season.isin([1, 2, 3, 4]) | ~frame.holiday.isin([0, 1]) | ~frame.weathersit.isin([1, 2, 3, 4])).sum()),
            "total_rentals": int(frame.cnt.sum()),
        }
    checks["hour"]["invalid_hours"] = int((~hour.hr.between(0, 23)).sum())
    reconciled = day.set_index("dteday")[["cnt", "casual", "registered"]].join(hour.groupby("dteday")[["cnt", "casual", "registered"]].sum(), rsuffix="_hour")
    mismatch = sum(int((reconciled[col] != reconciled[col + "_hour"]).sum()) for col in ["cnt", "casual", "registered"])
    checks.update({
        "date_start": str(day.dteday.min().date()), "date_end": str(day.dteday.max().date()),
        "missing_hour_slots": len(missing), "incomplete_days": int((hour.groupby("dteday").size() < 24).sum()),
        "daily_reconciliation_mismatches": mismatch,
        "zero_humidity_rows": int(hour.hum.eq(0).sum()),
        "weather_category_counts": {str(k): int(v) for k, v in hour.weathersit.value_counts().sort_index().items()},
        "handling": "Keep supplied rows. Do not zero-fill absent hours or delete unusual weather/demand without evidence. Zeros in humidity are flagged, not silently imputed."
    })
    return checks, pd.DataFrame({"timestamp": missing})


def make_features(frame):
    """No cnt, casual, registered, row ID, or future-derived statistics allowed."""
    out = frame.copy()
    out["elapsed_days"] = (out.dteday - pd.Timestamp("2011-01-01")).dt.days
    out["hour_sin"] = np.sin(2 * np.pi * out.hr / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out.hr / 24)
    out["year_sin"] = np.sin(2 * np.pi * (out.dteday.dt.dayofyear - 1) / 365.25)
    out["year_cos"] = np.cos(2 * np.pi * (out.dteday.dt.dayofyear - 1) / 365.25)
    return out[FEATURES]
