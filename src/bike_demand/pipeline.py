"""Run the complete case study: python -m bike_demand.pipeline."""
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from .data import ROOT, RAW, load_data, audit, make_features
from .hypothesis import test_morning_share
from .model import fit_experiment
from .reporting import save_plots, write_findings, environment_manifest


def run(output=ROOT / "reports", model_path=ROOT / "models" / "demand_model.joblib"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hour, day = load_data()
    quality, missing = audit(hour, day)
    # Fail on confirmed structural errors; unusual but valid rows stay in the data.
    for label in ["hour", "day"]:
        for key in ["missing_cells", "duplicate_keys", "invalid_count_identity", "negative_counts", "invalid_scaled_weather", "invalid_calendar", "invalid_categories"]:
            if quality[label][key]:
                raise ValueError(f"Data audit failed: {label}.{key}={quality[label][key]}")
    if quality["hour"]["invalid_hours"] or quality["daily_reconciliation_mismatches"]:
        raise ValueError("Hourly keys or hourly/daily reconciliation failed")
    missing.to_csv(output / "missing_hours.csv", index=False)
    hypothesis, daily_hypothesis = test_morning_share(hour)
    daily_hypothesis.to_csv(output / "hypothesis_daily_data.csv", index=False)
    print("Audit and hypothesis complete", flush=True)
    bundle, cv, evaluation, test, groups = fit_experiment(hour)
    print("Model selection and held-out evaluation complete", flush=True)
    calibration = hour[(hour.dteday >= "2012-07-01") & (hour.dteday < "2012-09-01")]
    result = permutation_importance(bundle["model"], make_features(calibration), calibration.cnt,
                                    scoring="neg_mean_absolute_error", n_repeats=3, random_state=42)
    importance = pd.DataFrame({"feature": make_features(calibration).columns,
                              "MAE_increase_mean": result.importances_mean,
                              "MAE_increase_std": result.importances_std}).sort_values("MAE_increase_mean", ascending=False)
    for filename, frame in [("cross_validation.csv", cv), ("test_metrics.csv", evaluation), ("test_predictions.csv", test), ("error_segments.csv", groups), ("feature_importance.csv", importance)]:
        frame.to_csv(output / filename, index=False)
    for filename, value in [("data_quality.json", quality), ("hypothesis.json", hypothesis), ("model_metadata.json", bundle["metadata"]), ("reproducibility.json", environment_manifest(RAW))]:
        (output / filename).write_text(json.dumps(value, indent=2), encoding="utf-8")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    save_plots(hour, day, cv, evaluation, test, importance, output / "figures")
    write_findings(quality, hypothesis, bundle, cv, evaluation, test, groups, importance, output)
    print(evaluation.round(3).to_string(index=False), flush=True)
    print(f"Saved reports to {output}", flush=True)
    return bundle, cv, evaluation, test


if __name__ == "__main__":
    run()
