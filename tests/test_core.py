import numpy as np
import pandas as pd
import pytest
from bike_demand.data import load_data, audit, make_features
from bike_demand.operations import capacity_scenario, answer_question, forecast_scenario
from bike_demand.model import CalendarBaseline, new_model, predict, FOLDS
from bike_demand.hypothesis import test_morning_share as morning_share_analysis


@pytest.fixture(scope="module")
def data():
    return load_data()


def test_supplied_data_integrity(data):
    h, d = data
    result, missing = audit(h, d)
    assert result["hour"]["rows"] == 17379
    assert result["day"]["rows"] == 731
    assert result["missing_hour_slots"] == len(missing) == 165
    assert result["daily_reconciliation_mismatches"] == 0
    assert result["hour"]["total_rentals"] == result["day"]["total_rentals"] == 3292679


def test_audit_detects_broken_data(data):
    h, d = data
    h = h.copy()
    h.loc[0, "cnt"] += 1
    result, _ = audit(h, d)
    assert result["hour"]["invalid_count_identity"] == 1
    assert result["daily_reconciliation_mismatches"] == 1


def test_target_leakage_invariance(data):
    h, _ = data
    expected = make_features(h.head(50))
    changed = h.head(50).copy()
    changed[["cnt", "casual", "registered", "instant"]] = 999999
    pd.testing.assert_frame_equal(expected, make_features(changed))
    assert set(expected).isdisjoint({"cnt", "casual", "registered", "instant", "timestamp"})


def test_temporal_boundaries(data):
    h, _ = data
    for start, end in FOLDS:
        train = h[h.dteday < start]
        validation = h[(h.dteday >= start) & (h.dteday < end)]
        assert train.timestamp.max() < validation.timestamp.min()
        assert validation.dteday.max() < pd.Timestamp("2012-07-01")


def test_baseline_predicts_unseen_holiday_group():
    X = pd.DataFrame({"hr": [8, 8], "weekday": [1, 1], "workingday": [1, 1], "elapsed_days": [1, 8]})
    baseline = CalendarBaseline().fit(X, [100, 200])
    X.loc[0, "workingday"] = 0
    assert baseline.predict(X)[0] == 150


def test_capacity_units_and_invalid_assumptions():
    p = pd.DataFrame({"prediction": [120.], "upper_90": [180.]})
    assert capacity_scenario(p, 30, 0, 1).tolist() == [60]
    assert capacity_scenario(p, 30, 0, .5).tolist() == [120]
    assert capacity_scenario(p, 30, 0, 1, True).tolist() == [90]
    for duration, turnaround, usable in [(0, 0, 1), (30, -1, 1), (30, 0, 0), (30, 0, 1.1)]:
        with pytest.raises(ValueError):
            capacity_scenario(p, duration, turnaround, usable)


@pytest.fixture(scope="module")
def scenario(data):
    h, _ = data
    train = h[h.dteday < "2012-07-01"]
    model = new_model("Calendar baseline").fit(make_features(train), train.cnt)
    bundle = {"model": model, "radius": 50}
    return bundle, forecast_scenario(bundle, "2012-10-15")


def test_scenario_and_bounds(scenario):
    bundle, forecast = scenario
    assert len(forecast) == 24
    assert forecast.season.eq(4).all()
    assert (forecast.lower_90 >= 0).all()
    assert (forecast.upper_90 >= forecast.prediction).all()
    with pytest.raises(ValueError, match="historical demo"):
        forecast_scenario(bundle, "2026-11-04")


def test_assistant_computes_and_refuses(scenario, data):
    _, f = scenario
    h, _ = data
    assert f"{f.prediction.sum():,.0f}" in answer_question("daily total", f, h)
    assert "08:00" in answer_question("rentals at 8 am", f, h)
    assert "20:00" in answer_question("rentals at 8 pm", f, h)
    assert "station inventory" in answer_question("Where should I buy bikes?", f, h)
    assert "not a live forecast" in answer_question("tomorrow", f, h)
    assert "Change the date" in answer_question("peak on 2012-11-01", f, h)
    assert "Supported questions" in answer_question("run arbitrary code", f, h)
    assert "1-12" in answer_question("rentals at 24 pm", f, h)


def test_hypothesis_excludes_incomplete_days_and_holdout(data):
    h, _ = data
    result, daily = morning_share_analysis(h)
    assert daily.dteday.max() < pd.Timestamp("2012-07-01")
    assert daily.share.between(0, 1).all()
    assert result["adjusted_difference_percentage_points"] > 0
    assert 0 <= result["one_sided_p_value"] <= 1
