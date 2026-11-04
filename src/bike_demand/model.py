"""Time-ordered rental forecasting with a baseline and held-out calibration."""
import math
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .data import make_features, CATEGORICAL, NUMERIC


class CalendarBaseline:
    """Mean rental count for each hour/weekday/workday, using last 8 train weeks."""
    keys = ["hr", "weekday", "workingday"]

    def fit(self, X, y):
        frame = X.copy()
        frame["target"] = np.asarray(y)
        frame = frame[frame.elapsed_days >= frame.elapsed_days.max() - 55]
        self.means_ = frame.groupby(self.keys).target.mean()
        self.hour_ = frame.groupby("hr").target.mean()
        self.global_ = float(frame.target.mean())
        return self

    def predict(self, X):
        keys = pd.MultiIndex.from_frame(X[self.keys])
        estimates = self.means_.reindex(keys).to_numpy()
        fallback = X.hr.map(self.hour_).fillna(self.global_).to_numpy()
        return np.where(np.isnan(estimates), fallback, estimates)


def new_model(name):
    if name == "Calendar baseline":
        return CalendarBaseline()
    if name == "Ridge":
        pre = ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("numeric", StandardScaler(), NUMERIC),
        ])
        return Pipeline([("preprocess", pre), ("regressor", Ridge(alpha=10))])
    if name == "Histogram boosting":
        return HistGradientBoostingRegressor(loss="poisson", max_iter=250, max_leaf_nodes=31,
            learning_rate=0.08, l2_regularization=5, min_samples_leaf=25,
            categorical_features=CATEGORICAL, early_stopping=False, random_state=42)
    raise ValueError(f"Unknown model {name}")


def predict(model, frame):
    return np.maximum(0, model.predict(make_features(frame)))


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    return {"MAE": float(mean_absolute_error(y, prediction)),
            "RMSE": float(np.sqrt(mean_squared_error(y, prediction))),
            "WAPE_percent": float(100 * np.abs(y - prediction).sum() / y.sum()),
            "R2": float(r2_score(y, prediction)),
            "bias_rentals": float((prediction - y).mean())}


FOLDS = [("2012-01-01", "2012-03-01"), ("2012-03-01", "2012-05-01"), ("2012-05-01", "2012-07-01")]
MODELS = ["Calendar baseline", "Ridge", "Histogram boosting"]


def fit_experiment(hour):
    rows = []
    for start, end in FOLDS:
        train = hour[hour.dteday < start]
        valid = hour[(hour.dteday >= start) & (hour.dteday < end)]
        for name in MODELS:
            model = new_model(name).fit(make_features(train), train.cnt)
            rows.append({"model": name, "validation_start": start, "validation_end_exclusive": end,
                         "train_rows": len(train), "validation_rows": len(valid), **metrics(valid.cnt, predict(model, valid))})
    cv = pd.DataFrame(rows)
    winner = cv.groupby("model").MAE.mean().idxmin()
    train = hour[hour.dteday < "2012-07-01"]
    calibration = hour[(hour.dteday >= "2012-07-01") & (hour.dteday < "2012-09-01")]
    test = hour[hour.dteday >= "2012-09-01"].copy()
    fitted = {name: new_model(name).fit(make_features(train), train.cnt) for name in MODELS}
    model = fitted[winner]
    residual = np.abs(calibration.cnt.to_numpy() - predict(model, calibration))
    # Finite-sample split-conformal rank. Serial dependence/drift means 90% is nominal,
    # not a coverage guarantee; held-out coverage and subgroup coverage are reported.
    rank = min(len(residual), math.ceil((len(residual) + 1) * .90))
    q = float(np.sort(residual)[rank - 1])
    for name, fitted_model in fitted.items():
        test[name] = predict(fitted_model, test)
    test["prediction"] = test[winner]
    test["lower_90"] = np.maximum(0, test.prediction - q)
    test["upper_90"] = test.prediction + q
    test["absolute_error"] = np.abs(test.cnt - test.prediction)
    test["covered"] = (test.cnt >= test.lower_90) & (test.cnt <= test.upper_90)
    evaluation = pd.DataFrame([{"model": name, **metrics(test.cnt, test[name])} for name in MODELS])
    # Peak definition is fixed in advance and does not depend on held-out target values.
    test["commute_peak"] = test.workingday.eq(1) & test.hr.isin([7, 8, 9, 16, 17, 18])
    groups = []
    for col in ["hr", "workingday", "weathersit", "commute_peak"]:
        for value, group in test.groupby(col):
            groups.append({"group": col, "value": str(value), "rows": len(group),
                           **metrics(group.cnt, group.prediction), "interval_coverage_percent": float(group.covered.mean() * 100)})
    daily = test.groupby("dteday")[["cnt", "prediction"]].sum()
    meta = {
        "selected_model": winner,
        "selection": "Lowest unweighted mean MAE across three expanding-window validation folds. Fixed candidate settings; test used once for final reporting.",
        "training_end": "2012-06-30", "calibration_start": "2012-07-01", "calibration_end": "2012-08-31",
        "test_start": "2012-09-01", "test_end": "2012-12-31",
        "train_rows": len(train), "calibration_rows": len(calibration), "test_rows": len(test),
        "nominal_interval_coverage": .90, "calibration_absolute_error_quantile": q,
        "test_interval_coverage_percent": float(test.covered.mean() * 100),
        "test_average_interval_width": float((test.upper_90 - test.lower_90).mean()),
        "test_daily_aggregation_metrics": metrics(daily.cnt, daily.prediction),
        "forecast_contract": "Retrospective hourly rental estimates conditional on observed weather. Operational use needs hourly weather forecasts; forecast-weather degradation has not been measured. No lags, no target components, no station allocation. Not a 2026 production forecast.",
        "interval_limitations": "Absolute-residual split calibration on July-August. Nominal 90% marginal intervals; temporal dependence and seasonal drift violate exchangeability. Hourly intervals are not a joint daily interval."
    }
    bundle = {"model": model, "radius": q, "metadata": meta}
    return bundle, cv, evaluation, test, pd.DataFrame(groups)
