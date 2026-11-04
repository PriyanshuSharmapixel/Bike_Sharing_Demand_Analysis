"""Predefined morning-share hypothesis, adjusted for calendar month and year."""
import numpy as np
import pandas as pd
from scipy.stats import t


def test_morning_share(hour):
    """OLS month fixed effects with week-clustered standard errors.

    Development data only; complete days only. One day is one observation,
    rather than treating 24 correlated hourly rows as independent samples.
    This is observational association, not a causal effect of work schedules.
    """
    h = hour[hour.dteday < "2012-07-01"].copy()
    full_dates = h.groupby("dteday").size().loc[lambda x: x == 24].index
    h = h[h.dteday.isin(full_dates)]
    daily = h.groupby("dteday").agg(total=("cnt", "sum"), workingday=("workingday", "first"))
    daily["morning"] = h[h.hr.between(7, 9)].groupby("dteday").cnt.sum()
    daily["share"] = daily.morning / daily.total
    monthly = pd.get_dummies(daily.index.to_period("M").astype(str), drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(daily)), daily.workingday.to_numpy(), monthly.to_numpy()])
    y = daily.share.to_numpy()
    inv = np.linalg.pinv(X.T @ X)
    beta = inv @ X.T @ y
    residual = y - X @ beta
    clusters = daily.index.to_period("W-SUN")
    meat = np.zeros((X.shape[1], X.shape[1]))
    for week in clusters.unique():
        idx = np.asarray(clusters == week)
        score = X[idx].T @ residual[idx]
        meat += np.outer(score, score)
    n, k = X.shape
    g = clusters.nunique()
    covariance = inv @ meat @ inv * g / (g - 1) * (n - 1) / (n - k)
    se = float(np.sqrt(covariance[1, 1]))
    critical = t.ppf(0.975, g - 1)
    p = float(t.sf(beta[1] / se, g - 1))
    return {
        "hypothesis": "Working days have a greater 07:00-09:59 share of daily rentals than non-working days, adjusting for calendar month-year.",
        "null": "The adjusted working-day coefficient is <= 0.",
        "method": "Daily OLS with month-year fixed effects; ISO-style Monday-Sunday week-clustered covariance; one-sided t-test, alpha=0.05; two-sided 95% confidence interval.",
        "period": "2011-01-01 through 2012-06-30, complete 24-hour days only",
        "days": n, "week_clusters": g,
        "workingday_mean_share": float(daily.loc[daily.workingday.eq(1), "share"].mean()),
        "nonworking_mean_share": float(daily.loc[daily.workingday.eq(0), "share"].mean()),
        "adjusted_difference_percentage_points": float(beta[1] * 100),
        "ci95_percentage_points": [float((beta[1] - critical * se) * 100), float((beta[1] + critical * se) * 100)],
        "one_sided_p_value": p, "reject_null_at_005": bool(p < 0.05),
        "limitations": "Week clustering addresses within-week dependence, not all long-range dependence. Weather, rider composition, and unobserved availability can confound this observational association. Missing-day exclusions may select days. No causal claim."
    }, daily.reset_index()
