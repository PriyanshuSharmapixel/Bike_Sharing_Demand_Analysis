"""Auditable charts and a generated, evidence-backed findings report."""
import json
import hashlib
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from .operations import capacity_scenario

COLORS = ["#157a6e", "#e8923d", "#405e82"]


def save_plots(hour, day, cv, evaluation, test, importance, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(name):
        plt.tight_layout()
        plt.savefig(output / name, bbox_inches="tight")
        plt.close()

    development = hour[hour.dteday < "2012-07-01"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, customer in zip(axes, ["cnt", "casual"]):
        for flag, label, color in [(1, "Working day", COLORS[0]), (0, "Non-working day", COLORS[1])]:
            development[development.workingday.eq(flag)].groupby("hr")[customer].mean().plot(ax=ax, label=label, color=color, linewidth=2)
        ax.set(title="All rentals" if customer == "cnt" else "Casual-user rentals", xlabel="Hour", ylabel="Mean recorded rentals/hour")
        ax.legend(frameon=False)
    fig.suptitle("Development-period hourly patterns (Jan 2011-Jun 2012)")
    save("hourly_patterns.png")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(day.dteday, day.cnt, color=COLORS[0], alpha=.35, linewidth=.8)
    ax.plot(day.dteday, day.cnt.rolling(28, min_periods=1).mean(), color=COLORS[0], label="28-day trailing average")
    ax.axvspan(pd.Timestamp("2012-07-01"), pd.Timestamp("2012-08-31"), color=COLORS[1], alpha=.12, label="Calibration")
    ax.axvspan(pd.Timestamp("2012-09-01"), pd.Timestamp("2012-12-31"), color=COLORS[2], alpha=.10, label="Final test")
    ax.set(title="Daily recorded rentals and experiment periods", ylabel="Rentals/day", xlabel="Date")
    ax.legend(frameon=False)
    save("daily_trend.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, column, labels in [(axes[0], "weathersit", ["Clear", "Mist/cloud", "Light rain/snow", "Severe"]), (axes[1], "season", ["Winter (1)", "Spring (2)", "Summer (3)", "Autumn (4)"])]:
        g = development.groupby(column).cnt.agg(["mean", "size"])
        ax.bar(g.index.astype(str), g["mean"], color=COLORS[0])
        ax.set_xticks(range(len(g)), [labels[int(i)-1] for i in g.index], rotation=15)
        for j, row in enumerate(g.itertuples()):
            ax.text(j, row.mean + 3, f"n={row.size:,}", ha="center", fontsize=8)
        ax.set(ylabel="Mean recorded rentals/hour", title=f"Observed association with {column}")
    fig.suptitle("Unadjusted associations; severe weather is very sparse")
    save("weather_season.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, frame, title in [(axes[0], cv.groupby("model", as_index=False).MAE.mean(), "Mean validation MAE (model selection)"), (axes[1], evaluation, "Final test MAE (Sep-Dec 2012)")]:
        ax.barh(frame.model, frame.MAE, color=COLORS[:len(frame)])
        ax.set(title=title, xlabel="Mean absolute error (rentals/hour)")
    save("model_comparison.png")

    first_week = test[test.dteday < "2012-09-08"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(first_week.timestamp, first_week.cnt, color=COLORS[2], linewidth=1.3, label="Recorded rentals")
    ax.plot(first_week.timestamp, first_week.prediction, color=COLORS[0], label="Predicted rentals")
    ax.fill_between(first_week.timestamp, first_week.lower_90, first_week.upper_90, color=COLORS[0], alpha=.16, label="Nominal 90% hourly band")
    ax.set(title="First held-out week: predicted and recorded rentals", ylabel="Rentals/hour", xlabel="Date")
    ax.legend(frameon=False, ncol=3)
    save("heldout_week.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    p = importance.nlargest(10, "MAE_increase_mean").sort_values("MAE_increase_mean")
    ax.barh(p.feature, p.MAE_increase_mean, xerr=p.MAE_increase_std, color=COLORS[0])
    ax.set(title="Permutation importance on calibration period", xlabel="Increase in MAE after shuffling (rentals/hour)")
    save("feature_importance.png")


def environment_manifest(raw):
    files = {}
    for file in sorted(Path(raw).iterdir()):
        if file.is_file():
            files[file.name] = {"sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "bytes": file.stat().st_size}
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "random_seed": 42, "input_files": files}


def write_findings(audit, hypothesis, bundle, cv, evaluation, test, groups, importance, output):
    meta = bundle["metadata"]
    winner = meta["selected_model"]
    score = evaluation.set_index("model").loc[winner]
    baseline = evaluation.set_index("model").loc["Calendar baseline", "MAE"]
    improvement = 100 * (baseline - score.MAE) / baseline
    peak = groups[(groups.group == "commute_peak") & (groups.value == "True")].iloc[0]
    sensitivity = []
    for duration in [15, 30, 45]:
        for usable in [.75, .85, .95]:
            capacity = capacity_scenario(test, trip_minutes=duration, turnaround_minutes=5, usable_fraction=usable)
            sensitivity.append({"trip_minutes": duration, "turnaround_minutes": 5, "usable_fraction": usable, "largest_hourly_workload_bikes": int(capacity.max())})
    pd.DataFrame(sensitivity).to_csv(Path(output) / "capacity_sensitivity.csv", index=False)
    ci = hypothesis["ci95_percentage_points"]
    lines = [
        "# Bike-sharing case study: final findings", "",
        "## Recommendation to the CEO", "",
        "Do not approve a fleet purchase from these aggregate counts alone. Use demand forecasts to schedule operational checks around busy periods; collect station inventory, trip duration/returns, unavailable-bike events, maintenance status, and failed rental attempts. Test redistribution at matched stations before estimating a fleet expansion's incremental value.", "",
        "## What was supplied and verified", "",
        f"- Washington, D.C., 2011-2012: {audit['hour']['rows']:,} hourly records and {audit['day']['rows']:,} daily records; {audit['hour']['total_rentals']:,} completed rentals.",
        f"- Blank cells: {audit['hour']['missing_cells']} hourly / {audit['day']['missing_cells']} daily. Duplicate date-hour/date keys: {audit['hour']['duplicate_keys']} / {audit['day']['duplicate_keys']}.",
        f"- {audit['missing_hour_slots']} missing hourly slots across {audit['incomplete_days']} days. Missing slots remain missing; zero demand is not established.",
        f"- All rental component identities pass; hourly/daily component reconciliation mismatches: {audit['daily_reconciliation_mismatches']}.",
        f"- {audit['zero_humidity_rows']} zero-humidity hourly rows flagged. No unsupported weather imputation or demand-outlier deletion.",
        "- Daily and hourly files summarize the same activity; daily records are used for reconciliation and trends, never stacked with hourly records.",
        "- README season labels are shifted relative to calendar dates: code 1 occurs in winter, 2 spring, 3 summer, 4 autumn. Codes are preserved; plots use calendar-consistent labels.", "",
        "## Hypothesis and experimental design", "",
        hypothesis["hypothesis"], "",
        f"{hypothesis['days']} complete development-period days. {hypothesis['method']}", "",
        f"Unadjusted morning shares: working {hypothesis['workingday_mean_share']:.1%}, non-working {hypothesis['nonworking_mean_share']:.1%}. Adjusted difference: **{hypothesis['adjusted_difference_percentage_points']:.2f} percentage points**, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}], one-sided p={hypothesis['one_sided_p_value']:.3g}.",
        "Reject the null at alpha=0.05." if hypothesis["reject_null_at_005"] else "Do not reject the null at alpha=0.05.",
        hypothesis["limitations"], "",
        "## Forecasting experiment", "",
        "Three expanding training windows evaluate Jan-Feb, Mar-Apr, and May-Jun 2012. Candidates: recent-eight-week calendar mean, Ridge regression, and Poisson histogram gradient boosting. Select by mean validation MAE. Fit through June 30; calibrate uncertainty on July-August; reserve September-December for final reporting. No random train/test split or target-derived input columns.", "",
        f"Selected model: **{winner}**. Final test MAE **{score.MAE:.2f} rentals/hour**, RMSE {score.RMSE:.2f}, WAPE {score.WAPE_percent:.2f}%, R² {score.R2:.3f}; MAE improvement over calendar baseline **{improvement:.1f}%**.",
        "MAE is an average absolute count error, not an accuracy percentage. WAPE is total absolute error divided by total observed rentals.", "",
        "| Model | Test MAE | Test RMSE | Test WAPE |",
        "|---|---:|---:|---:|",
        *[f"| {r.model} | {r.MAE:.2f} | {r.RMSE:.2f} | {r.WAPE_percent:.2f}% |" for r in evaluation.itertuples()], "",
        f"Predetermined working-day commute hours (07-09 and 16-18): MAE {peak.MAE:.2f}; nominal-band coverage {peak.interval_coverage_percent:.1f}%. Subgroup errors and coverage are in `error_segments.csv`.",
        f"Nominal 90% hourly intervals achieved **{meta['test_interval_coverage_percent']:.1f}%** test coverage; average width {meta['test_average_interval_width']:.1f} rentals. Calibration radius {bundle['radius']:.1f}. A nominal level is not a guarantee with seasonal/temporal drift. Do not add hourly bands to claim a calibrated daily interval.",
        f"Summed daily predictions have MAE {meta['test_daily_aggregation_metrics']['MAE']:.1f} rentals/day. They are an aggregation of the hourly model, not a separately trained daily model.", "",
        meta["forecast_contract"], "",
        "## How to interpret feature importance", "",
        "Permutation importance is measured on the calibration period after model selection, using MAE deterioration when one feature is shuffled (three repeats, seed 42). Correlated calendar/weather features share information, so this is predictive association, not a causal attribution or a promise about individual explanations.", "",
        "## Bikes needed: an explicit scenario, not an identified fleet size", "",
        "Hourly rental flow × (assumed trip + turnaround minutes) / 60 / assumed usable fraction estimates average citywide bicycle workload under a steady-state approximation. Round up. This is motivated by flow-times-cycle-time accounting and ignores precise returns, travel direction, start-of-hour inventory, and station imbalance. It cannot identify a fleet purchase quantity.", "",
        "`capacity_sensitivity.csv` shows 15/30/45-minute trip assumptions, 5-minute turnaround, and 75/85/95% usable fractions. These inputs are assumptions, not measurements. The upper hourly rental band is an optional stress scenario; it does not turn the capacity estimate into a 90% fleet guarantee.", "",
        "## Operations-manager prototype", "",
        "Run `python -m streamlit run app.py`. Explore historical patterns; enter an explicitly labeled 2012 weather scenario; inspect hourly predictions/bands; vary trip-duration/reserve assumptions; download the scenario; ask supported natural-language questions about peaks, totals, an hour, historical patterns, and capacity. No API key is required. The assistant uses bounded intent routing and computed model/data answers, not an LLM. Unsupported questions receive an explicit limitation.", "",
        "Scenario weather is constant across a day unless supplied programmatically. Apparent temperature defaults to air temperature as an approximation and is separately editable. Forecasts on synthetic weather combinations are not externally validated. Severe-weather category 4 has only three rows in the entire hourly file; do not trust extrapolation there.", "",
        "## Proposed operational experiment", "",
        "1. Instrument station stock every 5-15 minutes, trip starts/returns, maintenance, and failed searches/unlocks. Define stockout station-minutes and failed attempts as primary availability outcomes.",
        "2. Pair comparable stations by pre-period rentals and stockouts. Randomize station clusters to forecast-guided redistribution versus current operations to reduce nearby-station spillovers. Run a pre-period, then an intervention spanning working/non-working days and weather variation.",
        "3. Fix a practically meaningful improvement target and sample size from baseline variability; do not invent a required sample size without those measurements. Track truck distance, labor cost, dock-full minutes, and completed rentals as secondary/guardrail measures.",
        "4. Analyze the randomized cluster-level effect with uncertainty and operational cost. Only if usable bikes remain broadly saturated after redistribution should a measured marginal fleet-expansion trial inform procurement.", "",
        "## Reproduction and source", "",
        "See README for setup, notebook execution, and tests. Source data and README are preserved in `data/raw/`; fingerprints and tested library versions are in `reproducibility.json`. The supplied PDF is included unchanged as `case_study.pdf`.", "",
        "Dataset attribution: Hadi Fanaee-T and Joao Gama, 'Event labeling combining ensemble detectors and background knowledge', Progress in Artificial Intelligence (2013), DOI: 10.1007/s13748-013-0040-3. The supplied README requests this citation; no additional dataset license is inferred.", ""
    ]
    (Path(output) / "findings.md").write_text("\n".join(lines), encoding="utf-8")
