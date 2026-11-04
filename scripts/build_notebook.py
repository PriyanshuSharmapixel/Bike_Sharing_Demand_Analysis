"""Build the explanatory notebook, then optionally execute it with --execute."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import argparse
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md("""# Bike-Sharing Demand Analysis & Operations Prototype

**Business question:** Riders cannot find bikes while others sit unused. Should the company buy more bikes?

**Approach:** validate the supplied data, investigate demand patterns, test one predefined hypothesis, forecast hourly recorded rentals, and provide an operations prototype with natural-language questions.

**Honest boundary:** these are completed rentals, not all desired trips or unique bikes. Without station inventories, failed rental attempts and trip durations, the exact fleet size and station redistribution cannot be identified.

Run cells top to bottom. Install the repository dependencies first (see README). All raw source files are preserved in `data/raw/`."""),
        code("""from pathlib import Path
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image

ROOT = Path.cwd().resolve()
if not (ROOT / 'src').exists():
    ROOT = ROOT.parent
assert (ROOT / 'src' / 'bike_demand').exists(), 'Open this notebook from the repository.'
sys.path.insert(0, str(ROOT / 'src'))
from bike_demand.data import load_data, audit, make_features
from bike_demand.hypothesis import test_morning_share
from bike_demand.pipeline import run
from bike_demand.operations import forecast_scenario, capacity_scenario, answer_question

hour, day = load_data()
print('Hourly:', hour.shape, '| Daily:', day.shape)
display(hour.head())"""),
        md("""## 1. Feature understanding and assumptions

| Field | Meaning | Treatment |
|---|---|---|
| `dteday`, `hr` | Rental date and hour | Sort time; derive calendar/cyclic features |
| `weekday`, `workingday`, `holiday` | Calendar/day type | Known in advance; 0 Sunday, 6 Saturday |
| `season`, `mnth`, `yr` | Season/month/year codes | Preserve codes; derive elapsed time from date |
| `temp`, `atemp` | Scaled air/feels-like temperature | README scales: ×41 and ×50 °C |
| `hum`, `windspeed` | Scaled humidity/wind | README scales: ×100 and ×67 |
| `weathersit` | Weather code 1–4 | Categorical; code 4 extremely sparse |
| `casual`, `registered` | Components of total rentals | Explore, but **exclude from predictors** |
| `cnt` | Total recorded rentals | Prediction target |
| `instant` | Row number | Exclude |

The README names seasons 1–4 as spring/summer/fall/winter, but actual dates map them to winter/spring/summer/autumn. We preserve numeric codes and explain the mismatch.

Daily data is used for validation and trend summaries; it is not extra independent training data. Weather is observed retrospectively; real future use requires forecast weather. No station-level or causal claims are assumed."""),
        code("""display(pd.DataFrame({
    'dtype': hour.dtypes.astype(str),
    'missing': hour.isna().sum(),
    'unique': hour.nunique()
}))
display(hour[['cnt', 'temp', 'atemp', 'hum', 'windspeed']].describe())"""),
        md("""## 2. Data quality

Check unique date-hour/date keys, missing cells, category/range validity, calendar consistency, non-negative counts, `cnt = casual + registered`, and per-day reconciliation of all three counts. Enumerate missing timestamps separately from missing cells.

**Policy:** do not zero-fill absent hours; do not drop extreme demand or unusual weather without evidence. Flag zero humidity. Count reconciliation establishes consistency, not why hours are absent."""),
        code("""quality, missing_hours = audit(hour, day)
display(pd.DataFrame({k: quality[k] for k in ['hour', 'day']}))
print('Missing hourly slots:', quality['missing_hour_slots'])
print('Incomplete days:', quality['incomplete_days'])
print('Daily reconciliation mismatches:', quality['daily_reconciliation_mismatches'])
display(missing_hours.head(10))"""),
        md("""## 3. Explore demand without inspecting final-test errors

Use the development period (before July 2012) to inspect time-of-day patterns. Full-period daily trends are descriptive context, not used to tune models. Differences by weather or season are unadjusted associations and may reflect other factors."""),
        code("""development = hour[hour.dteday < '2012-07-01']
patterns = development.groupby(['hr', 'workingday']).cnt.mean().unstack()
patterns.columns = ['Non-working day', 'Working day']
ax = patterns.plot(figsize=(10, 4), color=['#e8923d', '#157a6e'])
ax.set(xlabel='Hour', ylabel='Mean recorded rentals/hour', title='Development-period hourly demand')
plt.tight_layout()
plt.show()
display(development.groupby('weathersit').cnt.agg(['count', 'mean', 'median']))"""),
        md("""## 4. One testable hypothesis

**H1:** The share of daily rentals occurring from 07:00 through 09:59 is higher on working days than on non-working days, adjusting for calendar month-year. **H0:** the adjusted difference is ≤0.

Use complete 24-hour development-period days, one observation per day. Regress morning share on working-day status and month-year fixed effects. Cluster standard errors by Monday–Sunday week; use a one-sided t-test at α=0.05 and report a two-sided 95% CI. This reduces within-week pseudo-replication but does not eliminate confounding or all time dependence. This is not a causal experiment."""),
        code("""hypothesis, hypothesis_daily = test_morning_share(hour)
display(pd.Series(hypothesis))
display(hypothesis_daily.head())"""),
        md("""## 5. Features and experimental design

Exclude `cnt`, `casual`, `registered`, and `instant`. Use calendar, weather, elapsed days, and sine/cosine hour and annual cycles. A cyclic encoding puts 23:00 near 00:00. No lagged target features are used, avoiding recursive forecasts and unavailable within-day actuals.

| Stage | Dates | Purpose |
|---|---|---|
| Validation 1 | Jan–Feb 2012; train earlier dates | Compare fixed candidates |
| Validation 2 | Mar–Apr 2012; train earlier dates | Compare fixed candidates |
| Validation 3 | May–Jun 2012; train earlier dates | Compare fixed candidates |
| Final fit | Jan 2011–Jun 2012 | Fit selected model |
| Calibration | Jul–Aug 2012 | Estimate hourly uncertainty radius |
| Final test | Sep–Dec 2012 | Report performance once |

Candidates: recent-eight-week hour/weekday/workday mean; Ridge with one-hot categories and scaled numeric features; Poisson histogram gradient boosting. Select the lowest mean validation MAE. Settings are fixed, not an extensive search.

**Metrics:** MAE, RMSE, WAPE, R², bias; predetermined commute-hour errors and subgroup interval coverage. Evaluate against a simple baseline rather than describing regression as classification accuracy."""),
        code("""X = make_features(hour)
assert set(X).isdisjoint({'cnt', 'casual', 'registered', 'instant'})
display(X.head())
bundle, cv, test_metrics, test_predictions = run()
display(cv.round(3))
print('Selected model:', bundle['metadata']['selected_model'])"""),
        md("""## 6. Final held-out results and uncertainty

The following results are generated from the supplied records, not hardcoded example scores. A split-calibration absolute-residual radius produces nominal 90% hourly bands. Serial dependence and seasonal drift mean 90% coverage is not guaranteed; show observed coverage. Clipping the lower bound at zero respects non-negative counts. Do not sum hourly bands to claim a calibrated daily interval."""),
        code("""display(test_metrics.round(3))
display(pd.Series(bundle['metadata']))
segments = pd.read_csv(ROOT / 'reports' / 'error_segments.csv')
display(segments.round(3))
display(Image(filename=str(ROOT / 'reports' / 'figures' / 'heldout_week.png')))
display(Image(filename=str(ROOT / 'reports' / 'figures' / 'model_comparison.png')))"""),
        md("""## 7. Interpretation and daily cross-check

Permutation importance is measured on the calibration set after model selection. Shuffling one feature can produce unusual combinations; correlated features share information. Importance is predictive association, not causality.

The daily file provides exact aggregation checks and broader trends. Daily predictions below sum the hourly model; no second daily model or duplicate activity is introduced."""),
        code("""display(pd.read_csv(ROOT / 'reports' / 'feature_importance.csv').round(3))
display(Image(filename=str(ROOT / 'reports' / 'figures' / 'feature_importance.png')))
display(Image(filename=str(ROOT / 'reports' / 'figures' / 'daily_trend.png')))
display(Image(filename=str(ROOT / 'reports' / 'figures' / 'weather_season.png')))
daily_predictions = test_predictions.groupby('dteday')[['cnt', 'prediction']].sum()
daily_comparison = daily_predictions.join(day.set_index('dteday')[['cnt']], rsuffix='_source')
assert daily_comparison.cnt.equals(daily_comparison.cnt_source)
display(daily_comparison.head())"""),
        md("""## 8. Translate rental forecasts into a transparent capacity scenario

Hourly rental starts are not unique bikes. Under an explicitly assumed steady state:

**Average bicycle workload ≈ rentals/hour × (trip minutes + turnaround minutes) / 60 / usable fraction.**

Use assumptions, not invented measured values: 30-minute trips, 5-minute turnaround, 85% usable. Show sensitivity. This ignores direction, station imbalance, exact return times and lost demand, so it cannot identify how many bikes to buy. The largest hourly workload is a planning scenario, not a validated fleet requirement.

The example date is restricted to the historical holdout period. Constant weather across the day is hypothetical; air and apparent temperature are independently controllable in the app."""),
        code("""scenario = forecast_scenario(bundle, '2012-10-15', temperature_c=20,
    humidity_percent=60, windspeed=12, weather=1, holiday=False,
    apparent_temperature_c=20)
scenario['workload_bikes'] = capacity_scenario(scenario, 30, 5, .85)
display(scenario[['hr', 'prediction', 'lower_90', 'upper_90', 'workload_bikes']].round(1))
display(pd.read_csv(ROOT / 'reports' / 'capacity_sensitivity.csv'))"""),
        md("""## 9. Natural-language operations prototype

The local Streamlit app exposes date/weather controls, hourly forecasts, interval limitations, sensitivity analysis, CSV export, evidence, and plain-English questions. This implementation is a bounded intent router that calculates answers from the model and data. It has no LLM dependency or API key and does not execute user-generated code/SQL. Unsupported/ambiguous intents receive limitations.

Run from the repository terminal: `python -m streamlit run app.py`. Questions about specific stations, purchasing, or current dates are declined because required data is missing."""),
        code("""for question in ['What are the peak hours?', 'What is the daily total?',
                 'How many rentals at 8 am?', 'How many bikes under these capacity assumptions?',
                 'Should we buy more bikes?']:
    print('Q:', question)
    print('A:', answer_question(question, scenario, hour))
    print()"""),
        md("""## 10. Business recommendation, next experiment, and findings

Do not infer a fleet purchase from aggregate completed rentals. Use hourly forecasts to prioritize availability checks. Collect station inventory, failed searches/unlocks, trip durations/returns, and maintenance. Randomize matched station clusters to forecast-guided redistribution versus current operations; measure stockout minutes and failed attempts with cost/dock-full guardrails. Estimate sample size from pre-period variation before fixing trial length. Test additional bikes only if measured scarcity remains after redistribution.

Known limitations: two historical years, observed rather than forecast weather, no station/stockout data, missing hours of unknown cause, scarce severe weather, possible drift and underprediction, and no field validation of capacity assumptions.

The complete generated report follows."""),
        code("display(Markdown((ROOT / 'reports' / 'findings.md').read_text()))"),
        md("""## Attribution

Dataset supplied with the case study. Hadi Fanaee-T and Joao Gama, *Event labeling combining ensemble detectors and background knowledge*, Progress in Artificial Intelligence (2013), DOI: 10.1007/s13748-013-0040-3. Original README and case-study PDF retained in `data/raw/`. Input fingerprints and tested dependency versions are in `reports/reproducibility.json`."""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}})
    target = ROOT / "notebooks" / "bike_demand_analysis.ipynb"
    nbf.write(notebook, target)
    return notebook, target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--in-process', action='store_true', help='Execute cells with IPython without a separate kernel server')
    args = parser.parse_args()
    notebook, target = build()
    if args.in_process:
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.utils.capture import capture_output
        shell = InteractiveShell.instance()
        os.chdir(ROOT)
        count = 0
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            count += 1
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=True)
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(f'Notebook cell {count} failed: {result.error_before_exec or result.error_in_exec}')
            outputs = []
            if captured.stdout:
                outputs.append(nbf.v4.new_output('stream', name='stdout', text=captured.stdout))
            if captured.stderr:
                outputs.append(nbf.v4.new_output('stream', name='stderr', text=captured.stderr))
            for rich in captured.outputs:
                outputs.append(nbf.v4.new_output('display_data', data=rich.data, metadata=rich.metadata))
            cell.execution_count = count
            cell.outputs = outputs
        notebook.metadata['verification'] = 'All code cells executed sequentially using IPython in-process.'
        nbf.write(notebook, target)
    elif args.execute:
        from nbclient import NotebookClient
        client = NotebookClient(notebook, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}})
        client.execute()
        nbf.write(notebook, target)
    print(target)
