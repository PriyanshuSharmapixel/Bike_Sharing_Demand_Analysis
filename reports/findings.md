# Bike-sharing case study: final findings

## Recommendation to the CEO

Do not approve a fleet purchase from these aggregate counts alone. Use demand forecasts to schedule operational checks around busy periods; collect station inventory, trip duration/returns, unavailable-bike events, maintenance status, and failed rental attempts. Test redistribution at matched stations before estimating a fleet expansion's incremental value.

## What was supplied and verified

- Washington, D.C., 2011-2012: 17,379 hourly records and 731 daily records; 3,292,679 completed rentals.
- Blank cells: 0 hourly / 0 daily. Duplicate date-hour/date keys: 0 / 0.
- 165 missing hourly slots across 76 days. Missing slots remain missing; zero demand is not established.
- All rental component identities pass; hourly/daily component reconciliation mismatches: 0.
- 22 zero-humidity hourly rows flagged. No unsupported weather imputation or demand-outlier deletion.
- Daily and hourly files summarize the same activity; daily records are used for reconciliation and trends, never stacked with hourly records.
- README season labels are shifted relative to calendar dates: code 1 occurs in winter, 2 spring, 3 summer, 4 autumn. Codes are preserved; plots use calendar-consistent labels.

## Hypothesis and experimental design

Working days have a greater 07:00-09:59 share of daily rentals than non-working days, adjusting for calendar month-year.

477 complete development-period days. Daily OLS with month-year fixed effects; ISO-style Monday-Sunday week-clustered covariance; one-sided t-test, alpha=0.05; two-sided 95% confidence interval.

Unadjusted morning shares: working 22.4%, non-working 7.3%. Adjusted difference: **15.26 percentage points**, 95% CI [14.42, 16.10], one-sided p=2.05e-50.
Reject the null at alpha=0.05.
Week clustering addresses within-week dependence, not all long-range dependence. Weather, rider composition, and unobserved availability can confound this observational association. Missing-day exclusions may select days. No causal claim.

## Forecasting experiment

Three expanding training windows evaluate Jan-Feb, Mar-Apr, and May-Jun 2012. Candidates: recent-eight-week calendar mean, Ridge regression, and Poisson histogram gradient boosting. Select by mean validation MAE. Fit through June 30; calibrate uncertainty on July-August; reserve September-December for final reporting. No random train/test split or target-derived input columns.

Selected model: **Histogram boosting**. Final test MAE **47.98 rentals/hour**, RMSE 72.91, WAPE 19.97%, R² 0.888; MAE improvement over calendar baseline **37.9%**.
MAE is an average absolute count error, not an accuracy percentage. WAPE is total absolute error divided by total observed rentals.

| Model | Test MAE | Test RMSE | Test WAPE |
|---|---:|---:|---:|
| Calendar baseline | 77.24 | 124.20 | 32.15% |
| Ridge | 99.45 | 134.97 | 41.40% |
| Histogram boosting | 47.98 | 72.91 | 19.97% |

Predetermined working-day commute hours (07-09 and 16-18): MAE 104.96; nominal-band coverage 64.4%. Subgroup errors and coverage are in `error_segments.csv`.
Nominal 90% hourly intervals achieved **89.4%** test coverage; average width 206.5 rentals. Calibration radius 117.8. A nominal level is not a guarantee with seasonal/temporal drift. Do not add hourly bands to claim a calibrated daily interval.
Summed daily predictions have MAE 947.6 rentals/day. They are an aggregation of the hourly model, not a separately trained daily model.

Retrospective hourly rental estimates conditional on observed weather. Operational use needs hourly weather forecasts; forecast-weather degradation has not been measured. No lags, no target components, no station allocation. Not a 2026 production forecast.

## How to interpret feature importance

Permutation importance is measured on the calibration period after model selection, using MAE deterioration when one feature is shuffled (three repeats, seed 42). Correlated calendar/weather features share information, so this is predictive association, not a causal attribution or a promise about individual explanations.

## Bikes needed: an explicit scenario, not an identified fleet size

Hourly rental flow × (assumed trip + turnaround minutes) / 60 / assumed usable fraction estimates average citywide bicycle workload under a steady-state approximation. Round up. This is motivated by flow-times-cycle-time accounting and ignores precise returns, travel direction, start-of-hour inventory, and station imbalance. It cannot identify a fleet purchase quantity.

`capacity_sensitivity.csv` shows 15/30/45-minute trip assumptions, 5-minute turnaround, and 75/85/95% usable fractions. These inputs are assumptions, not measurements. The upper hourly rental band is an optional stress scenario; it does not turn the capacity estimate into a 90% fleet guarantee.

## Operations-manager prototype

Run `python -m streamlit run app.py`. Explore historical patterns; enter an explicitly labeled 2012 weather scenario; inspect hourly predictions/bands; vary trip-duration/reserve assumptions; download the scenario; ask supported natural-language questions about peaks, totals, an hour, historical patterns, and capacity. No API key is required. The assistant uses bounded intent routing and computed model/data answers, not an LLM. Unsupported questions receive an explicit limitation.

Scenario weather is constant across a day unless supplied programmatically. Apparent temperature defaults to air temperature as an approximation and is separately editable. Forecasts on synthetic weather combinations are not externally validated. Severe-weather category 4 has only three rows in the entire hourly file; do not trust extrapolation there.

## Proposed operational experiment

1. Instrument station stock every 5-15 minutes, trip starts/returns, maintenance, and failed searches/unlocks. Define stockout station-minutes and failed attempts as primary availability outcomes.
2. Pair comparable stations by pre-period rentals and stockouts. Randomize station clusters to forecast-guided redistribution versus current operations to reduce nearby-station spillovers. Run a pre-period, then an intervention spanning working/non-working days and weather variation.
3. Fix a practically meaningful improvement target and sample size from baseline variability; do not invent a required sample size without those measurements. Track truck distance, labor cost, dock-full minutes, and completed rentals as secondary/guardrail measures.
4. Analyze the randomized cluster-level effect with uncertainty and operational cost. Only if usable bikes remain broadly saturated after redistribution should a measured marginal fleet-expansion trial inform procurement.

## Reproduction and source

See README for setup, notebook execution, and tests. Source data and README are preserved in `data/raw/`; fingerprints and tested library versions are in `reproducibility.json`. The supplied PDF is included unchanged as `case_study.pdf`.

Dataset attribution: Hadi Fanaee-T and Joao Gama, 'Event labeling combining ensemble detectors and background knowledge', Progress in Artificial Intelligence (2013), DOI: 10.1007/s13748-013-0040-3. The supplied README requests this citation; no additional dataset license is inferred.
