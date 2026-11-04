"""Historical demo scenarios, explicit capacity assumptions, and bounded NL queries."""
import re
import numpy as np
import pandas as pd
from .model import predict

DEMO_START = pd.Timestamp("2012-09-01")
DEMO_END = pd.Timestamp("2012-12-31")


def forecast_scenario(bundle, date, temperature_c=20., humidity_percent=60., windspeed=12., weather=1, holiday=False, apparent_temperature_c=None):
    """A what-if day using constant user-specified weather, not a live forecast.

    Retain the supplied dataset's seasonal encoding: 1 winter, 2 spring,
    3 summer, 4 autumn (the README's season names are shifted).
    """
    date = pd.Timestamp(date).normalize()
    if not DEMO_START <= date <= DEMO_END:
        raise ValueError("This historical demo supports 2012-09-01 through 2012-12-31 only. Retrain on recent data for current forecasts.")
    if not (0 <= temperature_c <= 41 and 0 <= humidity_percent <= 100 and 0 <= windspeed <= 67 and weather in [1, 2, 3, 4]):
        raise ValueError("Scenario weather is outside the supported normalized input ranges.")
    month_day = date.strftime("%m-%d")
    season = 1 if month_day < "03-21" or month_day >= "12-21" else 2 if month_day < "06-21" else 3 if month_day < "09-23" else 4
    frame = pd.DataFrame({"dteday": [date] * 24, "hr": range(24)})
    frame["weekday"] = (date.dayofweek + 1) % 7
    frame["mnth"] = date.month
    frame["season"] = season
    frame["holiday"] = int(holiday)
    frame["workingday"] = int(date.dayofweek < 5 and not holiday)
    frame["weathersit"] = int(weather)
    frame["temp"] = temperature_c / 41
    # Expose this approximation in app/docs; actual apparent temperature is not
    # determined from temperature alone. An independent control can override it.
    apparent = temperature_c if apparent_temperature_c is None else apparent_temperature_c
    if not 0 <= apparent <= 50:
        raise ValueError("Apparent temperature must be between 0 and 50 degrees C for this demo.")
    frame["atemp"] = apparent / 50
    frame["hum"] = humidity_percent / 100
    frame["windspeed"] = windspeed / 67
    return scenario_predictions(bundle, frame)


def scenario_predictions(bundle, frame):
    out = frame.copy()
    out["prediction"] = predict(bundle["model"], out)
    out["lower_90"] = np.maximum(0, out.prediction - bundle["radius"])
    out["upper_90"] = out.prediction + bundle["radius"]
    return out


def capacity_scenario(predictions, trip_minutes=30., turnaround_minutes=5., usable_fraction=.85, use_upper=False):
    """Little's-Law-style hourly workload approximation; not fleet optimization.

    Rental flow/hour * cycle time in hours = average bikes occupied in cycle.
    Divide by usable fraction to illustrate an assumed reserve. This uses a
    steady-state hourly approximation, not a return/stock simulation.
    """
    if trip_minutes <= 0 or turnaround_minutes < 0 or not 0 < usable_fraction <= 1:
        raise ValueError("Trip duration must be positive, turnaround non-negative, and usable fraction in (0,1].")
    flow = np.asarray(predictions["upper_90" if use_upper else "prediction"], dtype=float)
    if (flow < 0).any() or not np.isfinite(flow).all():
        raise ValueError("Rental estimates must be finite and non-negative.")
    return np.ceil(flow * (trip_minutes + turnaround_minutes) / 60 / usable_fraction).astype(int)


def answer_question(question, forecast, historical, trip_minutes=30., turnaround_minutes=5., usable_fraction=.85, use_upper=False):
    """Deterministic intent routing with computed answers. No generated SQL/code.

    Small supported English vocabulary is intentional. Unknown/ambiguous requests
    receive an explicit limitation rather than an invented number. Date questions
    must match the selected scenario; arbitrary historical SQL is out of scope.
    """
    q = question.lower().strip()
    if not q:
        return "Ask about peak hours, total rentals, hourly demand, historical working-day patterns, or capacity assumptions."
    if re.search(r"\b(station|location|neighbou?rhood|redistribut|rebalance|where|buy|purchase|unmet|shortage)\w*", q):
        return "The supplied data has citywide completed rental counts, not station inventory, failed rental attempts, trip durations, or fleet size. I cannot identify a station shortage or justify purchasing bikes. Collect those operational records and pilot redistribution first."
    if re.search(r"\b(today|tomorrow|next week|202[0-9])\b", q):
        return "This is a historical 2012 demonstration, not a live forecast. Ask about the selected scenario date; current forecasts require recent demand and forecast weather."
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", q)
    selected = str(pd.Timestamp(forecast.dteday.iloc[0]).date())
    if any(date != selected for date in dates):
        return f"The current scenario is {selected}. Change the date control before asking about a different date."
    if re.search(r"\b(why|explain|cause|reason)\b", q):
        return "The model uses hour, calendar, workday, and weather features. Associations and feature importance do not prove causes. See the validation and permutation-importance results; this prototype does not provide a causal explanation for one prediction."
    if re.search(r"\b(history|historical|working days?|weekends?|non.?working)\b", q):
        means = historical.groupby(["workingday", "hr"]).cnt.mean()
        peaks = {int(k): int(g.droplevel(0).idxmax()) for k, g in means.groupby(level=0)}
        return f"Across the supplied 2011-2012 observed rows, average working-day rentals peak at {peaks[1]:02d}:00 and non-working-day rentals peak at {peaks[0]:02d}:00. These are historical associations, not station inventory requirements."
    if re.search(r"\b(bikes?|capacity|fleet|reserve)\b", q):
        capacity = capacity_scenario(forecast, trip_minutes, turnaround_minutes, usable_fraction, use_upper)
        basis = "upper hourly band" if use_upper else "point predictions"
        return f"Under your assumptions ({trip_minutes:g} min trip + {turnaround_minutes:g} min turnaround, {usable_fraction:.0%} usable; {basis}), the largest hourly workload estimate is {capacity.max():,} bikes. This steady-state citywide scenario is not a purchase recommendation; location imbalance and unserved riders are unmeasured."
    if re.search(r"\b(peak|busiest|busy|highest|top)\b", q):
        top = forecast.nlargest(3, "prediction")
        values = "; ".join(f"{int(r.hr):02d}:00: {r.prediction:.0f} predicted rentals" for r in top.itertuples())
        return f"Top three hours for the {selected} weather scenario: {values}. Counts are rental starts, not distinct bikes."
    if re.search(r"\b(total|whole day|daily|all day)\b", q):
        return f"The {selected} scenario totals {forecast.prediction.sum():,.0f} predicted rentals across 24 hours. Summed hourly counts are not unique bikes. Hourly uncertainty bands cannot be added to obtain a calibrated daily interval."
    match = re.search(r"\b(?:at|hour)\s+(\d{1,2})(?::00)?\s*(am|pm)?\b", q)
    if match:
        hr = int(match.group(1))
        meridiem = match.group(2)
        if meridiem and not 1 <= hr <= 12:
            return "For AM/PM, use hours 1-12. For 24-hour time, use 0-23."
        if meridiem:
            hr = hr % 12 + (12 if meridiem == "pm" else 0)
        if not 0 <= hr <= 23:
            return "Please specify an hour between 0 and 23."
        row = forecast.loc[forecast.hr.eq(hr)].iloc[0]
        return f"At {hr:02d}:00 on {selected}: {row.prediction:.0f} predicted rentals; nominal hourly band {row.lower_90:.0f}-{row.upper_90:.0f}. Temporal drift means the nominal 90% level is not guaranteed."
    return "Supported questions: 'What are the peak hours?', 'What is the daily total?', 'How many rentals at 8 am?', 'Compare historical working days and weekends', and 'How many bikes under these capacity assumptions?'. Adjust date/weather controls for other scenarios."
