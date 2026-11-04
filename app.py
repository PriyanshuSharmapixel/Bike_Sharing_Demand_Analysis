"""Local operations prototype. Run with python -m streamlit run app.py."""
from pathlib import Path
import sys
import json
import datetime as dt
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from bike_demand.data import load_data
from bike_demand.operations import forecast_scenario, capacity_scenario, answer_question

st.set_page_config(page_title="Bike Demand | Operations", page_icon="🚲", layout="wide")
st.markdown("""<style>
.block-container {padding-top:2rem; max-width:1400px;}
[data-testid="stMetric"] {background:#f0f7f5;padding:18px;border-radius:12px;border:1px solid #d4e8e1;}
[data-testid="stMetric"] label, [data-testid="stMetricValue"] {color:#163d34;}
h1,h2,h3 {letter-spacing:-.025em;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    path = ROOT / "models" / "demand_model.joblib"
    if not path.exists():
        from bike_demand.pipeline import run
        run()
    # Load only our locally generated artifact. Never accept uploaded pickle files.
    bundle = joblib.load(path)
    hour, day = load_data()
    reports = ROOT / "reports"
    return bundle, hour, day, pd.read_csv(reports / "test_metrics.csv"), json.loads((reports / "data_quality.json").read_text())


st.caption("CAPITAL BIKESHARE · OPERATIONS DEMONSTRATOR")
st.title("Plan around demand.")
st.write("Explore rental patterns, test weather scenarios, and make operational assumptions visible.")
st.info("Historical demonstration using 2011–2012 data. Predictions are completed rentals across the city; they do not measure station shortages or unique bikes required.")
with st.spinner("Loading the analysis and demand model. First launch may train the model…"):
    bundle, hour, day, scores, quality = load_assets()

with st.sidebar:
    st.header("Weather scenario")
    date = st.date_input("Historical scenario date", value=dt.date(2012, 10, 15), min_value=dt.date(2012, 9, 1), max_value=dt.date(2012, 12, 31))
    st.caption("Choose a held-out-period date. Weather inputs below are hypothetical and constant across the day.")
    temp = st.slider("Air temperature (°C)", 0., 41., 20., .5)
    apparent = st.slider("Feels-like temperature (°C)", 0., 50., 20., .5)
    humidity = st.slider("Humidity (%)", 0, 100, 60)
    wind = st.slider("Wind speed (dataset scale × 67)", 0., 67., 12., .5)
    weather = st.selectbox("Weather category", [1, 2, 3, 4], format_func=lambda x: {1: "Clear / partly cloudy", 2: "Mist / cloudy", 3: "Light rain / snow", 4: "Severe rain / snow (sparse data)"}[x])
    holiday_default = bool(day.loc[day.dteday.eq(pd.Timestamp(date)), "holiday"].iloc[0])
    holiday = st.checkbox("Holiday", value=holiday_default, key=f"holiday_{date}")
    st.divider()
    st.subheader("Capacity assumptions")
    duration = st.slider("Average trip duration (minutes)", 5, 90, 30, 5)
    turnaround = st.slider("Turnaround time (minutes)", 0, 30, 5)
    usable = st.slider("Assumed usable fleet (%)", 50, 100, 85, 5) / 100
    stress = st.checkbox("Use upper hourly rental band for capacity", value=False)

forecast = forecast_scenario(bundle, date, temp, humidity, wind, weather, holiday, apparent)
forecast["workload_bikes"] = capacity_scenario(forecast, duration, turnaround, usable, stress)
if weather == 4:
    st.warning("Severe weather has only three hourly observations in the complete dataset. Scenario estimates are unreliable in this category.")
peak = forecast.loc[forecast.prediction.idxmax()]
cols = st.columns(4)
cols[0].metric("Scenario rentals / day", f"{forecast.prediction.sum():,.0f}")
cols[1].metric("Busiest hour", f"{int(peak.hr):02d}:00")
cols[2].metric("Peak rentals / hour", f"{peak.prediction:,.0f}")
cols[3].metric("Peak workload scenario", f"{forecast.workload_bikes.max():,} bikes")
st.caption("Workload = rental flow × assumed cycle time ÷ usable fraction. This is a steady-state scenario, not an exact fleet size or purchase recommendation.")

planning, evidence, assistant = st.tabs(["Demand & capacity", "Evidence & limitations", "Ask operations"])
with planning:
    st.subheader(f"Hourly rental scenario · {date}")
    st.line_chart(forecast.set_index("hr")[["prediction", "lower_90", "upper_90"]].rename(columns={"prediction": "Predicted rentals", "lower_90": "Lower hourly band", "upper_90": "Upper hourly band"}), x_label="Hour", y_label="Rental starts / hour", color=["#157a6e", "#b2c7c1", "#e8923d"])
    st.caption(f"Bands are nominal 90% hourly intervals. Actual held-out coverage: {bundle['metadata']['test_interval_coverage_percent']:.1f}%. They do not provide a calibrated daily or fleet interval.")
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Priority hours for operational checks")
        top = forecast.nlargest(5, "prediction")[["hr", "prediction", "lower_90", "upper_90", "workload_bikes"]].round(1)
        st.dataframe(top, hide_index=True, width="stretch")
        st.write("Check station availability before these busy hours. The aggregate dataset cannot tell you where to redistribute bikes.")
    with right:
        st.subheader("How assumptions affect workload")
        sensitivity = []
        for minutes in [15, 30, 45, 60]:
            sensitivity.append({"Trip minutes": minutes, "Peak workload bikes": int(capacity_scenario(forecast, minutes, turnaround, usable, stress).max())})
        st.bar_chart(pd.DataFrame(sensitivity).set_index("Trip minutes"), color="#157a6e")
        st.caption("Durations are hypothetical. Validate them using trip records before making staffing or procurement commitments.")
    st.download_button("Download this scenario (CSV)", forecast.to_csv(index=False).encode(), file_name=f"bike_scenario_{date}.csv", mime="text/csv")

with evidence:
    st.subheader("What the held-out experiment shows")
    st.write("Model selection: three expanding validation windows. Final fit: through June 2012. Uncertainty calibration: July–August. Final test: September–December.")
    st.dataframe(scores.round(3), hide_index=True, width="stretch")
    st.write(f"Selected using validation MAE: **{bundle['metadata']['selected_model']}**. MAE is an average count error, not an accuracy percentage.")
    st.warning("Evaluation uses observed weather. Real operations would require weather forecasts; their effect on error has not been evaluated.")
    st.image(str(ROOT / "reports" / "figures" / "hourly_patterns.png"))
    st.image(str(ROOT / "reports" / "figures" / "daily_trend.png"))
    with st.expander("Data quality, hypothesis, and feature importance"):
        st.json(quality)
        st.json(json.loads((ROOT / "reports" / "hypothesis.json").read_text()))
        st.image(str(ROOT / "reports" / "figures" / "feature_importance.png"))
    st.subheader("Before deciding to buy bikes")
    st.write("Collect station inventory, failed rental attempts, trip durations/returns, and maintenance status. Pilot forecast-guided redistribution against current operations, measuring stockout minutes, failed attempts, and transport costs.")
    st.download_button("Download full findings (Markdown)", (ROOT / "reports" / "findings.md").read_bytes(), file_name="bike_sharing_findings.md")

with assistant:
    st.subheader("Ask about the selected scenario")
    st.caption("A bounded English-language prototype using computed answers. No LLM or API key. Change date and weather in the sidebar; complex unsupported requests are declined.")
    example = st.selectbox("Example questions", ["What are the peak hours?", "What is the daily total?", "How many rentals at 8 am?", "Compare historical working days and weekends", "How many bikes under these capacity assumptions?", "Should we buy more bikes?"])
    question = st.text_input("Your question", value=example)
    if st.button("Answer question", type="primary"):
        st.write(answer_question(question, forecast, hour, duration, turnaround, usable, stress))
