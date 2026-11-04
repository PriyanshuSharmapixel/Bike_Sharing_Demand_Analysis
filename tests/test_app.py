"""Headless Streamlit integration checks exercise real controls and computed answers."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_loads_and_answers():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not app.exception
    assert len(app.metric) == 4
    app.text_input[0].set_value("What is the daily total?")
    app.button[0].click().run()
    assert not app.exception
    assert any("predicted rentals across 24 hours" in x.value for x in app.markdown)
    original = app.metric[0].value
    app.slider[0].set_value(30.).run()
    assert not app.exception
    assert app.metric[0].value != original
    app.text_input[0].set_value("Should we buy more bikes?")
    app.button[0].click().run()
    assert not app.exception
    assert any("cannot identify a station shortage" in x.value for x in app.markdown)


def test_app_severe_weather_warning():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    app.sidebar.selectbox[0].set_value(4).run()
    assert not app.exception
    assert any("only three" in x.value for x in app.warning)
