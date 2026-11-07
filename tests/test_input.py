import numpy as np
import pandas as pd
import pytest

from tetouan.data import read


def test_schema_and_continuity(tmp_path):
    columns = [
        "Temperature",
        "Humidity",
        "Wind Speed",
        "general diffuse flows",
        "diffuse flows",
        "Zone 1 Power Consumption",
        "Zone 2 Power Consumption",
        "Zone 3 Power Consumption",
    ]
    frame = pd.DataFrame(np.ones((150, 8)), columns=columns)
    frame.insert(
        0,
        "DateTime",
        pd.date_range("2017-01-01", periods=150, freq="10min").strftime(
            "%m/%d/%Y %H:%M"
        ),
    )
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    assert len(read(path)) == 150
    frame.iloc[10, 0] = frame.iloc[9, 0]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="continuous"):
        read(path)


def test_streamlit_startup():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    assert not app.exception
