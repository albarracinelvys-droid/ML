"""Strict ten-minute input validation and causal features."""

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Zone 1 Power Consumption"
URL = "https://archive.ics.uci.edu/static/public/849/power+consumption+of+tetouan+city.zip"


def download(path):
    path = Path(path)
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=90) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        names = [n for n in archive.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Unexpected UCI archive")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.read(names[0]))
    return path


def read(path):
    frame = pd.read_csv(path)
    frame.columns = [" ".join(c.split()) for c in frame.columns]
    expected = [
        "DateTime",
        "Temperature",
        "Humidity",
        "Wind Speed",
        "general diffuse flows",
        "diffuse flows",
        TARGET,
        "Zone 2 Power Consumption",
        "Zone 3 Power Consumption",
    ]
    if list(frame.columns) != expected:
        raise ValueError("Unexpected input schema")
    frame.index = pd.to_datetime(frame.pop("DateTime"), format="%m/%d/%Y %H:%M")
    if (
        len(frame) < 145
        or not frame.index.to_series()
        .diff()
        .iloc[1:]
        .eq(pd.Timedelta(minutes=10))
        .all()
    ):
        raise ValueError(
            "Need at least 145 ordered, unique, continuous ten-minute rows"
        )
    frame = frame.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Missing or nonfinite values")
    return frame


def features(frame):
    x = frame.copy()
    for lag in [1, 6, 12, 138, 144]:
        x[f"lag_{lag}"] = frame[TARGET].shift(lag)
    for window in [6, 36, 144]:
        x[f"mean_{window}"] = frame[TARGET].rolling(window).mean()
        x[f"std_{window}"] = frame[TARGET].rolling(window).std()
    x["difference"] = frame[TARGET].diff(6)
    for name, values, period in [
        ("hour", frame.index.hour + frame.index.minute / 60, 24),
        ("weekday", frame.index.day_of_week, 7),
        ("year", frame.index.day_of_year, 365),
    ]:
        x[name + "_sin"] = np.sin(2 * np.pi * values / period)
        x[name + "_cos"] = np.cos(2 * np.pi * values / period)
    return x.dropna()


def supervised(frame):
    x = features(frame)
    y = frame[TARGET].shift(-6).reindex(x.index)
    return x.loc[y.notna()], y.dropna()


def splits(n):
    a, b = int(n * 0.70), int(n * 0.85)
    return np.arange(a - 6), np.arange(a, b - 6), np.arange(b, n)
