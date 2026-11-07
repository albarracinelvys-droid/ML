"""Inference uses exactly the training feature builder."""

from .data import features


def predict(bundle, history):
    x = features(history).iloc[[-1]]
    if list(x.columns) != bundle["columns"]:
        raise ValueError("Feature schema mismatch")
    return float(bundle["model"].predict(x)[0])
