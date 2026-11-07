"""Inference uses exactly the training feature builder."""

from .data import features


def predict(bundle, history):
    x = features(history).iloc[[-1]]
    if not set(bundle["columns"]).issubset(x.columns):
        raise ValueError("Feature schema mismatch")
    return float(bundle["model"].predict(x.loc[:, bundle["columns"]])[0])
