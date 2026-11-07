import numpy as np
import pandas as pd

from tetouan.data import TARGET, features, splits, supervised


def sample():
    return pd.DataFrame(
        {TARGET: np.arange(600.0)},
        index=pd.date_range("2017-01-01", periods=600, freq="10min"),
    )


def test_causality():
    a = sample()
    b = a.copy()
    b.iloc[400:] = -1000
    pd.testing.assert_frame_equal(
        features(a).loc[: a.index[399]], features(b).loc[: a.index[399]]
    )


def test_target_alignment():
    x, y = supervised(sample())
    assert (y.to_numpy() == x[TARGET].to_numpy() + 6).all()


def test_purged_boundaries():
    x, _y = supervised(sample())
    tr, va, te = splits(len(x))
    assert x.index[tr[-1]] + pd.Timedelta(hours=1) < x.index[va[0]]
    assert x.index[va[-1]] + pd.Timedelta(hours=1) < x.index[te[0]]


def test_serialized_inference(tmp_path):
    import joblib
    from sklearn.linear_model import Ridge

    from tetouan.inference import predict

    x, y = supervised(sample())
    model = Ridge().fit(x, y)
    bundle = {"model": model, "columns": list(x.columns)}
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    assert np.isfinite(predict(joblib.load(path), sample()))
