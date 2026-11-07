"""Chronological benchmark; validation selects winner, test only evaluates."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytabkit import RealMLP_TD_Regressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import TARGET, download, read, splits, supervised


def metrics(y, p):
    return {
        "mae": mean_absolute_error(y, p),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": r2_score(y, p),
    }


def run(path, output, epochs=16):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frame = read(path)
    audit = {
        "rows": len(frame),
        "columns": list(frame.columns),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "missing": int(frame.isna().sum().sum()),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "interval_minutes": 10,
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    x, y = supervised(frame)
    tr, va, te = splits(len(x))
    frame.iloc[: int(len(frame) * 0.7)].describe().to_csv(
        out / "training_statistics.csv"
    )
    frame.iloc[: int(len(frame) * 0.7)].corr().to_csv(out / "training_correlations.csv")
    training = frame.iloc[: int(len(frame) * 0.7)]
    training.groupby(training.index.hour).mean().to_csv(out / "daily_patterns.csv")
    training.groupby(training.index.day_of_week).mean().to_csv(
        out / "weekly_patterns.csv"
    )
    training.resample("MS").mean().to_csv(out / "monthly_patterns.csv")
    q1, q3 = training.quantile(0.25), training.quantile(0.75)
    (
        (training < q1 - 1.5 * (q3 - q1)) | (training > q3 + 1.5 * (q3 - q1))
    ).sum().to_csv(out / "iqr_outliers.csv")
    frame[TARGET].iloc[: int(len(frame) * 0.7)].resample("D").mean().plot(
        title="Training daily demand"
    )
    plt.tight_layout()
    plt.savefig(out / "eda.png")
    plt.close()
    models = {
        "dummy": DummyRegressor(),
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "forest": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=4, n_jobs=4, random_state=42
        ),
        "hist": HistGradientBoostingRegressor(
            max_iter=150, early_stopping=False, random_state=42
        ),
        "realmlp": RealMLP_TD_Regressor(
            n_epochs=epochs, device="cpu", random_state=42, n_threads=4, verbosity=0
        ),
    }
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    models.update(
        xgb=XGBRegressor(n_estimators=150, n_jobs=4, random_state=42),
        lgbm=LGBMRegressor(n_estimators=150, n_jobs=4, random_state=42, verbosity=-1),
        catboost=CatBoostRegressor(
            iterations=150, thread_count=4, random_seed=42, verbose=False
        ),
    )
    results, fitted = [], {}
    for name, model in models.items():
        start = time.perf_counter()
        print("Training", name, flush=True)
        try:
            if name == "realmlp":
                model.fit(x.iloc[tr], y.iloc[tr], x.iloc[va], y.iloc[va])
            else:
                model.fit(x.iloc[tr], y.iloc[tr])
            elapsed = time.perf_counter() - start
            row = dict(
                model=name,
                train_seconds=elapsed,
                **metrics(y.iloc[va], model.predict(x.iloc[va])),
            )
            results.append(row)
            fitted[name] = model
        except Exception as exc:  # noqa: BLE001 -- persist each model failure and continue benchmark
            (out / (name + "_failure.txt")).write_text(repr(exc))
            print(name, "FAILED", repr(exc), flush=True)
    validation = pd.DataFrame(results).sort_values("rmse")
    validation.to_csv(out / "validation.csv", index=False)
    winner = validation.iloc[0]["model"]
    test = []
    for name, model in fitted.items():
        start = time.perf_counter()
        p = model.predict(x.iloc[te])
        test.append(
            dict(
                model=name,
                inference_seconds=time.perf_counter() - start,
                **metrics(y.iloc[te], p),
            )
        )
    for name, col in [("persistence", TARGET), ("seasonal", "lag_138")]:
        test.append(dict(model=name, **metrics(y.iloc[te], x.iloc[te][col])))
    pd.DataFrame(test).to_csv(out / "test.csv", index=False)
    model = fitted[winner]
    bundle = {
        "model": model,
        "columns": list(x.columns),
        "winner": winner,
        "horizon_minutes": 60,
    }
    bundle["training_min"] = x.iloc[tr].min()
    bundle["training_max"] = x.iloc[tr].max()
    joblib.dump(bundle, out / "model.joblib")
    p = joblib.load(out / "model.joblib")["model"].predict(x.iloc[te])
    pred = pd.DataFrame({"observed": y.iloc[te], "predicted": p})
    pred["error"] = pred.predicted - pred.observed
    pred.to_csv(out / "predictions.csv")
    threshold = y.iloc[tr].quantile(0.9)
    high = pred.observed >= threshold
    (out / "high_demand.json").write_text(
        json.dumps(
            {
                "training_threshold": threshold,
                "n": int(high.sum()),
                **metrics(pred.observed[high], pred.predicted[high]),
            },
            indent=2,
        )
    )
    pred.assign(absolute_error=pred.error.abs()).nlargest(20, "absolute_error").to_csv(
        out / "extreme_errors.csv"
    )
    pred.groupby(pred.index.hour).error.agg(["mean", "std"]).to_csv(
        out / "errors_by_hour.csv"
    )
    pred.groupby(pred.index.day_of_week).error.agg(["mean", "std"]).to_csv(
        out / "errors_by_weekday.csv"
    )
    pred[["observed", "predicted"]].iloc[:288].plot(title=winner)
    plt.tight_layout()
    plt.savefig(out / "predictions.png")
    plt.close()
    pred.error.hist(bins=50)
    plt.savefig(out / "residuals.png")
    plt.close()
    importance = permutation_importance(
        model,
        x.iloc[va].iloc[::10],
        y.iloc[va].iloc[::10],
        n_repeats=3,
        random_state=42,
        scoring="neg_root_mean_squared_error",
    )
    pd.Series(importance.importances_mean, index=x.columns).sort_values().to_csv(
        out / "importance.csv"
    )
    (out / "selection.json").write_text(
        json.dumps(
            {"winner_by_validation": winner, "epochs": epochs, "seed": 42}, indent=2
        )
    )
    print(pd.DataFrame(test).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/tetouan.csv")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--epochs", type=int, default=16)
    args = parser.parse_args()
    run(download(args.data), args.output, args.epochs)
