"""Optional expanding-window Ridge tuning, never touching final test."""

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import read, splits, supervised
from .train import metrics


def tune(path, output, trials=12):
    x, y = supervised(read(path))
    tr, _, _ = splits(len(x))
    x, y = x.iloc[tr], y.iloc[tr]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.001, 10000, log=True)
        scores = []
        for a, b in TimeSeriesSplit(n_splits=3, gap=6).split(x):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x.iloc[a], y.iloc[a])
            scores.append(metrics(y.iloc[b], model.predict(x.iloc[b]))["rmse"])
        trial.set_user_attr("window_rmse", scores)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=trials)
    study.trials_dataframe().to_csv(out / "tuning.csv", index=False)
    (out / "tuning_best.json").write_text(json.dumps(study.best_params))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/tetouan.csv")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    tune(args.data, args.output)
