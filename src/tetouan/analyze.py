"""Training-only diagnostics for the expanded causal feature set."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import read, splits, supervised


def analyze(path, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frame = read(path)
    x, y = supervised(frame)
    tr, _, _ = splits(len(x))
    x, y = x.iloc[tr], y.iloc[tr]
    summary = x.describe().T
    summary["skewness"] = x.skew()
    summary["missing"] = x.isna().sum()
    summary["unique"] = x.nunique()
    summary.to_csv(out / "feature_statistics.csv")
    corr = x.corr()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(mask).stack().rename("correlation")
    pairs[pairs.abs() > 0.95].to_csv(out / "redundant_pairs.csv")
    x.corrwith(y).sort_values().to_csv(out / "future_target_correlations.csv")
    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=5)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=5)
    ax.set_title("Correlaciones de predictores — solo entrenamiento")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "feature_correlations.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.flat, frame.columns[:6], strict=True):
        ax.hist(frame.loc[x.index, col], bins=40)
        ax.set_title(col, fontsize=9)
    fig.suptitle("Distribuciones — solo entrenamiento")
    fig.tight_layout()
    fig.savefig(out / "distributions.png", dpi=150)
    plt.close(fig)
    dictionary = pd.DataFrame({"feature": x.columns})
    dictionary["availability"] = (
        "Observed at t or calculated from observations at/before t"
    )
    dictionary.to_csv(out / "feature_dictionary.csv", index=False)
    print(f"Analyzed {len(x.columns)} features on {len(x)} training rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/tetouan.csv")
    parser.add_argument("--output", default="artifacts/expanded")
    args = parser.parse_args()
    analyze(args.data, args.output)
