"""Report figures (matplotlib, Agg). I/O only: callers pass data, these write PNGs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure

BASE_LABELS = ["___", "1__", "_2_", "12_", "__3", "1_3", "_23", "123"]
COUNT_LABELS = [f"{b}-{s}" for b in range(4) for s in range(3)]


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def heatmap(
    a: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    path: Path,
    fmt: str = "{:.2f}",
    cmap: str = "viridis",
    center: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(1 + 0.55 * len(xlabels), 1 + 0.4 * len(ylabels)))
    vmin = vmax = None
    if center is not None:
        m = float(np.nanmax(np.abs(a - center)))
        vmin, vmax = center - m, center + m
    im = ax.imshow(a, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    for (i, j), v in np.ndenumerate(a):
        if np.isfinite(v):
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7, color="w")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    _save(fig, path)


def reliability_plot(curves: dict[str, pl.DataFrame], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    for name, r in curves.items():
        ax.plot(r["p_mean"], r["y_rate"], marker="o", label=name)
    ax.set_xlabel("predicted")
    ax.set_ylabel("observed")
    ax.set_title(title)
    ax.legend()
    _save(fig, path)


def lines(
    x: np.ndarray, ys: dict[str, np.ndarray], xlabel: str, ylabel: str, title: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, y in ys.items():
        ax.plot(x, y, label=k)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)
    _save(fig, path)
