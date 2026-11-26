"""Two report charts: a reliability diagram and a length-bias / probe chart."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e1e0d9"
# Fixed categorical order (blue, orange, aqua, yellow); checked for colour-vision deficiency.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SHORT = {
    "random": "random",
    "length": "longer wins",
    "bt_hand": "BT hand",
    "bt_tfidf": "BT TF-IDF",
    "neural": "DistilRoBERTa",
}
PLOTTED = ("length", "bt_hand", "bt_tfidf", "neural")
# The length rule's margin is a word-count difference, so its reliability curve says little.
CALIBRATED = ("bt_hand", "bt_tfidf", "neural")


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _models(report: dict[str, Any], names: tuple[str, ...] = PLOTTED) -> list[str]:
    return [m for m in names if m in report["models"]]


def plot_calibration(report: dict[str, Any], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=110, facecolor=SURFACE)
    _style(ax)
    ax.plot([0.5, 1.0], [0.5, 1.0], color=TEXT_MUTED, linewidth=1, label="perfect calibration")
    for i, name in enumerate(_models(report, CALIBRATED)):
        cal = report["models"][name]["calibration"]
        bins = [b for b in cal["bins"] if b["count"] >= 10]
        if not bins:
            continue
        ax.plot(
            [b["mean_confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            color=SERIES[i % len(SERIES)],
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=f"{SHORT[name]} (ECE {cal['ece']:.3f}, T {cal['temperature']:.2f})",
        )
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("confidence = sigmoid(|margin| / T), T fitted on val", color=TEXT, fontsize=9)
    ax.set_ylabel("fraction of test pairs ranked like the labellers", color=TEXT, fontsize=9)
    ax.set_title(
        "Reliability on the test split (bins with at least 10 pairs)",
        color=TEXT,
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def _bar_labels(ax: Axes, bars: Any, fmt: str) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=TEXT_MUTED,
        )


def plot_length_and_probes(report: dict[str, Any], path: Path) -> Path:
    models = _models(report)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=110, facecolor=SURFACE)
    for ax in (left, right):
        _style(ax)
    xs = list(range(len(models)))
    labels = [SHORT[m] for m in models]

    width = 0.24
    for j, (key, label) in enumerate(
        (("chosen_longer", "chosen answer is longer"), ("chosen_shorter", "chosen is shorter"))
    ):
        values = [report["models"][m]["length"][key]["accuracy"] or 0.0 for m in models]
        offset = (j - 0.5) * width
        bars = left.bar(
            [x + offset for x in xs],
            values,
            width=width - 0.04,
            color=SERIES[j],
            label=label,
        )
        _bar_labels(left, bars, "{:.2f}")
    left.axhline(0.5, color=TEXT_MUTED, linewidth=1, zorder=0.5)
    left.set_xticks(xs, labels, color=TEXT, fontsize=9)
    left.set_ylim(0, 1.1)
    left.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    left.set_ylabel("test accuracy", color=TEXT, fontsize=9)
    left.set_title(
        "Accuracy split by which answer is longer (line = chance)",
        color=TEXT,
        fontsize=10,
        loc="left",
    )
    left.legend(
        frameon=False,
        fontsize=8,
        labelcolor=TEXT,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncols=2,
    )

    probe_names = [n for n, p in report["probes"].items() if p["expected"] == "not up"]
    width = 0.66 / max(1, len(probe_names))
    for j, probe in enumerate(probe_names):
        values = [
            report["models"][m]["probes"].get(probe, {}).get("rejected_wins_after_edit") or 0.0
            for m in models
        ]
        offset = (j - (len(probe_names) - 1) / 2) * width
        bars = right.bar(
            [x + offset for x in xs],
            values,
            width=width - 0.04,
            color=SERIES[j % len(SERIES)],
            label=probe.replace("_", " "),
        )
        _bar_labels(right, bars, "{:.2f}")
    right.set_xticks(xs, labels, color=TEXT, fontsize=9)
    right.set_ylim(0, 1.1)
    right.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    right.set_ylabel("share of correctly ranked pairs that flip", color=TEXT, fontsize=9)
    right.set_title(
        "Edit the rejected answer only: does it now win?", color=TEXT, fontsize=10, loc="left"
    )
    right.legend(
        frameon=False,
        fontsize=8,
        labelcolor=TEXT,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncols=3,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
