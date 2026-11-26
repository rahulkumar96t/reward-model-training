"""Pairwise metrics for reward models.

Everything works on the reward margin ``m = r(chosen) - r(rejected)`` of each test pair.
Under the Bradley-Terry model the probability that the chosen response wins is
``sigmoid(m / T)``. The model's pick is the response with the higher reward; its confidence
is ``sigmoid(|m| / T)``, which lies in [0.5, 1]. Calibration compares that confidence with how
often the pick agrees with the human label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import optimize, special, stats


def correctness(margins: np.ndarray) -> np.ndarray:
    """1 when the chosen response scores higher, 0 when lower, 0.5 on an exact tie."""
    m = np.asarray(margins, dtype=np.float64)
    return np.where(m > 0, 1.0, np.where(m < 0, 0.0, 0.5))


def pairwise_accuracy(margins: np.ndarray) -> float:
    m = np.asarray(margins)
    return float(correctness(m).mean()) if m.size else float("nan")


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return float(lo), float(hi)


def pairwise_nll(margins: np.ndarray, temperature: float = 1.0) -> float:
    """Mean of -log sigmoid(m / T), the Bradley-Terry loss on held-out pairs."""
    m = np.asarray(margins, dtype=np.float64) / temperature
    return float(np.logaddexp(0.0, -m).mean())


def fit_temperature(margins: np.ndarray) -> float:
    """Temperature T > 0 that minimises the held-out Bradley-Terry loss of ``m / T``."""
    m = np.asarray(margins, dtype=np.float64)
    if m.size == 0 or not np.any(m):
        return 1.0
    result = optimize.minimize_scalar(
        lambda log_t: pairwise_nll(m, float(np.exp(log_t))),
        bounds=(-12.0, 12.0),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(np.exp(result.x))


def confidence(margins: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return special.expit(np.abs(np.asarray(margins, dtype=np.float64)) / temperature)


@dataclass(frozen=True)
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def reliability(
    margins: np.ndarray, temperature: float = 1.0, n_bins: int = 10
) -> list[ReliabilityBin]:
    """Equal-width confidence bins over [0.5, 1]; empty bins are omitted."""
    conf = confidence(margins, temperature)
    correct = correctness(margins)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    index = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = index == b
        if mask.any():
            bins.append(
                ReliabilityBin(
                    lo=float(edges[b]),
                    hi=float(edges[b + 1]),
                    count=int(mask.sum()),
                    mean_confidence=float(conf[mask].mean()),
                    accuracy=float(correct[mask].mean()),
                )
            )
    return bins


def expected_calibration_error(
    margins: np.ndarray, temperature: float = 1.0, n_bins: int = 10
) -> float:
    bins = reliability(margins, temperature, n_bins)
    total = sum(b.count for b in bins)
    if total == 0:
        return float("nan")
    return float(sum(b.count * abs(b.accuracy - b.mean_confidence) for b in bins) / total)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)
