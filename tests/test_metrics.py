from __future__ import annotations

import math

import numpy as np
import pytest

from reward_model.metrics import (
    bootstrap_ci,
    confidence,
    correctness,
    expected_calibration_error,
    fit_temperature,
    pairwise_accuracy,
    pairwise_nll,
    reliability,
    spearman,
)


def _bradley_terry_margins(n: int, temperature: float, seed: int = 0) -> np.ndarray:
    """Margins whose sign follows a Bradley-Terry draw with the given temperature."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(0.0, 2.0, size=n)
    chosen_wins = rng.random(n) < 1.0 / (1.0 + np.exp(-scores / temperature))
    return np.where(chosen_wins, scores, -scores)


def test_correctness_counts_ties_as_half() -> None:
    assert correctness(np.array([1.0, -2.0, 0.0])).tolist() == [1.0, 0.0, 0.5]
    assert pairwise_accuracy(np.array([1.0, -2.0, 0.0, 3.0])) == pytest.approx(0.625)
    assert math.isnan(pairwise_accuracy(np.array([])))


def test_bootstrap_interval_contains_mean_and_shrinks_with_n() -> None:
    rng = np.random.default_rng(1)
    small = (rng.random(100) < 0.6).astype(float)
    large = (rng.random(10_000) < 0.6).astype(float)
    lo, hi = bootstrap_ci(small, seed=0)
    assert lo < small.mean() < hi
    lo2, hi2 = bootstrap_ci(large, seed=0)
    assert hi2 - lo2 < (hi - lo) / 5


def test_nll_of_zero_margins_is_log_two() -> None:
    assert pairwise_nll(np.zeros(5)) == pytest.approx(math.log(2))
    assert pairwise_nll(np.array([1000.0])) == pytest.approx(0.0, abs=1e-9)
    assert pairwise_nll(np.array([-1000.0])) == pytest.approx(1000.0)


@pytest.mark.parametrize("true_t", [0.5, 1.0, 3.0])
def test_fit_temperature_recovers_scale(true_t: float) -> None:
    margins = _bradley_terry_margins(40_000, true_t, seed=2)
    assert fit_temperature(margins) == pytest.approx(true_t, rel=0.08)


def test_fit_temperature_degenerate_input() -> None:
    assert fit_temperature(np.zeros(4)) == 1.0
    assert fit_temperature(np.array([])) == 1.0


def test_calibrated_scores_have_small_ece_and_overconfident_large() -> None:
    margins = _bradley_terry_margins(40_000, 1.0, seed=3)
    assert expected_calibration_error(margins, 1.0) < 0.02
    assert expected_calibration_error(margins * 5.0, 1.0) > 0.1
    # Temperature scaling repairs the over-confident version.
    t = fit_temperature(margins * 5.0)
    assert expected_calibration_error(margins * 5.0, t) < 0.02


def test_reliability_bins_cover_all_pairs() -> None:
    margins = np.array([0.0, 0.1, -0.2, 2.0, 5.0, -5.0])
    bins = reliability(margins, 1.0, n_bins=5)
    assert sum(b.count for b in bins) == margins.size
    assert all(0.5 <= b.mean_confidence <= 1.0 for b in bins)
    top = bins[-1]
    assert top.hi == 1.0 and top.count == 2 and top.accuracy == 0.5
    assert confidence(np.array([0.0]))[0] == pytest.approx(0.5)


def test_spearman() -> None:
    a = np.arange(10.0)
    assert spearman(a, a**3) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    assert math.isnan(spearman(a, np.ones(10)))
