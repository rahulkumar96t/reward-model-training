"""Reward baselines that need no GPU: random, longer-wins, and linear Bradley-Terry models.

The linear models learn a reward ``r(x) = w . f(x)``. Under Bradley-Terry,
``P(chosen beats rejected) = sigmoid(r(chosen) - r(rejected)) = sigmoid(w . (f_c - f_r))``,
which is a logistic regression without intercept on feature differences. Each pair is used
in both orientations (difference with label 1, negated difference with label 0) so the
model cannot learn a side preference.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from .data import PreferencePair
from .features import ResponseFeaturizer, word_count
from .metrics import pairwise_accuracy, pairwise_nll

DEFAULT_CS: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


@dataclass
class RandomScorer:
    """A fixed pseudo-random reward per text: about 50 % pairwise accuracy by construction."""

    seed: int = 0
    name: str = "random"

    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> np.ndarray:
        out = np.empty(len(responses))
        for i, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
            h = hashlib.sha256(f"{self.seed}\x1f{prompt}\x1f{response}".encode()).digest()
            out[i] = np.random.default_rng(int.from_bytes(h[:8], "little")).standard_normal()
        return out


@dataclass
class LengthScorer:
    """Longer answer wins: reward = log(1 + number of words)."""

    name: str = "length"

    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> np.ndarray:
        return np.asarray([math.log1p(word_count(r)) for r in responses], dtype=np.float64)


@dataclass
class BradleyTerryScorer:
    featurizer: ResponseFeaturizer
    weights: np.ndarray
    name: str
    c: float
    tuning: list[dict[str, float]] = field(default_factory=list)

    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> np.ndarray:
        return np.asarray(self.featurizer.transform(prompts, responses) @ self.weights).ravel()

    def top_features(self, k: int = 10) -> dict[str, list[tuple[str, float]]]:
        """The ``k`` largest positive and negative weights."""
        names = self.featurizer.feature_names
        order = np.argsort(self.weights)
        w = self.weights
        positive = [(names[i], round(float(w[i]), 4)) for i in order[::-1][:k] if w[i] > 0]
        negative = [(names[i], round(float(w[i]), 4)) for i in order[:k] if w[i] < 0]
        return {"positive": positive, "negative": negative}


def _difference(featurizer: ResponseFeaturizer, pairs: Sequence[PreferencePair]):
    prompts = [p.prompt for p in pairs]
    chosen = featurizer.transform(prompts, [p.chosen for p in pairs])
    rejected = featurizer.transform(prompts, [p.rejected for p in pairs])
    return chosen - rejected


def symmetric_design(diff: sparse.spmatrix) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Stack each difference with its negation; labels 1 and 0."""
    n = diff.shape[0]
    x = sparse.vstack([diff, -diff], format="csr")
    y = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)])
    return x, y


def fit_bradley_terry(
    train: Sequence[PreferencePair],
    val: Sequence[PreferencePair],
    use_tfidf: bool,
    name: str,
    cs: Sequence[float] = DEFAULT_CS,
) -> BradleyTerryScorer:
    """Fit on ``train``; choose C by validation accuracy (ties broken by validation loss)."""
    featurizer = ResponseFeaturizer(use_tfidf=use_tfidf)
    featurizer.fit(
        [p.prompt for p in train] * 2, [p.chosen for p in train] + [p.rejected for p in train]
    )
    x, y = symmetric_design(_difference(featurizer, train))
    val_diff = _difference(featurizer, val)
    tuning: list[dict[str, float]] = []
    best: tuple[float, float, float, np.ndarray] | None = None
    for c in cs:
        model = LogisticRegression(C=c, fit_intercept=False, max_iter=5000)
        model.fit(x, y)
        w = model.coef_.ravel().copy()
        margins = np.asarray(val_diff @ w).ravel()
        acc, nll = pairwise_accuracy(margins), pairwise_nll(margins)
        tuning.append({"C": c, "val_accuracy": round(acc, 4), "val_nll": round(nll, 4)})
        if best is None or (acc, -nll) > (best[0], -best[1]):
            best = (acc, nll, c, w)
    assert best is not None
    return BradleyTerryScorer(featurizer, best[3], name=name, c=best[2], tuning=tuning)


def train_baselines(
    train: Sequence[PreferencePair], val: Sequence[PreferencePair], seed: int = 0
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Fit all baselines. Returns the scorers and per-model training info."""
    scorers: dict[str, Any] = {"random": RandomScorer(seed=seed), "length": LengthScorer()}
    info: dict[str, dict[str, Any]] = {"random": {}, "length": {}}
    for name, use_tfidf in (("bt_hand", False), ("bt_tfidf", True)):
        start = time.perf_counter()
        scorer = fit_bradley_terry(train, val, use_tfidf=use_tfidf, name=name)
        scorers[name] = scorer
        info[name] = {
            "train_seconds": round(time.perf_counter() - start, 2),
            "C": scorer.c,
            "n_features": len(scorer.weights),
            "tuning": scorer.tuning,
            "top_features": scorer.top_features(8),
        }
    return scorers, info


def save_baselines(scorers: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scorers, path)
    return path


def load_baselines(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `reward-model baselines` first")
    return joblib.load(path)
