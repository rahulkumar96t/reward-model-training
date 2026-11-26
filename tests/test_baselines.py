from __future__ import annotations

from pathlib import Path

import numpy as np
from helpers import make_pair
from scipy import sparse

from reward_model.baselines import (
    LengthScorer,
    RandomScorer,
    fit_bradley_terry,
    load_baselines,
    save_baselines,
    symmetric_design,
    train_baselines,
)
from reward_model.data import PreferencePair
from reward_model.metrics import pairwise_accuracy
from reward_model.scoring import score_pairs


def test_random_scorer_is_deterministic_per_text_and_near_chance(
    splits: dict[str, list[PreferencePair]],
) -> None:
    scorer = RandomScorer(seed=5)
    a = scorer.score(["p", "p"], ["x", "y"])
    b = scorer.score(["p", "p"], ["x", "y"])
    assert np.array_equal(a, b) and a[0] != a[1]
    train = splits["train"]
    acc = pairwise_accuracy(score_pairs(scorer, train).margins)
    assert 0.38 < acc < 0.62


def test_length_scorer_prefers_longer() -> None:
    scores = LengthScorer().score(["p", "p"], ["one two three", "one"])
    assert scores[0] > scores[1]


def test_symmetric_design_doubles_and_negates() -> None:
    diff = sparse.csr_matrix(np.array([[1.0, -2.0], [0.5, 0.0]]))
    x, y = symmetric_design(diff)
    assert x.shape == (4, 2)
    assert np.allclose(x.toarray()[2:], -diff.toarray())
    assert y.tolist() == [1, 1, 0, 0]


def test_bradley_terry_learns_planted_signal(splits: dict[str, list[PreferencePair]]) -> None:
    scorer = fit_bradley_terry(splits["train"], splits["val"], use_tfidf=True, name="bt")
    acc = pairwise_accuracy(score_pairs(scorer, splits["test"]).margins)
    assert acc > 0.9
    assert scorer.c in {row["C"] for row in scorer.tuning}
    top = scorer.top_features(5)
    assert len(top["positive"]) == len(top["negative"]) == 5
    assert all(w > 0 for _, w in top["positive"]) and all(w < 0 for _, w in top["negative"])


def test_bradley_terry_ranks_its_training_pairs() -> None:
    pairs = [
        make_pair("A careful, correct answer with steps.", "No.", prompt=f"q{i}") for i in range(12)
    ] + [make_pair("Sure, a full answer with details.", "Nope.", prompt=f"r{i}") for i in range(12)]
    scorer = fit_bradley_terry(pairs, pairs, use_tfidf=False, name="bt", cs=(1.0,))
    forward = score_pairs(scorer, pairs).margins
    assert np.all(forward > 0)


def test_train_save_load_round_trip(
    splits: dict[str, list[PreferencePair]], tmp_path: Path
) -> None:
    scorers, info = train_baselines(splits["train"], splits["val"], seed=0)
    assert set(scorers) == {"random", "length", "bt_hand", "bt_tfidf"}
    assert info["bt_tfidf"]["n_features"] > info["bt_hand"]["n_features"]
    path = save_baselines({"scorers": scorers, "temperatures": {}}, tmp_path / "b.joblib")
    loaded = load_baselines(path)["scorers"]
    test = splits["test"]
    for name, scorer in scorers.items():
        assert np.allclose(
            score_pairs(scorer, test).margins, score_pairs(loaded[name], test).margins
        )
