from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from helpers import make_pair

from reward_model.baselines import train_baselines
from reward_model.data import PreferencePair
from reward_model.probes import PROBE_NAMES
from reward_model.report import (
    build_report,
    human_reference,
    model_metrics,
    summary_table,
    write_report,
)
from reward_model.scoring import ScoreSet, SplitScores, collect_scores, load_score_sets


@pytest.fixture(scope="module")
def score_sets(splits: dict[str, list[PreferencePair]]) -> list[ScoreSet]:
    scorers, _ = train_baselines(splits["train"], splits["val"], seed=0)
    return [collect_scores(s, splits["val"], splits["test"]) for s in scorers.values()]


def test_score_set_round_trip(score_sets: list[ScoreSet], tmp_path: Path) -> None:
    original = score_sets[0]
    original.save(tmp_path / "scores" / "x.json")
    (loaded,) = load_score_sets(tmp_path)
    assert loaded.model == original.model
    assert set(loaded.probes) == set(PROBE_NAMES)
    assert np.allclose(loaded.test.margins, original.test.margins, atol=1e-5)


def test_report_end_to_end(
    score_sets: list[ScoreSet], splits: dict[str, list[PreferencePair]], tmp_path: Path
) -> None:
    report = build_report(score_sets, splits["test"], {"pairs": {"test": len(splits["test"])}})
    assert list(report["models"]) == ["random", "length", "bt_hand", "bt_tfidf"]
    bt = report["models"]["bt_tfidf"]
    assert bt["accuracy"] > 0.9
    assert bt["accuracy_ci95"][0] <= bt["accuracy"] <= bt["accuracy_ci95"][1]
    assert set(bt["probes"]) == set(PROBE_NAMES)
    length = report["models"]["length"]["length"]
    assert length["chosen_longer"]["accuracy"] == 1.0
    assert length["chosen_shorter"]["accuracy"] in (0.0, None)
    paths = write_report(report, tmp_path)
    assert [p.name for p in paths] == ["metrics.json", "calibration.png", "length_bias.png"]
    for png in paths[1:]:
        assert 0 < png.stat().st_size < 300_000
    assert json.loads(paths[0].read_text())["models"]["bt_tfidf"]["n_test"] == bt["n_test"]
    assert "bt_tfidf" in summary_table(report)


def _scores(chosen: list[float], rejected: list[float], ids: list[str]) -> SplitScores:
    return SplitScores(ids, np.array(chosen), np.array(rejected))


def test_model_metrics_slices_on_hand_made_pairs() -> None:
    pairs = [
        make_pair("long chosen answer here", "short", strength=1, pair_id="a"),
        make_pair("tiny", "a much longer rejected answer", strength=3, pair_id="b"),
        make_pair("same len", "two words", strength=3, votes=(2, 1, 0), pair_id="c"),
    ]
    ids = ["a", "b", "c"]
    test = _scores([2.0, 0.0, 1.0], [1.0, 1.0, 1.0], ids)  # right, wrong, tie
    probe = _scores([3.0, 0.0, 1.0], [2.5, 1.0, 0.5], ids)
    scores = ScoreSet("toy", val=test, test=test, probes={"append_filler": probe})
    m = model_metrics(scores, {p.pair_id: p for p in pairs}, n_boot=200)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["by_strength"]["1"]["accuracy"] == 1.0
    assert m["by_strength"]["3"]["accuracy"] == pytest.approx(0.25)
    assert m["by_agreement"]["contested"] == {"n": 1, "accuracy": 0.5}
    assert m["length"]["chosen_longer"] == {"n": 1, "accuracy": 1.0}
    assert m["length"]["chosen_shorter"] == {"n": 1, "accuracy": 0.0}
    assert m["length"]["same_length"] == {"n": 1, "accuracy": 0.5}
    assert m["length"]["gap"] == 1.0
    filler = m["probes"]["append_filler"]
    # Only pair "a" was ranked right; its edited rejected answer (2.5) beats chosen (2.0).
    assert filler["rejected_wins_after_edit"] == 1.0
    assert filler["chosen_still_wins_after_edit"] == 1.0
    assert filler["frac_up"] == pytest.approx(2 / 6, abs=1e-3)


def test_human_reference() -> None:
    pairs = [
        make_pair("a", "b", votes=(3, 0, 0)),
        make_pair("c", "d", votes=(2, 1, 1)),
    ]
    ref = human_reference(pairs)
    assert ref["votes"] == 7
    assert ref["single_labeller_agreement_excl_ties"] == pytest.approx(5 / 6, abs=1e-4)
    assert ref["single_labeller_agreement_ties_as_half"] == pytest.approx(5.5 / 7, abs=1e-4)
    assert ref["unanimous_pairs"] == 0.5
