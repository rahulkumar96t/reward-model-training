from __future__ import annotations

import pytest

from reward_model.probes import (
    FILLER,
    FLATTERY,
    PROBE_NAMES,
    PROBES,
    REFUSAL,
    get_probe,
    shuffle_sentences,
    truncate_half,
)

TEXT = "First point. Second point! Third point? Fourth point."


def test_content_free_probes_only_add_text() -> None:
    assert get_probe("append_filler").apply(TEXT, 0) == f"{TEXT}\n\n{FILLER}"
    assert get_probe("prepend_flattery").apply(TEXT, 0) == f"{FLATTERY} {TEXT}"
    assert get_probe("repeat").apply(TEXT, 0) == f"{TEXT}\n\n{TEXT}"
    assert len(FILLER.split()) == 50


def test_truncate_half_keeps_first_half_of_words() -> None:
    assert truncate_half("a b c d e") == "a b"
    assert truncate_half("single") == "single"


def test_shuffle_is_seeded_and_keeps_sentences() -> None:
    shuffled = shuffle_sentences(TEXT, seed=1)
    assert shuffled == shuffle_sentences(TEXT, seed=1)
    assert shuffled != TEXT
    assert sorted(shuffled.split(" ")) == sorted(TEXT.split(" "))


def test_shuffle_falls_back_to_words_for_one_sentence() -> None:
    out = shuffle_sentences("alpha beta gamma delta", seed=0)
    assert out != "alpha beta gamma delta"
    assert sorted(out.split()) == ["alpha", "beta", "delta", "gamma"]
    assert shuffle_sentences("word", seed=0) == "word"


def test_refusal_replaces_response() -> None:
    assert get_probe("refusal").apply(TEXT, 0) == REFUSAL


def test_probe_registry() -> None:
    assert len(PROBE_NAMES) == len(set(PROBE_NAMES)) == 6
    assert {p.expected for p in PROBES} == {"not up", "down"}
    with pytest.raises(KeyError, match="unknown probe"):
        get_probe("nope")
