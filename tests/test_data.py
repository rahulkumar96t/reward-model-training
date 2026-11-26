from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from synthetic import make_rows, write_raw

from reward_model.data import (
    RAW_NAME,
    SPLITS,
    ChecksumError,
    build_pairs,
    download,
    load_pairs,
    load_summary,
    prepare,
    render_prompt,
    sample_pairs,
    sha256_file,
    split_for_prompt,
)


def _row(strength: int, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "split": "train",
        "prompt": "How do I boil an egg?",
        "response_1": "Put it in boiling water for nine minutes.",
        "response_2": "Eggs are food.",
        "preference_strength": strength,
        "all_preferences_unprocessed": [
            {"strength": strength},
            {"strength": strength},
            {"strength": 0},
            {"strength": -strength},
        ],
    }
    row.update(extra)
    return row


def test_render_single_turn_prompt_is_cleaned() -> None:
    assert render_prompt("  Hello\r\nworld \n\n\n\n end ") == "Hello\nworld \n\n end"


def test_render_multi_turn_prompt_labels_turns() -> None:
    raw = "First question\n<extra_id_1>Assistant\nAn answer\n<extra_id_1>User\nFollow-up"
    assert render_prompt(raw) == "User: First question\n\nModel: An answer\n\nUser: Follow-up"


def test_negative_strength_means_first_response_wins() -> None:
    pairs, _ = build_pairs([_row(-2)])
    assert pairs[0].chosen.startswith("Put it in boiling water")
    assert pairs[0].strength == 2


def test_positive_strength_means_second_response_wins() -> None:
    pairs, _ = build_pairs([_row(3)])
    assert pairs[0].chosen == "Eggs are food."
    assert pairs[0].strength == 3


def test_votes_are_counted_against_the_aggregate() -> None:
    pairs, _ = build_pairs([_row(1)])
    pair = pairs[0]
    assert (pair.votes_chosen, pair.votes_rejected, pair.votes_tie) == (2, 1, 1)


def test_ties_identical_and_duplicates_are_dropped() -> None:
    rows = [
        _row(0),
        _row(1, response_2="Put it in  boiling water for nine minutes."),
        _row(2),
        _row(
            -1, response_1="Eggs are food.", response_2="Put it in boiling water for nine minutes."
        ),
    ]
    pairs, drops = build_pairs(rows)
    assert len(pairs) == 1
    assert drops == {"ties": 1, "identical": 1, "duplicates": 1, "raw": 4}


def test_split_assignment_is_deterministic_and_whitespace_insensitive() -> None:
    assert split_for_prompt("What is a monad?") == split_for_prompt("What  is a\nmonad? ")
    seen = {split_for_prompt(f"prompt {i}") for i in range(300)}
    assert seen == set(SPLITS)


def test_prepare_writes_disjoint_splits(raw_file: tuple[Path, str], tmp_path: Path) -> None:
    summary = prepare(raw_file[0], tmp_path)
    loaded = {split: load_pairs(tmp_path, split) for split in SPLITS}
    assert {s: len(p) for s, p in loaded.items()} == summary.pairs
    assert summary.dropped_ties == 1
    assert summary.dropped_identical == 1
    assert summary.dropped_duplicates == 1
    prompts = {s: {p.prompt for p in items} for s, items in loaded.items()}
    assert not prompts["train"] & prompts["val"]
    assert not prompts["train"] & prompts["test"]
    assert not prompts["val"] & prompts["test"]
    assert load_summary(tmp_path)["pairs"] == summary.pairs
    multi = [p for p in loaded["train"] if p.prompt.startswith("User: ")]
    assert multi and "<extra_id_1>" not in multi[0].prompt


def test_download_verifies_checksum(raw_file: tuple[Path, str], tmp_path: Path) -> None:
    path, digest = raw_file
    target = download(tmp_path, url=path.as_uri(), sha256=digest)
    assert target == tmp_path / RAW_NAME
    assert sha256_file(target) == digest


def test_download_mismatch_leaves_no_file(raw_file: tuple[Path, str], tmp_path: Path) -> None:
    with pytest.raises(ChecksumError):
        download(tmp_path, url=raw_file[0].as_uri(), sha256="0" * 64)
    assert list(tmp_path.iterdir()) == []


def test_download_reuses_verified_file_and_replaces_corrupt_one(
    raw_file: tuple[Path, str], tmp_path: Path
) -> None:
    path, digest = raw_file
    download(tmp_path, url=path.as_uri(), sha256=digest)
    # A verified copy is reused without touching the network (the URL does not exist).
    assert download(tmp_path, url=(tmp_path / "missing").as_uri(), sha256=digest).exists()
    (tmp_path / RAW_NAME).write_bytes(b"corrupt")
    download(tmp_path, url=path.as_uri(), sha256=digest)
    assert sha256_file(tmp_path / RAW_NAME) == digest


def test_load_pairs_without_data_explains_what_to_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="reward-model download"):
        load_pairs(tmp_path, "train")


def test_sample_pairs_is_seeded_subset(splits: dict[str, list]) -> None:
    train = splits["train"]
    a = sample_pairs(train, 50, seed=3)
    b = sample_pairs(train, 50, seed=3)
    c = sample_pairs(train, 50, seed=4)
    assert len(a) == 50
    assert [p.pair_id for p in a] == [p.pair_id for p in b]
    assert {p.pair_id for p in a} != {p.pair_id for p in c}
    assert sample_pairs(train, 0, seed=3) == train


def test_synthetic_raw_file_round_trips(tmp_path: Path) -> None:
    rows = make_rows(n=10, seed=2)
    write_raw(tmp_path / "x.jsonl.gz", rows)
    with gzip.open(tmp_path / "x.jsonl.gz", "rt") as fh:
        assert [json.loads(line) for line in fh] == rows
