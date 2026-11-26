"""Small builders shared by the tests."""

from __future__ import annotations

import hashlib

from reward_model.data import PreferencePair


def make_pair(
    chosen: str,
    rejected: str,
    prompt: str = "Explain the topic.",
    strength: int = 2,
    votes: tuple[int, int, int] = (3, 0, 0),
    pair_id: str | None = None,
) -> PreferencePair:
    return PreferencePair(
        pair_id=pair_id
        or hashlib.sha256(f"{prompt}|{chosen}|{rejected}".encode()).hexdigest()[:12],
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        strength=strength,
        votes_chosen=votes[0],
        votes_rejected=votes[1],
        votes_tie=votes[2],
        source_split="train",
    )
