"""Seeded synthetic preference rows in the raw HelpSteer2 format, for tests only.

What is planted: the preferred response stays on topic (it uses the topic's keywords, is a
numbered list and ends cleanly); the other one is generic filler, repetitive, cut off, or a
refusal. ``p_longer`` sets how often the preferred response is also the longer one, and
``label_noise`` flips that share of labels. A few rows exercise the cleaning rules: a tie,
two identical responses, an exact duplicate and multi-turn prompts.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Any

TOPICS: dict[str, list[str]] = {
    "sourdough bread": ["starter", "flour", "hydration", "proof", "oven", "crust"],
    "solar panels": ["inverter", "sunlight", "battery", "roof", "efficiency", "watts"],
    "python decorators": ["function", "wrapper", "closure", "arguments", "syntax", "return"],
    "marathon training": ["mileage", "tempo", "recovery", "pace", "shoes", "taper"],
    "compound interest": ["principal", "rate", "years", "growth", "savings", "returns"],
    "houseplant care": ["watering", "light", "soil", "drainage", "humidity", "repotting"],
    "git branching": ["commit", "merge", "branch", "rebase", "conflict", "history"],
    "public speaking": ["audience", "practice", "slides", "pauses", "nerves", "structure"],
}
FILLER = ["thing", "stuff", "generally", "basically", "various", "aspects", "really", "overall"]
ASKS = ["Explain {t} to a beginner.", "Give me practical tips on {t}.", "How does {t} work?"]


def good_response(topic: str, rng: random.Random, points: int) -> str:
    words = TOPICS[topic]
    lines = [f"Here is a short guide to {topic}:"]
    for i in range(points):
        a, b = rng.sample(words, 2)
        lines.append(f"{i + 1}. Check the {a} first, then adjust the {b} carefully.")
    lines.append(f"With these steps, {topic} becomes much easier.")
    return "\n".join(lines)


def bad_response(rng: random.Random, sentences: int) -> str:
    kind = rng.choice(["generic", "repetitive", "cut", "refusal"])
    if kind == "refusal":
        return "I'm sorry, but I can't help with that."
    if kind == "repetitive":
        return " ".join(["It depends on the situation really."] * sentences)
    text = " ".join(
        f"It is {rng.choice(FILLER)} about {rng.choice(FILLER)} {rng.choice(FILLER)} things."
        for _ in range(sentences)
    )
    return text[: max(10, len(text) * 2 // 3)] if kind == "cut" else text


def make_rows(
    n: int = 240, seed: int = 0, p_longer: float = 0.6, label_noise: float = 0.0
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    topics = sorted(TOPICS)
    for i in range(n):
        topic = topics[i % len(topics)]
        prompt = rng.choice(ASKS).format(t=topic) + f" (case {i})"
        if i % 7 == 3:
            prompt = (
                f"I want to learn about {topic}.\n<extra_id_1>Assistant\nSure, what do you "
                f"want to know?\n<extra_id_1>User\n{prompt}"
            )
        longer = rng.random() < p_longer
        good = good_response(topic, rng, points=6 if longer else 2)
        bad = bad_response(rng, sentences=2 if longer else 9)
        good_first = rng.random() < 0.5
        magnitude = rng.choice([1, 2, 3])
        sign = -1 if good_first else 1  # negative strength = response_1 preferred
        if rng.random() < label_noise:
            sign = -sign
        votes = [{"strength": sign * magnitude, "justification": ""} for _ in range(3)]
        if i % 5 == 0:
            votes[0] = {"strength": -sign, "justification": ""}
        rows.append(
            {
                "split": "val" if i % 10 == 9 else "train",
                "prompt": prompt,
                "response_1": good if good_first else bad,
                "response_2": bad if good_first else good,
                "preference_strength": sign * magnitude,
                "all_preferences_unprocessed": votes,
            }
        )
    tie = dict(rows[0], prompt="A tied prompt.", preference_strength=0)
    same = dict(rows[1], prompt="Same answers twice.", response_2=rows[1]["response_1"])
    rows.extend([tie, same, dict(rows[2])])  # the last one is an exact duplicate
    return rows


def write_raw(path: Path, rows: list[dict[str, Any]]) -> str:
    """Write rows as gzipped JSON lines; return the file's SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()
