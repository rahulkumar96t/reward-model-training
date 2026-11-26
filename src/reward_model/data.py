"""Download, verify and prepare the HelpSteer2 preference pairs.

The source file is pinned to one commit of the Hugging Face dataset repository and checked
against a SHA-256 digest. ``prepare`` turns the raw rows into oriented (chosen, rejected)
pairs, drops ties and duplicates, and assigns a fixed train / val / test split by hashing the
prompt, so every pair of the same prompt lands in the same split.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DATASET = "nvidia/HelpSteer2"
REVISION = "990b2711a36180dd19d9c94b8627844866f8982a"
SOURCE_FILE = "preference/preference.jsonl.gz"
URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{SOURCE_FILE}"
SHA256 = "a5cd48600fb7a330cf0ccc8f59051e24e8f236907c379f42eff1ba18da55204b"
LICENSE = "CC-BY-4.0"
RAW_NAME = "helpsteer2_preference.jsonl.gz"
SPLITS = ("train", "val", "test")
# Buckets 0-79 train, 80-89 val, 90-99 test (bucket = prompt hash mod 100).
SPLIT_UPPER = {"train": 80, "val": 90, "test": 100}

# Multi-turn prompts mark later turns with this token followed by the speaker name.
_TURN = re.compile(r"\s*<extra_id_1>(User|Assistant)[ \t]*\n?")
_ROLE = {"User": "User", "Assistant": "Model"}
_BLANK_RUN = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
_SPACE = re.compile(r"\s+")


class ChecksumError(RuntimeError):
    """The downloaded file does not match the pinned SHA-256 digest."""


@dataclass(frozen=True)
class PreferencePair:
    """One human preference judgement, oriented so that ``chosen`` won."""

    pair_id: str
    prompt: str
    chosen: str
    rejected: str
    strength: int  # 1 = slightly better, 2 = better, 3 = much better
    votes_chosen: int  # individual labellers who preferred the chosen response
    votes_rejected: int  # individual labellers who preferred the rejected response
    votes_tie: int  # individual labellers who rated them about the same
    source_split: str  # "train" or "val" in the original release

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> PreferencePair:
        return cls(**row)


@dataclass
class PrepareSummary:
    raw_rows: int
    dropped_ties: int
    dropped_identical: int
    dropped_empty: int
    dropped_duplicates: int
    pairs: dict[str, int]
    prompts: dict[str, int]
    strength: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(data_dir: Path, url: str = URL, sha256: str = SHA256, timeout: float = 120.0) -> Path:
    """Fetch the raw file into ``data_dir`` unless a verified copy is already there.

    The file is written to a temporary name and only renamed after the checksum matches, so
    a failed or tampered download never leaves a file that looks valid.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / RAW_NAME
    if target.exists():
        if sha256_file(target) == sha256:
            return target
        target.unlink()
    fd, tmp_name = tempfile.mkstemp(dir=data_dir, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=timeout) as resp:
            shutil.copyfileobj(resp, out, 1 << 20)
        actual = sha256_file(tmp)
        if actual != sha256:
            raise ChecksumError(f"SHA-256 mismatch for {url}: expected {sha256}, got {actual}")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    return _BLANK_RUN.sub("\n\n", text)


def render_prompt(raw: str) -> str:
    """Render a single- or multi-turn prompt as plain text.

    Single-turn prompts are returned cleaned. Multi-turn prompts become ``User:`` /
    ``Model:`` blocks; the last turn is always the user's, and the response under
    evaluation answers it.
    """
    parts = _TURN.split(raw)
    if len(parts) == 1:
        return clean_text(raw)
    turns = [("User", parts[0]), *zip(parts[1::2], parts[2::2], strict=True)]
    blocks = [f"{_ROLE[role]}: {clean_text(text)}" for role, text in turns if text.strip()]
    return "\n\n".join(blocks)


def _norm(text: str) -> str:
    return _SPACE.sub(" ", text).strip().lower()


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def split_for_prompt(prompt: str) -> str:
    bucket = int(_digest(_norm(prompt))[:8], 16) % 100
    for split in SPLITS:
        if bucket < SPLIT_UPPER[split]:
            return split
    raise AssertionError("unreachable")


def _votes(row: dict[str, Any], sign: int) -> tuple[int, int, int]:
    agree = disagree = tie = 0
    for vote in row.get("all_preferences_unprocessed") or []:
        value = int(vote["strength"])
        if value == 0:
            tie += 1
        elif (value > 0) == (sign > 0):
            agree += 1
        else:
            disagree += 1
    return agree, disagree, tie


def read_raw(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def build_pairs(rows: Iterable[dict[str, Any]]) -> tuple[list[PreferencePair], dict[str, int]]:
    """Orient, filter and deduplicate raw rows. Returns pairs and drop counts."""
    drops = Counter[str]()
    seen: set[str] = set()
    pairs: list[PreferencePair] = []
    raw = 0
    for row in rows:
        raw += 1
        strength = int(row["preference_strength"])
        prompt = render_prompt(row["prompt"])
        first, second = clean_text(row["response_1"]), clean_text(row["response_2"])
        if strength == 0:
            drops["ties"] += 1
            continue
        if not prompt or not first or not second:
            drops["empty"] += 1
            continue
        if _norm(first) == _norm(second):
            drops["identical"] += 1
            continue
        key = _digest(_norm(prompt), *sorted((_norm(first), _norm(second))))
        if key in seen:
            drops["duplicates"] += 1
            continue
        seen.add(key)
        chosen, rejected = (second, first) if strength > 0 else (first, second)
        agree, disagree, tie = _votes(row, strength)
        pairs.append(
            PreferencePair(
                pair_id=key[:12],
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                strength=abs(strength),
                votes_chosen=agree,
                votes_rejected=disagree,
                votes_tie=tie,
                source_split=str(row.get("split", "")),
            )
        )
    drops["raw"] = raw
    return pairs, dict(drops)


def pairs_path(data_dir: Path, split: str) -> Path:
    return data_dir / "pairs" / f"{split}.jsonl"


def summary_path(data_dir: Path) -> Path:
    return data_dir / "pairs" / "summary.json"


def prepare(raw_path: Path, data_dir: Path) -> PrepareSummary:
    """Write ``data/pairs/{train,val,test}.jsonl`` and ``summary.json``."""
    pairs, drops = build_pairs(read_raw(raw_path))
    by_split: dict[str, list[PreferencePair]] = {split: [] for split in SPLITS}
    for pair in pairs:
        by_split[split_for_prompt(pair.prompt)].append(pair)
    (data_dir / "pairs").mkdir(parents=True, exist_ok=True)
    for split, items in by_split.items():
        with open(pairs_path(data_dir, split), "w", encoding="utf-8") as fh:
            for pair in items:
                fh.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
    summary = PrepareSummary(
        raw_rows=drops.get("raw", 0),
        dropped_ties=drops.get("ties", 0),
        dropped_identical=drops.get("identical", 0),
        dropped_empty=drops.get("empty", 0),
        dropped_duplicates=drops.get("duplicates", 0),
        pairs={split: len(items) for split, items in by_split.items()},
        prompts={split: len({p.prompt for p in items}) for split, items in by_split.items()},
        strength={
            split: {str(k): v for k, v in sorted(Counter(p.strength for p in items).items())}
            for split, items in by_split.items()
        },
    )
    summary_path(data_dir).write_text(json.dumps(summary.to_dict(), indent=2) + "\n")
    return summary


def load_pairs(data_dir: Path, split: str) -> list[PreferencePair]:
    path = pairs_path(data_dir, split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `reward-model download` first")
    with open(path, encoding="utf-8") as fh:
        return [PreferencePair.from_dict(json.loads(line)) for line in fh if line.strip()]


def load_summary(data_dir: Path) -> dict[str, Any]:
    path = summary_path(data_dir)
    return json.loads(path.read_text()) if path.exists() else {}


def sample_pairs(pairs: list[PreferencePair], limit: int, seed: int) -> list[PreferencePair]:
    """Deterministic subset: the ``limit`` pairs with the smallest seeded hash."""
    if limit <= 0 or limit >= len(pairs):
        return list(pairs)
    ranked = sorted(pairs, key=lambda p: _digest(str(seed), p.pair_id))
    keep = {p.pair_id for p in ranked[:limit]}
    return [p for p in pairs if p.pair_id in keep]
