"""Score val / test pairs and probe variants with any reward model.

Every model, linear or neural, implements ``RewardScorer.score(prompts, responses)``. The
scores are saved as a ``ScoreSet`` JSON file, and the report is built from those files only.
That keeps the evaluation code free of torch: the neural model writes its score file once
(with the ``train`` extra), and ``reward-model report`` runs in the default install.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .data import PreferencePair
from .probes import PROBES, Probe


class RewardScorer(Protocol):
    name: str

    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> np.ndarray:
        """Scalar reward per (prompt, response); higher means better."""
        ...


@dataclass
class SplitScores:
    pair_ids: list[str]
    chosen: np.ndarray
    rejected: np.ndarray

    @property
    def margins(self) -> np.ndarray:
        return self.chosen - self.rejected

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_ids": self.pair_ids,
            "chosen": [round(float(x), 6) for x in self.chosen],
            "rejected": [round(float(x), 6) for x in self.rejected],
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> SplitScores:
        return cls(
            pair_ids=list(row["pair_ids"]),
            chosen=np.asarray(row["chosen"], dtype=np.float64),
            rejected=np.asarray(row["rejected"], dtype=np.float64),
        )


@dataclass
class ScoreSet:
    model: str
    val: SplitScores
    test: SplitScores
    probes: dict[str, SplitScores] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "meta": self.meta,
            "val": self.val.to_dict(),
            "test": self.test.to_dict(),
            "probes": {name: s.to_dict() for name, s in self.probes.items()},
        }
        path.write_text(json.dumps(payload) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> ScoreSet:
        row = json.loads(path.read_text())
        return cls(
            model=row["model"],
            meta=row.get("meta", {}),
            val=SplitScores.from_dict(row["val"]),
            test=SplitScores.from_dict(row["test"]),
            probes={k: SplitScores.from_dict(v) for k, v in row.get("probes", {}).items()},
        )


def score_pairs(
    scorer: RewardScorer,
    pairs: Sequence[PreferencePair],
    transform: Callable[[str], str] | None = None,
) -> SplitScores:
    """Score chosen and rejected responses of ``pairs`` in one call to the scorer."""
    edit = transform or (lambda text: text)
    prompts = [p.prompt for p in pairs] * 2
    responses = [edit(p.chosen) for p in pairs] + [edit(p.rejected) for p in pairs]
    scores = np.asarray(scorer.score(prompts, responses), dtype=np.float64)
    n = len(pairs)
    return SplitScores([p.pair_id for p in pairs], scores[:n], scores[n:])


def collect_scores(
    scorer: RewardScorer,
    val: Sequence[PreferencePair],
    test: Sequence[PreferencePair],
    probes: Sequence[Probe] = PROBES,
    seed: int = 0,
    meta: dict[str, Any] | None = None,
) -> ScoreSet:
    return ScoreSet(
        model=scorer.name,
        val=score_pairs(scorer, val),
        test=score_pairs(scorer, test),
        probes={
            probe.name: score_pairs(scorer, test, lambda text, p=probe: p.apply(text, seed))
            for probe in probes
        },
        meta=dict(meta or {}),
    )


def scores_path(artifacts: Path, model: str) -> Path:
    return artifacts / "scores" / f"{model}.json"


def load_score_sets(artifacts: Path) -> list[ScoreSet]:
    folder = artifacts / "scores"
    return [ScoreSet.load(p) for p in sorted(folder.glob("*.json"))] if folder.exists() else []
