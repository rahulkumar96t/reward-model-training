"""Turn score files into ``reports/metrics.json`` and two charts.

For every model on the held-out test split:

- pairwise accuracy with a bootstrap 95 % interval;
- calibration: a temperature fitted on val, ECE and loss before and after it, and the
  reliability bins;
- accuracy by labeller-reported preference strength and by labeller agreement;
- length bias: accuracy when the chosen response is longer vs shorter, and the rank
  correlation between reward and response length;
- probes: how rewards move when responses are padded, flattered, repeated or damaged.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .data import PreferencePair
from .features import word_count
from .metrics import (
    bootstrap_ci,
    correctness,
    expected_calibration_error,
    fit_temperature,
    pairwise_nll,
    reliability,
    spearman,
)
from .probes import PROBES
from .scoring import ScoreSet

MODEL_ORDER = ("random", "length", "bt_hand", "bt_tfidf", "neural")
MODEL_LABELS = {
    "random": "random",
    "length": "longer answer wins",
    "bt_hand": "Bradley-Terry, hand features",
    "bt_tfidf": "Bradley-Terry, TF-IDF + hand features",
    "neural": "DistilRoBERTa reward model",
}


def _r(x: float, digits: int = 4) -> float | None:
    return None if x is None or math.isnan(x) else round(float(x), digits)


def _slice(correct: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    n = int(mask.sum())
    return {"n": n, "accuracy": _r(correct[mask].mean()) if n else None}


def _ordered(score_sets: Sequence[ScoreSet]) -> list[ScoreSet]:
    rank = {name: i for i, name in enumerate(MODEL_ORDER)}
    return sorted(score_sets, key=lambda s: (rank.get(s.model, len(rank)), s.model))


def model_metrics(
    scores: ScoreSet, pairs: dict[str, PreferencePair], n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    test = scores.test
    batch = [pairs[pid] for pid in test.pair_ids]
    margins = test.margins
    correct = correctness(margins)
    lo, hi = bootstrap_ci(correct, n_boot=n_boot, seed=seed)

    temperature = fit_temperature(scores.val.margins)
    calibration = {
        "temperature": _r(temperature),
        "ece_raw": _r(expected_calibration_error(margins, 1.0)),
        "ece": _r(expected_calibration_error(margins, temperature)),
        "nll_raw": _r(pairwise_nll(margins, 1.0)),
        "nll": _r(pairwise_nll(margins, temperature)),
        "bins": [
            {k: _r(v) if isinstance(v, float) else v for k, v in b.to_dict().items()}
            for b in reliability(margins, temperature)
        ],
    }

    strength = np.array([p.strength for p in batch])
    unanimous = np.array([p.votes_rejected == 0 for p in batch])
    len_c = np.array([word_count(p.chosen) for p in batch])
    len_r = np.array([word_count(p.rejected) for p in batch])
    longer, shorter = _slice(correct, len_c > len_r), _slice(correct, len_c < len_r)
    gap = balanced = None
    if longer["accuracy"] is not None and shorter["accuracy"] is not None:
        gap = _r(longer["accuracy"] - shorter["accuracy"])
        balanced = _r((longer["accuracy"] + shorter["accuracy"]) / 2)

    rewards = np.concatenate([test.chosen, test.rejected])
    log_len = np.log1p(np.concatenate([len_c, len_r]))
    sd = float(rewards.std()) or 1.0

    probes: dict[str, Any] = {}
    ranked_right = margins > 0
    for probe in PROBES:
        if probe.name not in scores.probes:
            continue
        ps = scores.probes[probe.name]
        delta = np.concatenate([ps.chosen - test.chosen, ps.rejected - test.rejected])
        probes[probe.name] = {
            "expected": probe.expected,
            "mean_delta_sd": _r(delta.mean() / sd, 3),
            "frac_up": _r((delta > 0).mean(), 3),
            # Among pairs ranked correctly: the edited rejected answer now beats the chosen one.
            "rejected_wins_after_edit": _r((ps.rejected > test.chosen)[ranked_right].mean(), 3),
            # Among pairs ranked correctly: the edited chosen answer still beats the rejected one.
            "chosen_still_wins_after_edit": _r((ps.chosen > test.rejected)[ranked_right].mean(), 3),
        }

    return {
        "label": MODEL_LABELS.get(scores.model, scores.model),
        "n_test": len(batch),
        "accuracy": _r(correct.mean()),
        "accuracy_ci95": [_r(lo), _r(hi)],
        "val_accuracy": _r(correctness(scores.val.margins).mean()),
        "calibration": calibration,
        "by_strength": {
            str(s): _slice(correct, strength == s) for s in sorted(set(strength.tolist()))
        },
        "by_agreement": {
            "unanimous": _slice(correct, unanimous),
            "contested": _slice(correct, ~unanimous),
        },
        "length": {
            "chosen_longer": longer,
            "chosen_shorter": shorter,
            "same_length": _slice(correct, len_c == len_r),
            "gap": gap,
            "length_balanced_accuracy": balanced,
            "spearman_reward_vs_log_words": _r(spearman(rewards, log_len)),
            "spearman_margin_vs_log_length_ratio": _r(
                spearman(margins, np.log1p(len_c) - np.log1p(len_r))
            ),
        },
        "probes": probes,
        "meta": scores.meta,
    }


def human_reference(test_pairs: Sequence[PreferencePair]) -> dict[str, Any]:
    """How often a single labeller agrees with the aggregated label on the test pairs.

    Each labeller is part of the aggregate, so this is an optimistic reference, not a
    ceiling a model can be held to.
    """
    agree = sum(p.votes_chosen for p in test_pairs)
    disagree = sum(p.votes_rejected for p in test_pairs)
    tie = sum(p.votes_tie for p in test_pairs)
    total = agree + disagree + tie
    return {
        "votes": total,
        "single_labeller_agreement_excl_ties": _r(agree / max(1, agree + disagree)),
        "single_labeller_agreement_ties_as_half": _r((agree + 0.5 * tie) / max(1, total)),
        "unanimous_pairs": _r(np.mean([p.votes_rejected == 0 for p in test_pairs])),
    }


def describe_test_set(test_pairs: Sequence[PreferencePair]) -> dict[str, Any]:
    len_c = np.array([word_count(p.chosen) for p in test_pairs])
    len_r = np.array([word_count(p.rejected) for p in test_pairs])
    return {
        "pairs": len(test_pairs),
        "chosen_longer": _r((len_c > len_r).mean()),
        "chosen_shorter": _r((len_c < len_r).mean()),
        "median_words_chosen": int(np.median(len_c)),
        "median_words_rejected": int(np.median(len_r)),
    }


def build_report(
    score_sets: Sequence[ScoreSet],
    test_pairs: Sequence[PreferencePair],
    data_summary: dict[str, Any] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    lookup = {p.pair_id: p for p in test_pairs}
    return {
        "data": data_summary or {},
        "test_set": describe_test_set(test_pairs),
        "human_reference": human_reference(test_pairs),
        "probes": {p.name: {"expected": p.expected, "edit": p.description} for p in PROBES},
        "models": {
            s.model: model_metrics(s, lookup, n_boot=n_boot, seed=seed)
            for s in _ordered(score_sets)
        },
    }


def write_report(report: dict[str, Any], reports_dir: Path, charts: bool = True) -> list[Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "metrics.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    written = [path]
    if charts:
        from .plots import plot_calibration, plot_length_and_probes

        written.append(plot_calibration(report, reports_dir / "calibration.png"))
        written.append(plot_length_and_probes(report, reports_dir / "length_bias.png"))
    return written


def summary_table(report: dict[str, Any]) -> str:
    """Plain-text table for the terminal."""
    head = f"{'model':<10} {'acc':>6} {'95% CI':>15} {'ECE':>6} {'longer':>7} {'shorter':>7}"
    lines = [head, "-" * len(head)]
    for name, m in report["models"].items():
        ci = m["accuracy_ci95"]
        length = m["length"]
        lines.append(
            f"{name:<10} {m['accuracy']:>6.3f} {f'{ci[0]:.3f}-{ci[1]:.3f}':>15} "
            f"{m['calibration']['ece']:>6.3f} "
            f"{length['chosen_longer']['accuracy'] or 0:>7.3f} "
            f"{length['chosen_shorter']['accuracy'] or 0:>7.3f}"
        )
    return "\n".join(lines)
