"""Response features for the linear Bradley-Terry baselines.

Hand features describe the shape of a response (length, structure, repetition, overlap with
the prompt, refusal phrases). The optional TF-IDF branch adds word uni- and bi-grams of the
response. Features are computed per (prompt, response); a pair is represented by the
difference between the chosen and the rejected response's features.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_HEADER = re.compile(r"^\s*#{1,6}\s+\S", re.MULTILINE)
_REFUSAL = re.compile(
    r"\b(?:i'?m sorry|i am sorry|i apologi[sz]e|i can(?:no|')t|i am unable|i'?m unable|"
    r"as an ai|i do not have the ability|i don't have the ability)\b",
    re.IGNORECASE,
)
_STOP_WORDS = """
the a an and or of to in on for with is are was were be this that it as at by from what how
why which who when where can could would should do does did you your i me my we our they their
them he she his her its if not no yes
"""
_STOP = frozenset(_STOP_WORDS.split())

HAND_FEATURES: tuple[str, ...] = (
    "log_words",
    "log_lines",
    "log_list_items",
    "log_headers",
    "code_blocks",
    "log_questions",
    "prompt_overlap",
    "distinct_bigrams",
    "refusal",
    "clean_ending",
    "log_words_per_sentence",
    "log_length_ratio",
)


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def _content_words(text: str) -> set[str]:
    return {w for w in words(text) if len(w) > 2 and w not in _STOP}


def hand_features(prompt: str, response: str) -> np.ndarray:
    tokens = words(response)
    n = len(tokens)
    lines = [line for line in response.splitlines() if line.strip()]
    sentences = max(1, len(_SENTENCE_END.findall(response)))
    prompt_words = _content_words(prompt)
    overlap = len(prompt_words & set(tokens)) / len(prompt_words) if prompt_words else 0.0
    bigrams = list(pairwise(tokens))
    distinct = len(set(bigrams)) / len(bigrams) if bigrams else 1.0
    stripped = response.rstrip()
    clean_end = 1.0 if stripped.endswith((".", "!", "?", ")", '"', "`", ":", "*")) else 0.0
    values = (
        math.log1p(n),
        math.log1p(len(lines)),
        math.log1p(len(_LIST_ITEM.findall(response))),
        math.log1p(len(_HEADER.findall(response))),
        1.0 if "```" in response else 0.0,
        math.log1p(response.count("?")),
        overlap,
        distinct,
        1.0 if _REFUSAL.search(response) else 0.0,
        clean_end,
        math.log1p(n / sentences),
        math.log1p(n) - math.log1p(word_count(prompt)),
    )
    return np.asarray(values, dtype=np.float64)


def hand_feature_matrix(prompts: Sequence[str], responses: Sequence[str]) -> np.ndarray:
    if not prompts:
        return np.zeros((0, len(HAND_FEATURES)))
    rows = [hand_features(p, r) for p, r in zip(prompts, responses, strict=True)]
    return np.vstack(rows)


class ResponseFeaturizer:
    """Standardised hand features, optionally followed by TF-IDF of the response."""

    def __init__(self, use_tfidf: bool = True, max_features: int = 50_000, min_df: int = 3) -> None:
        self.use_tfidf = use_tfidf
        self.scaler = StandardScaler()
        self.tfidf = (
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=min_df,
                max_features=max_features,
                sublinear_tf=True,
                dtype=np.float64,
            )
            if use_tfidf
            else None
        )

    @property
    def feature_names(self) -> list[str]:
        names = list(HAND_FEATURES)
        if self.tfidf is not None:
            names += [f"tfidf:{t}" for t in self.tfidf.get_feature_names_out()]
        return names

    def fit(self, prompts: Sequence[str], responses: Sequence[str]) -> ResponseFeaturizer:
        self.scaler.fit(hand_feature_matrix(prompts, responses))
        if self.tfidf is not None:
            self.tfidf.fit(responses)
        return self

    def transform(self, prompts: Sequence[str], responses: Sequence[str]) -> sparse.csr_matrix:
        hand = sparse.csr_matrix(self.scaler.transform(hand_feature_matrix(prompts, responses)))
        if self.tfidf is None:
            return hand
        return sparse.hstack([hand, self.tfidf.transform(responses)], format="csr")
