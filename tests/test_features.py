from __future__ import annotations

import math

import numpy as np
import pytest

from reward_model.features import (
    HAND_FEATURES,
    ResponseFeaturizer,
    hand_feature_matrix,
    hand_features,
    word_count,
    words,
)


def _feature(prompt: str, response: str, name: str) -> float:
    return float(hand_features(prompt, response)[HAND_FEATURES.index(name)])


def test_words_keep_contractions_and_lowercase() -> None:
    assert words("Don't PANIC, it's 42.") == ["don't", "panic", "it's", "42"]
    assert word_count("one two  three\nfour") == 4


def test_length_features() -> None:
    assert _feature("q", "a b c", "log_words") == pytest.approx(math.log1p(3))
    assert _feature("one two", "a b c d", "log_length_ratio") == pytest.approx(
        math.log1p(4) - math.log1p(2)
    )


def test_structure_features() -> None:
    response = "# Steps\n1. Mix flour\n2. Add water\n- Rest\n```\ncode\n```\nDone?"
    assert _feature("q", response, "log_list_items") == pytest.approx(math.log1p(3))
    assert _feature("q", response, "log_headers") == pytest.approx(math.log1p(1))
    assert _feature("q", response, "code_blocks") == 1.0
    assert _feature("q", response, "log_questions") == pytest.approx(math.log1p(1))


def test_prompt_overlap_ignores_stop_words() -> None:
    prompt = "How do I bake sourdough bread with rye flour?"
    # Content words: bake, sourdough, bread, rye, flour.
    overlap = _feature(prompt, "Use rye flour and a warm oven.", "prompt_overlap")
    assert overlap == pytest.approx(2 / 5)
    assert _feature("the of and", "anything", "prompt_overlap") == 0.0


def test_repetition_lowers_distinct_bigrams() -> None:
    varied = "the cat sat on a warm mat near the door"
    repeated = " ".join(["the cat sat"] * 5)
    assert _feature("q", varied, "distinct_bigrams") == 1.0
    assert _feature("q", repeated, "distinct_bigrams") < 0.3


def test_refusal_and_clean_ending() -> None:
    assert _feature("q", "I'm sorry, but I can't do that.", "refusal") == 1.0
    assert _feature("q", "Sure, here it is.", "refusal") == 0.0
    assert _feature("q", "It ends properly.", "clean_ending") == 1.0
    assert _feature("q", "It stops in the mid", "clean_ending") == 0.0


def test_matrix_shape_and_empty_input() -> None:
    assert hand_feature_matrix(["a", "b"], ["x y", "z"]).shape == (2, len(HAND_FEATURES))
    assert hand_feature_matrix([], []).shape == (0, len(HAND_FEATURES))


def test_featurizer_with_and_without_tfidf() -> None:
    prompts = ["q"] * 6
    responses = ["good answer here", "bad answer", "good good", "answer", "here here", "bad"]
    hand_only = ResponseFeaturizer(use_tfidf=False).fit(prompts, responses)
    assert hand_only.transform(prompts, responses).shape == (6, len(HAND_FEATURES))
    full = ResponseFeaturizer(use_tfidf=True, min_df=1).fit(prompts, responses)
    x = full.transform(prompts, responses)
    assert x.shape[1] == len(full.feature_names) > len(HAND_FEATURES)
    assert "tfidf:good" in full.feature_names
    # Standardised hand features: mean zero on the fitted responses.
    hand_block = x[:, : len(HAND_FEATURES)].toarray()
    assert np.allclose(hand_block.mean(axis=0), 0.0, atol=1e-9)
