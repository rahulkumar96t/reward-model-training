"""Neural reward model tests. They need the ``train`` extra and skip without it.

A one-layer BERT with random weights and a word-level tokenizer built from the synthetic
vocabulary stand in for the pretrained encoder, so nothing is downloaded.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

from reward_model.data import PreferencePair  # noqa: E402
from reward_model.encoding import Budget  # noqa: E402
from reward_model.metrics import pairwise_accuracy  # noqa: E402
from reward_model.neural import (  # noqa: E402
    NeuralScorer,
    PairEncoder,
    RewardHead,
    RewardModel,
    TrainConfig,
    collate,
    length_grouped_batches,
    load_checkpoint,
    pairwise_loss,
    pick_device,
    save_checkpoint,
    train_reward_model,
)
from reward_model.scoring import score_pairs  # noqa: E402

CPU = torch.device("cpu")


def _tokenizer(pairs: list[PreferencePair]) -> transformers.PreTrainedTokenizerBase:
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]"])}
    for p in pairs:
        for text in (p.prompt, p.chosen, p.rejected):
            for token in re.findall(r"\w+|[^\w\s]", text.lower()):
                vocab.setdefault(token, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def _tiny_model(vocab_size: int, pooling: str = "cls", seed: int = 0) -> RewardModel:
    torch.manual_seed(seed)
    config = transformers.BertConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    return RewardModel(transformers.BertModel(config), 32, pooling=pooling, dropout=0.0)


def test_loss_equal_rewards_is_log_two() -> None:
    r = torch.zeros(4)
    assert float(pairwise_loss(r, r)) == pytest.approx(math.log(2))


def test_loss_decreases_with_margin_and_respects_options() -> None:
    chosen, rejected = torch.tensor([2.0, 1.0]), torch.tensor([0.0, 0.0])
    base = float(pairwise_loss(chosen, rejected))
    assert base < math.log(2)
    with_margin = float(pairwise_loss(chosen, rejected, margin=torch.tensor([1.0, 1.0])))
    assert with_margin > base
    expected = -torch.nn.functional.logsigmoid(torch.tensor([1.0, 0.0])).mean()
    assert with_margin == pytest.approx(float(expected))
    centred = float(pairwise_loss(chosen, rejected, center_coef=0.1))
    assert centred == pytest.approx(base + 0.1 * (4.0 + 1.0) / 2)


def test_loss_gradient_pushes_rewards_apart() -> None:
    chosen = torch.zeros(3, requires_grad=True)
    rejected = torch.zeros(3, requires_grad=True)
    pairwise_loss(chosen, rejected).backward()
    assert torch.all(chosen.grad < 0) and torch.all(rejected.grad > 0)


def test_head_pooling_ignores_padding_for_mean() -> None:
    head = RewardHead(4, pooling="mean", dropout=0.0)
    hidden = torch.tensor([[[1.0, 1, 1, 1], [3.0, 3, 3, 3], [100.0, 100, 100, 100]]])
    mask = torch.tensor([[1, 1, 0]])
    expected = head.proj(torch.full((1, 4), 2.0)).squeeze(-1)
    assert torch.allclose(head(hidden, mask), expected)
    cls = RewardHead(4, pooling="cls", dropout=0.0)
    assert torch.allclose(cls(hidden, mask), cls.proj(hidden[:, 0]).squeeze(-1))
    with pytest.raises(ValueError):
        RewardHead(4, pooling="max")


def test_model_returns_one_reward_per_sequence() -> None:
    model = _tiny_model(20)
    ids, mask = collate([[2, 5, 3, 6, 3], [2, 7, 3]], pad_id=0, device=CPU)
    assert ids.shape == mask.shape == (2, 5)
    assert mask.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert model(ids, mask).shape == (2,)


def test_pair_encoder_applies_budget(splits: dict[str, list[PreferencePair]]) -> None:
    tokenizer = _tokenizer(splits["train"])
    encoder = PairEncoder(tokenizer, Budget(24, 6))
    seqs, cut = encoder.encode(["a b c d e f g h"], ["x " * 40])
    assert len(seqs[0]) == 24
    assert seqs[0][0] == tokenizer.cls_token_id and seqs[0][-1] == tokenizer.sep_token_id
    assert cut == 1.0


def test_length_grouped_batches_cover_every_index_once() -> None:
    lengths = np.random.default_rng(0).integers(1, 500, size=103)
    batches = length_grouped_batches(lengths, 8, np.random.default_rng(1), group=4)
    flat = np.concatenate(batches)
    assert sorted(flat.tolist()) == list(range(103))
    assert max(len(b) for b in batches) == 8


def test_pick_device() -> None:
    assert pick_device("cpu").type == "cpu"
    assert pick_device("auto").type in {"cpu", "mps", "cuda"}


def test_tiny_model_learns_planted_signal_and_round_trips(
    splits: dict[str, list[PreferencePair]], tmp_path: Path
) -> None:
    torch.set_num_threads(2)
    train, val, test = splits["train"][:160], splits["val"], splits["test"]
    tokenizer = _tokenizer(train + val + test)
    config = TrainConfig(
        max_length=96,
        max_prompt_tokens=24,
        epochs=4,
        batch_size=8,
        lr=1e-3,
        warmup_ratio=0.1,
        center_coef=0.01,
        margin_scale=0.5,
        evals_per_epoch=1,
        eval_batch_size=64,
        log_every=10_000,
    )
    model = _tiny_model(tokenizer.vocab_size)
    encoder = PairEncoder(tokenizer, config.budget)
    scorer = NeuralScorer(model, encoder, CPU)
    before = pairwise_accuracy(score_pairs(scorer, test).margins)
    result = train_reward_model(model, encoder, config, train, val, CPU, log=lambda _: None)
    after = pairwise_accuracy(score_pairs(scorer, test).margins)
    assert len(result.history) == 4
    assert result.best["val_accuracy"] == max(r["val_accuracy"] for r in result.history)
    assert after > 0.85 and after > before
    assert result.temperature > 0

    save_checkpoint(model, tokenizer, config, tmp_path / "ckpt", {"temperature": 1.5})
    loaded = load_checkpoint(tmp_path / "ckpt", device="cpu")
    assert loaded.temperature == 1.5
    assert np.allclose(
        score_pairs(loaded, test).margins, score_pairs(scorer, test).margins, atol=1e-5
    )


def test_score_cli_with_checkpoint(
    splits: dict[str, list[PreferencePair]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from reward_model.cli import main

    tokenizer = _tokenizer(splits["train"])
    model = _tiny_model(tokenizer.vocab_size)
    save_checkpoint(
        model,
        tokenizer,
        TrainConfig(max_length=64, max_prompt_tokens=16),
        tmp_path / "ckpt",
        {"temperature": 1.0},
    )
    code = main(
        [
            "score",
            "--checkpoint",
            str(tmp_path / "ckpt"),
            "--device",
            "cpu",
            "--prompt",
            "hi",
            "--response",
            "check the starter",
            "--response",
            "stuff",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["rewards"]) == 2 and 0.0 < out["p_first_beats_second"] < 1.0


def test_gradient_checkpointing_trains(splits: dict[str, list[PreferencePair]]) -> None:
    train = splits["train"][:16]
    tokenizer = _tokenizer(train)
    config = TrainConfig(
        max_length=64, max_prompt_tokens=16, epochs=1, evals_per_epoch=1, grad_checkpointing=True
    )
    model = _tiny_model(tokenizer.vocab_size)
    encoder = PairEncoder(tokenizer, config.budget)
    result = train_reward_model(model, encoder, config, train, train, CPU, log=lambda _: None)
    flag = getattr(model.encoder, "is_activation_checkpointing", None)
    if flag is None:
        flag = model.encoder.is_gradient_checkpointing
    assert flag
    assert result.steps == 2
