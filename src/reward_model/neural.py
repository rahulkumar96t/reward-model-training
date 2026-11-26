"""Transformer reward model: a pretrained encoder with a scalar head.

Trained with the pairwise Bradley-Terry loss ``-log sigmoid(r_chosen - r_rejected)``, with two
options:

- a margin: ``-log sigmoid(r_c - r_r - m)`` where ``m = margin_scale * strength / 3`` grows
  with the labellers' preference strength (1-3), so clear wins must be separated more;
- reward centring: ``center_coef * (r_c + r_r)^2`` keeps rewards near zero, since the
  pairwise loss only fixes differences and leaves the offset free to drift.

Needs the ``train`` extra (torch, transformers). Nothing else in the package imports it.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from .data import PreferencePair
from .encoding import Budget, build_input, response_truncated
from .metrics import fit_temperature, pairwise_accuracy, pairwise_nll

DEFAULT_MODEL = "distilbert/distilroberta-base"
DEFAULT_REVISION = "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b"
POOLING = ("cls", "mean")


@dataclass
class TrainConfig:
    model_name: str = DEFAULT_MODEL
    revision: str | None = DEFAULT_REVISION
    max_length: int = 512
    max_prompt_tokens: int = 128
    pooling: str = "cls"
    dropout: float = 0.1
    epochs: int = 2
    batch_size: int = 8  # pairs per optimiser step (twice as many sequences)
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    grad_checkpointing: bool = False  # recompute activations in backward: less memory, slower
    margin_scale: float = 0.0
    center_coef: float = 0.0
    eval_batch_size: int = 32
    evals_per_epoch: int = 2
    seed: int = 13
    device: str = "auto"
    threads: int = 4
    log_every: int = 50

    @property
    def budget(self) -> Budget:
        return Budget(self.max_length, self.max_prompt_tokens)


def pick_device(name: str = "auto") -> torch.device:
    """``auto`` prefers Apple's MPS, then CUDA, then CPU; explicit names are checked."""
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def pairwise_loss(
    r_chosen: torch.Tensor,
    r_rejected: torch.Tensor,
    margin: torch.Tensor | None = None,
    center_coef: float = 0.0,
) -> torch.Tensor:
    """Bradley-Terry loss with an optional margin and reward-centring penalty."""
    diff = r_chosen - r_rejected
    if margin is not None:
        diff = diff - margin
    loss = -F.logsigmoid(diff).mean()
    if center_coef:
        loss = loss + center_coef * ((r_chosen + r_rejected) ** 2).mean()
    return loss


class RewardHead(nn.Module):
    """Pool the encoder output and project it to one scalar per sequence."""

    def __init__(self, hidden_size: int, pooling: str = "cls", dropout: float = 0.1) -> None:
        super().__init__()
        if pooling not in POOLING:
            raise ValueError(f"pooling must be one of {POOLING}")
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(hidden_size + 1))
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.proj(self.dropout(pooled)).squeeze(-1)


class RewardModel(nn.Module):
    def __init__(
        self, encoder: nn.Module, hidden_size: int, pooling: str = "cls", dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = RewardHead(hidden_size, pooling, dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state, attention_mask)


class PairEncoder:
    """Tokenise (prompt, response) pairs into ``[CLS] prompt [SEP] response [SEP]`` ids."""

    def __init__(self, tokenizer: Any, budget: Budget) -> None:
        self.tokenizer = tokenizer
        self.budget = budget
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        self.cls_id = cls_id if cls_id is not None else tokenizer.bos_token_id
        self.sep_id = sep_id if sep_id is not None else tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        self.pad_id = 0 if pad_id is None else pad_id
        if self.cls_id is None or self.sep_id is None:
            raise ValueError("tokenizer needs cls/sep (or bos/eos) tokens")

    def _token_ids(self, texts: Sequence[str]) -> dict[str, list[int]]:
        unique = list(dict.fromkeys(texts))
        ids = self.tokenizer(unique, add_special_tokens=False)["input_ids"]
        return dict(zip(unique, ids, strict=True))

    def encode(
        self, prompts: Sequence[str], responses: Sequence[str]
    ) -> tuple[list[list[int]], float]:
        """Input ids per pair and the fraction of responses cut by the budget."""
        prompt_ids = self._token_ids(prompts)
        response_ids = self._token_ids(responses)
        seqs, cut = [], 0
        for prompt, response in zip(prompts, responses, strict=True):
            p, r = prompt_ids[prompt], response_ids[response]
            cut += response_truncated(len(p), len(r), self.budget)
            seqs.append(build_input(p, r, self.budget, self.cls_id, self.sep_id))
        return seqs, cut / max(1, len(seqs))


def collate(
    seqs: Sequence[Sequence[int]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


def length_grouped_batches(
    lengths: np.ndarray, batch_size: int, rng: np.random.Generator, group: int = 50
) -> list[np.ndarray]:
    """Shuffle, then sort within chunks of ``group`` batches so padding stays small."""
    order = rng.permutation(len(lengths))
    chunk = batch_size * group
    batches: list[np.ndarray] = []
    for start in range(0, len(order), chunk):
        part = order[start : start + chunk]
        part = part[np.argsort(-lengths[part], kind="stable")]
        batches.extend(part[i : i + batch_size] for i in range(0, len(part), batch_size))
    perm = rng.permutation(len(batches))
    return [batches[i] for i in perm]


class NeuralScorer:
    """Batch scoring with a trained ``RewardModel``; implements ``RewardScorer``."""

    def __init__(
        self,
        model: RewardModel,
        encoder: PairEncoder,
        device: torch.device,
        batch_size: int = 32,
        temperature: float = 1.0,
        name: str = "neural",
    ) -> None:
        self.model = model.to(device)
        self.encoder = encoder
        self.device = device
        self.batch_size = batch_size
        self.temperature = temperature
        self.name = name
        self.last_truncated = 0.0

    def score_ids(self, seqs: Sequence[Sequence[int]]) -> np.ndarray:
        out = np.empty(len(seqs), dtype=np.float64)
        order = np.argsort([-len(s) for s in seqs], kind="stable")
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                ids, mask = collate([seqs[i] for i in idx], self.encoder.pad_id, self.device)
                out[idx] = self.model(ids, mask).float().cpu().numpy()
        self.model.train(was_training)
        return out

    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> np.ndarray:
        seqs, self.last_truncated = self.encoder.encode(prompts, responses)
        return self.score_ids(seqs)


@dataclass
class TrainResult:
    history: list[dict[str, Any]] = field(default_factory=list)
    best: dict[str, Any] = field(default_factory=dict)
    steps: int = 0
    train_seconds: float = 0.0
    temperature: float = 1.0
    truncated: dict[str, float] = field(default_factory=dict)
    peak_mps_gb: float | None = None


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        skip = param.ndim < 2 or "norm" in lowered or lowered.endswith("bias")
        (no_decay if skip else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def enable_checkpointing(encoder: nn.Module) -> None:
    """Recompute layer activations in the backward pass (renamed in transformers 5.2x)."""
    for name in ("activation_checkpointing_enable", "gradient_checkpointing_enable"):
        if hasattr(encoder, name):
            getattr(encoder, name)()
            return
    raise ValueError(f"{type(encoder).__name__} does not support activation checkpointing")


def _encode_pairs(
    encoder: PairEncoder, pairs: Sequence[PreferencePair]
) -> tuple[list[list[int]], list[list[int]], float]:
    seqs, cut = encoder.encode(
        [p.prompt for p in pairs] * 2, [p.chosen for p in pairs] + [p.rejected for p in pairs]
    )
    n = len(pairs)
    return seqs[:n], seqs[n:], cut


def train_reward_model(
    model: RewardModel,
    encoder: PairEncoder,
    config: TrainConfig,
    train_pairs: Sequence[PreferencePair],
    val_pairs: Sequence[PreferencePair],
    device: torch.device,
    log: Callable[[str], None] = print,
) -> TrainResult:
    """Train in place; the weights with the best validation accuracy are kept."""
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    result = TrainResult()
    train_c, train_r, train_cut = _encode_pairs(encoder, train_pairs)
    val_c, val_r, val_cut = _encode_pairs(encoder, val_pairs)
    result.truncated = {"train": round(train_cut, 4), "val": round(val_cut, 4)}
    strength = torch.tensor([p.strength for p in train_pairs], dtype=torch.float32)
    lengths = np.array([max(len(c), len(r)) for c, r in zip(train_c, train_r, strict=True)])

    model.to(device).train()
    if config.grad_checkpointing:
        enable_checkpointing(model.encoder)
    scorer = NeuralScorer(model, encoder, device, batch_size=config.eval_batch_size)
    optimizer = torch.optim.AdamW(_param_groups(model, config.weight_decay), lr=config.lr)
    steps_per_epoch = math.ceil(len(train_pairs) / config.batch_size)
    total = max(1, steps_per_epoch * config.epochs)
    warmup = max(1, int(total * config.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (total - s) / max(1, total - warmup)))
    )

    best_key = (-math.inf, -math.inf)
    best_state: dict[str, torch.Tensor] | None = None
    start = time.perf_counter()
    window_loss: list[float] = []
    window_acc: list[float] = []

    def evaluate(epoch: float) -> None:
        nonlocal best_key, best_state
        val_margins = scorer.score_ids(val_c) - scorer.score_ids(val_r)
        acc, nll = pairwise_accuracy(val_margins), pairwise_nll(val_margins)
        row = {
            "epoch": round(epoch, 2),
            "step": result.steps,
            "val_accuracy": round(acc, 4),
            "val_nll": round(nll, 4),
            "elapsed_s": round(time.perf_counter() - start, 1),
        }
        result.history.append(row)
        if device.type == "mps":
            torch.mps.empty_cache()
        log(f"  eval  epoch {row['epoch']:.2f}  val acc {acc:.4f}  val loss {nll:.4f}")
        if (acc, -nll) > best_key:
            best_key = (acc, -nll)
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            result.best = dict(row)
            result.temperature = fit_temperature(val_margins)

    for epoch in range(config.epochs):
        batches = length_grouped_batches(lengths, config.batch_size, rng)
        checkpoints = {
            math.ceil(len(batches) * (k + 1) / config.evals_per_epoch)
            for k in range(config.evals_per_epoch)
        }
        for i, idx in enumerate(batches, start=1):
            k = len(idx)
            ids, mask = collate(
                [train_c[j] for j in idx] + [train_r[j] for j in idx], encoder.pad_id, device
            )
            rewards = model(ids, mask)
            r_c, r_r = rewards[:k], rewards[k:]
            margin = None
            if config.margin_scale:
                margin = (strength[idx] * config.margin_scale / 3.0).to(device)
            loss = pairwise_loss(r_c, r_r, margin, config.center_coef)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            result.steps += 1
            window_loss.append(float(loss.detach()))
            window_acc.append(float((r_c > r_r).float().mean().detach()))
            if device.type == "mps":
                used = torch.mps.driver_allocated_memory() / 2**30
                result.peak_mps_gb = round(max(result.peak_mps_gb or 0.0, used), 2)
            if result.steps % config.log_every == 0:
                elapsed = time.perf_counter() - start
                log(
                    f"  step {result.steps:>5}/{total}  loss {np.mean(window_loss):.4f}  "
                    f"train acc {np.mean(window_acc):.3f}  {elapsed:.0f}s"
                )
                window_loss.clear()
                window_acc.clear()
            if i in checkpoints:
                evaluate(epoch + i / len(batches))
                model.train()

    result.train_seconds = round(time.perf_counter() - start, 1)
    if best_state is not None:
        model.load_state_dict(best_state)
    return result


def build_model(config: TrainConfig) -> tuple[RewardModel, Any]:
    """Load the pretrained encoder and tokenizer at the pinned revision."""
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision)
    encoder = AutoModel.from_pretrained(config.model_name, revision=config.revision)
    model = RewardModel(encoder, encoder.config.hidden_size, config.pooling, config.dropout)
    return model, tokenizer


def save_checkpoint(
    model: RewardModel,
    tokenizer: Any,
    config: TrainConfig,
    out_dir: Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``encoder/``, ``tokenizer/``, ``head.safetensors`` and ``reward_config.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    model.to("cpu")
    model.encoder.save_pretrained(out_dir / "encoder")
    save_file(
        {k: v.contiguous() for k, v in model.head.state_dict().items()},
        str(out_dir / "head.safetensors"),
    )
    model.to(device)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    payload = {"config": asdict(config), **(extra or {})}
    (out_dir / "reward_config.json").write_text(json.dumps(payload, indent=2) + "\n")
    return out_dir


def load_checkpoint(path: Path, device: str = "auto", name: str = "neural") -> NeuralScorer:
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    meta = json.loads((path / "reward_config.json").read_text())
    config = TrainConfig(**meta["config"])
    encoder = AutoModel.from_pretrained(path / "encoder")
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
    model = RewardModel(encoder, encoder.config.hidden_size, config.pooling, config.dropout)
    model.head.load_state_dict(load_file(str(path / "head.safetensors")))
    return NeuralScorer(
        model,
        PairEncoder(tokenizer, config.budget),
        pick_device(device),
        batch_size=config.eval_batch_size,
        temperature=float(meta.get("temperature", 1.0)),
        name=name,
    )
