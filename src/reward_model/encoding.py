"""Token budget for (prompt, response) inputs to the neural reward model.

The encoder sees ``[CLS] prompt [SEP] response [SEP]`` with at most ``max_length`` tokens.
The response matters most, so it gets the room first, but the prompt always keeps at least
its last ``max_prompt_tokens`` tokens (the latest user turn sits at the end of the prompt).

- The prompt is cut from the left: its last tokens are kept.
- The response is cut from the right: its first tokens are kept.
- A short response hands its unused budget back to the prompt.

This module has no torch dependency, so the truncation rules are tested in CI.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

SPECIAL_TOKENS = 3  # [CLS], [SEP] after the prompt, [SEP] after the response


@dataclass(frozen=True)
class Budget:
    max_length: int = 512
    max_prompt_tokens: int = 128

    def __post_init__(self) -> None:
        if self.max_length <= SPECIAL_TOKENS + 1:
            raise ValueError("max_length too small")
        if not 0 < self.max_prompt_tokens < self.max_length - SPECIAL_TOKENS:
            raise ValueError("max_prompt_tokens must leave room for the response")

    @property
    def content(self) -> int:
        return self.max_length - SPECIAL_TOKENS


def truncate_pair(
    prompt_ids: Sequence[int], response_ids: Sequence[int], budget: Budget
) -> tuple[list[int], list[int]]:
    """Return the kept prompt tokens (a suffix) and response tokens (a prefix)."""
    prompt_keep = min(len(prompt_ids), budget.max_prompt_tokens)
    response_keep = min(len(response_ids), budget.content - prompt_keep)
    prompt_keep = min(len(prompt_ids), budget.content - response_keep)
    prompt_part = list(prompt_ids[len(prompt_ids) - prompt_keep :]) if prompt_keep else []
    return prompt_part, list(response_ids[:response_keep])


def build_input(
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    budget: Budget,
    cls_id: int,
    sep_id: int,
) -> list[int]:
    prompt_part, response_part = truncate_pair(prompt_ids, response_ids, budget)
    return [cls_id, *prompt_part, sep_id, *response_part, sep_id]


def response_truncated(prompt_len: int, response_len: int, budget: Budget) -> bool:
    """True when part of the response falls outside the model's window."""
    prompt_keep = min(prompt_len, budget.max_prompt_tokens)
    return response_len > budget.content - prompt_keep
