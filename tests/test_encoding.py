from __future__ import annotations

import pytest

from reward_model.encoding import Budget, build_input, response_truncated, truncate_pair


def test_short_inputs_are_untouched() -> None:
    prompt, response = truncate_pair([1, 2], [3, 4, 5], Budget(16, 4))
    assert (prompt, response) == ([1, 2], [3, 4, 5])


def test_long_prompt_keeps_its_last_tokens() -> None:
    prompt, response = truncate_pair(list(range(20)), list(range(100, 109)), Budget(16, 4))
    assert response == list(range(100, 109))
    assert prompt == [16, 17, 18, 19]


def test_long_response_keeps_its_first_tokens() -> None:
    prompt, response = truncate_pair(list(range(20)), list(range(100, 140)), Budget(16, 4))
    assert prompt == [16, 17, 18, 19]
    assert response == list(range(100, 109))


def test_short_response_hands_room_back_to_the_prompt() -> None:
    prompt, response = truncate_pair(list(range(20)), [100, 101], Budget(16, 4))
    assert response == [100, 101]
    assert prompt == list(range(9, 20))


@pytest.mark.parametrize("p_len,r_len", [(0, 50), (50, 0), (7, 7), (300, 300), (1, 600)])
def test_total_length_never_exceeds_budget(p_len: int, r_len: int) -> None:
    budget = Budget(32, 8)
    ids = build_input(list(range(p_len)), list(range(r_len)), budget, cls_id=-1, sep_id=-2)
    assert len(ids) <= budget.max_length
    assert ids[0] == -1 and ids[-1] == -2 and ids.count(-2) == 2


def test_response_truncated_flag() -> None:
    budget = Budget(16, 4)
    assert not response_truncated(4, 9, budget)
    assert response_truncated(4, 10, budget)
    assert response_truncated(20, 10, budget)
    assert not response_truncated(2, 11, budget)


def test_budget_validation() -> None:
    with pytest.raises(ValueError):
        Budget(3, 1)
    with pytest.raises(ValueError):
        Budget(16, 13)
    with pytest.raises(ValueError):
        Budget(16, 0)
