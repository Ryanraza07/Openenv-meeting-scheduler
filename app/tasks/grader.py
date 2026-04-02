from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from app.env.reward import compute_score
from app.env.state import MeetingState

_GRADE_PRECISION = Decimal("0.000001")


def _normalize_grade(value: Decimal) -> float:
    return float(value.quantize(_GRADE_PRECISION, rounding=ROUND_HALF_UP))


def _score_or_zero(state: MeetingState, slot: str) -> float:
    if slot not in state.all_slots:
        return 0.0
    return compute_score(state, slot)


def find_best_slot(state: MeetingState) -> str:
    best_slot = state.all_slots[0]
    best_reward = _score_or_zero(state, best_slot)

    for slot in state.all_slots[1:]:
        reward = _score_or_zero(state, slot)
        if reward > best_reward:
            best_slot = slot
            best_reward = reward

    return best_slot


def grade(state: MeetingState, chosen_slot: str) -> float:
    best_slot = find_best_slot(state)
    best_reward = Decimal(str(_score_or_zero(state, best_slot)))
    chosen_reward = Decimal(str(_score_or_zero(state, chosen_slot)))

    if best_reward == 0:
        return 1.0 if chosen_reward == 0 else 0.0

    return _normalize_grade(chosen_reward / best_reward)
