from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

try:
    from ..env.reward import compute_score
    from ..env.state import MeetingState
except ImportError:
    from app.env.reward import compute_score
    from app.env.state import MeetingState

_GRADE_PRECISION = Decimal("0.000001")
_DISPLAY_SAFE_MARGIN = Decimal("0.0001")
_ONE = Decimal("1")


def _normalize_grade(value: Decimal) -> float:
    return float(value.quantize(_GRADE_PRECISION, rounding=ROUND_HALF_UP))


def _bound_grade(value: Decimal) -> float:
    bounded = min(_ONE - _DISPLAY_SAFE_MARGIN, max(_DISPLAY_SAFE_MARGIN, value))
    return _normalize_grade(bounded)


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
        raw_grade = _ONE if chosen_reward == 0 else Decimal("0")
        return _bound_grade(raw_grade)

    return _bound_grade(chosen_reward / best_reward)
