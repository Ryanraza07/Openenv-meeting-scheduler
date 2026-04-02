from __future__ import annotations

from .state import MeetingState


def compute_score(state: MeetingState, slot: str) -> float:
    if slot not in state.all_slots:
        return 0.0

    if any(participant.required and slot not in participant.available_slots for participant in state.participants):
        return 0.0

    total_priority = sum(participant.priority for participant in state.participants)
    if total_priority <= 0:
        return 0.0

    covered_priority = sum(
        participant.priority for participant in state.participants if slot in participant.available_slots
    )
    return covered_priority / total_priority
