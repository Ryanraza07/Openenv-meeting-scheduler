from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .action import Action
from .reward import compute_score
from .state import MeetingState

TaskLoader = Callable[[], MeetingState]


class MeetingEnv:
    def __init__(self, task_loader: TaskLoader):
        self._task_loader = task_loader
        self._current_state: MeetingState | None = None
        self._done = False

    def reset(self) -> MeetingState:
        self._current_state = self._task_loader()
        self._done = False
        return self._current_state.model_copy(deep=True)

    def state(self) -> MeetingState:
        if self._current_state is None:
            raise RuntimeError("Call reset before requesting the environment state.")
        return self._current_state.model_copy(deep=True)

    def step(self, action: Action) -> dict[str, Any]:
        if self._current_state is None:
            raise RuntimeError("Call reset before step.")
        if self._done:
            raise RuntimeError("Episode already completed. Call reset before step.")

        valid_action = action.chosen_slot in self._current_state.all_slots
        reason = None
        if valid_action:
            reward = compute_score(self._current_state, action.chosen_slot)
        else:
            reward = 0.0
            allowed_slots = ", ".join(self._current_state.all_slots)
            reason = f"chosen_slot must be one of [{allowed_slots}]"

        self._done = True
        return {
            "reward": reward,
            "done": True,
            "info": {
                "valid_action": valid_action,
                "score": reward,
                "reason": reason,
            },
        }
