from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import os
import sys

try:
    from .env.action import Action
    from .env.environment import MeetingEnv
    from .env.reward import compute_score
    from .env.state import MeetingState, Participant
    from .tasks.easy import load_easy_task
    from .tasks.grader import find_best_slot, grade
    from .tasks.hard import load_hard_task
    from .tasks.medium import load_medium_task
except ImportError:
    from app.env.action import Action
    from app.env.environment import MeetingEnv
    from app.env.reward import compute_score
    from app.env.state import MeetingState, Participant
    from app.tasks.easy import load_easy_task
    from app.tasks.grader import find_best_slot, grade
    from app.tasks.hard import load_hard_task
    from app.tasks.medium import load_medium_task

_DISPLAY_PRECISION = Decimal("0.0001")
_VERBOSE_ENV_VAR = "INFERENCE_VERBOSE"
_FALLBACK_SCORE = 0.0001
TASK_LOADERS: dict[str, Callable[[], MeetingState]] = {
    "easy": load_easy_task,
    "medium": load_medium_task,
    "hard": load_hard_task,
}


@dataclass(frozen=True)
class SlotAnalysis:
    slot: str
    attending: tuple[Participant, ...]
    missing: tuple[Participant, ...]
    covered_priority: Decimal
    total_priority: Decimal
    required_satisfied: bool
    score: float


@dataclass(frozen=True)
class AgentSelection:
    chosen_slot: str
    strategy: str
    note: str | None = None


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _format_decimal(value: Decimal) -> str:
    quantized = value.quantize(_DISPLAY_PRECISION, rounding=ROUND_HALF_UP)
    return f"{quantized:.4f}"


def _format_score(value: float) -> str:
    return _format_decimal(_decimal(value))


def _emit_structured(tag: str, **fields: object) -> None:
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[{tag}] {payload}".rstrip(), flush=True)


def _verbose_enabled() -> bool:
    return os.getenv(_VERBOSE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def _load_baseline_agent():
    try:
        from .agent.baseline_agent import choose_best_slot
    except ImportError:
        from app.agent.baseline_agent import choose_best_slot
    return choose_best_slot


def _choose_slot(state: MeetingState) -> AgentSelection:
    fallback_slot = find_best_slot(state)

    try:
        choose_best_slot = _load_baseline_agent()
    except ImportError as exc:
        return AgentSelection(
            chosen_slot=fallback_slot,
            strategy="deterministic_fallback",
            note=f"Optional baseline agent dependencies are unavailable: {exc}",
        )

    try:
        return AgentSelection(
            chosen_slot=choose_best_slot(state, validate=False),
            strategy="baseline_agent",
        )
    except Exception as exc:
        return AgentSelection(
            chosen_slot=fallback_slot,
            strategy="deterministic_fallback",
            note=f"Baseline agent failed and the local optimal solver was used instead: {exc}",
        )


def _sorted_participants(state: MeetingState) -> list[Participant]:
    return sorted(state.participants, key=lambda participant: participant.name.casefold())


def _sorted_slots(state: MeetingState) -> list[str]:
    return sorted(state.all_slots)


def _total_priority(state: MeetingState) -> Decimal:
    return sum((_decimal(participant.priority) for participant in state.participants), start=Decimal("0"))


def _format_weighted_participants(participants: tuple[Participant, ...] | list[Participant]) -> str:
    if not participants:
        return "None"
    return ", ".join(f"{participant.name}({_format_score(participant.priority)})" for participant in participants)


def _analyze_slot(state: MeetingState, slot: str) -> SlotAnalysis:
    participants = _sorted_participants(state)
    attending = tuple(participant for participant in participants if slot in participant.available_slots)
    missing = tuple(participant for participant in participants if slot not in participant.available_slots)
    covered_priority = sum((_decimal(participant.priority) for participant in attending), start=Decimal("0"))
    required_satisfied = all(
        not participant.required or slot in participant.available_slots for participant in state.participants
    )

    return SlotAnalysis(
        slot=slot,
        attending=attending,
        missing=missing,
        covered_priority=covered_priority,
        total_priority=_total_priority(state),
        required_satisfied=required_satisfied,
        score=compute_score(state, slot),
    )


def explain_decision(state: MeetingState, chosen_slot: str) -> str:
    if chosen_slot not in state.all_slots:
        return "\n".join(
            [
                "Chosen slot is invalid because it is not present in all_slots.",
                "Attending participants: None",
                "Missing participants: None",
                "Total priority covered: 0.0000 of "
                f"{_format_decimal(_total_priority(state))}",
                "Required participants satisfied: No",
            ]
        )

    analysis = _analyze_slot(state, chosen_slot)
    missing_required = [participant.name for participant in analysis.missing if participant.required]
    best_slot = find_best_slot(state)

    if not analysis.required_satisfied:
        summary = "Chosen slot does not satisfy all required participants, so the reward is zero."
    elif chosen_slot == best_slot:
        summary = "Chosen slot maximizes weighted attendance while satisfying all constraints."
    else:
        summary = "Chosen slot satisfies required participants but does not maximize weighted attendance."

    lines = [
        summary,
        f"Attending participants: {_format_weighted_participants(analysis.attending)}",
        f"Missing participants: {_format_weighted_participants(analysis.missing)}",
        "Total priority covered: "
        f"{_format_decimal(analysis.covered_priority)} of {_format_decimal(analysis.total_priority)}",
        f"Required participants satisfied: {'Yes' if analysis.required_satisfied else 'No'}",
    ]
    if missing_required:
        lines.append(f"Missing required participants: {', '.join(sorted(missing_required))}")

    return "\n".join(lines)


def _validate_chosen_slot(state: MeetingState, chosen_slot: str) -> str | None:
    if chosen_slot in state.all_slots:
        return None
    expected_slots = ", ".join(_sorted_slots(state))
    return f"chosen_slot must be one of [{expected_slots}]"


def _render_state_summary(state: MeetingState) -> str:
    participants = _sorted_participants(state)
    name_width = max(len(participant.name) for participant in participants)
    required_width = len("required=yes")

    lines = [
        "==== STATE ====",
        f"Meeting duration: {state.meeting_duration} minutes",
        "All slots:",
    ]
    for index, slot in enumerate(_sorted_slots(state), start=1):
        lines.append(f"  {index:>2}. {slot}")

    lines.append("Participants:")
    for participant in participants:
        availability = ", ".join(sorted(participant.available_slots))
        required_text = f"required={'yes' if participant.required else 'no'}"
        lines.append(
            f"  {participant.name:<{name_width}} | priority={_format_score(participant.priority)} | "
            f"{required_text:<{required_width}} | available={availability}"
        )

    return "\n".join(lines)


def _render_slot_analysis(state: MeetingState) -> str:
    lines = ["==== SLOT ANALYSIS ===="]
    for slot in _sorted_slots(state):
        analysis = _analyze_slot(state, slot)
        missing_required = [participant.name for participant in analysis.missing if participant.required]

        lines.extend(
            [
                f"Slot: {analysis.slot}",
                f"  Attending: {_format_weighted_participants(analysis.attending)}",
                f"  Missing: {_format_weighted_participants(analysis.missing)}",
                f"  Required satisfied: {'Yes' if analysis.required_satisfied else 'No'}",
                f"  Score: {_format_score(analysis.score)}",
            ]
        )
        if missing_required:
            lines.append(f"  Missing required: {', '.join(sorted(missing_required))}")
        lines.append("")

    return "\n".join(lines).rstrip()


def _render_ground_truth(best_slot: str, best_score: float) -> str:
    return "\n".join(
        [
            "==== GROUND TRUTH ====",
            f"Best slot: {best_slot}",
            f"Best score: {_format_score(best_score)}",
        ]
    )


def _render_agent_decision(
    chosen_slot: str,
    valid_action: bool,
    reward: float,
    final_score: float,
    reason: str | None,
    strategy: str,
    note: str | None,
) -> str:
    lines = [
        "==== AGENT DECISION ====",
        f"Agent strategy: {strategy}",
        f"Chosen slot: {chosen_slot}",
        f"Slot validation: {'valid' if valid_action else 'invalid'}",
        f"Reward: {_format_score(reward)}",
        f"Final score: {_format_score(final_score)}",
    ]
    if reason:
        lines.append(f"Validation message: {reason}")
    if note:
        lines.append(f"Agent note: {note}")
    return "\n".join(lines)


def _render_final_evaluation(agent_score: float, optimal_score: float, normalized_score: float) -> str:
    return "\n".join(
        [
            "==== FINAL EVALUATION ====",
            f"agent_score: {_format_score(agent_score)}",
            f"optimal_score: {_format_score(optimal_score)}",
            f"normalized_score: {_format_score(normalized_score)}",
        ]
    )


def _emit_verbose_report(
    task_name: str,
    state: MeetingState,
    best_slot: str,
    optimal_score: float,
    chosen_slot: str,
    valid_action: bool,
    reward: float,
    normalized_score: float,
    validation_reason: str | None,
    selection: AgentSelection,
) -> None:
    if not _verbose_enabled():
        return

    print(f"==== TASK {task_name.upper()} ====", file=sys.stderr, flush=True)
    print(_render_state_summary(state), file=sys.stderr, flush=True)
    print(file=sys.stderr, flush=True)
    print(_render_slot_analysis(state), file=sys.stderr, flush=True)
    print(file=sys.stderr, flush=True)
    print(_render_ground_truth(best_slot, optimal_score), file=sys.stderr, flush=True)
    print(file=sys.stderr, flush=True)
    print(
        _render_agent_decision(
            chosen_slot,
            valid_action,
            reward,
            normalized_score,
            validation_reason,
            selection.strategy,
            selection.note,
        ),
        file=sys.stderr,
        flush=True,
    )
    print(file=sys.stderr, flush=True)
    print("==== EXPLANATION ====", file=sys.stderr, flush=True)
    print(explain_decision(state, chosen_slot), file=sys.stderr, flush=True)
    print(file=sys.stderr, flush=True)
    print(
        _render_final_evaluation(reward, optimal_score, normalized_score),
        file=sys.stderr,
        flush=True,
    )


def _run_task(task_name: str, task_loader: Callable[[], MeetingState]) -> None:
    step_emitted = False

    _emit_structured("START", task=task_name)

    try:
        env = MeetingEnv(task_loader)
        current_state = env.reset()
        selection = _choose_slot(current_state)
        chosen_slot = selection.chosen_slot

        validation_error = _validate_chosen_slot(current_state, chosen_slot)
        if validation_error is None:
            step_result = env.step(Action(chosen_slot=chosen_slot))
            valid_action = bool(step_result["info"]["valid_action"])
            reward = float(step_result["reward"])
            validation_reason = step_result["info"]["reason"]
        else:
            valid_action = False
            reward = 0.0
            validation_reason = validation_error

        best_slot = find_best_slot(current_state)
        optimal_score = compute_score(current_state, best_slot)
        normalized_score = grade(current_state, chosen_slot)

        _emit_structured("STEP", step=1, reward=_format_score(reward))
        step_emitted = True
        _emit_structured("END", task=task_name, score=_format_score(normalized_score), steps=1)

        _emit_verbose_report(
            task_name,
            current_state,
            best_slot,
            optimal_score,
            chosen_slot,
            valid_action,
            reward,
            normalized_score,
            validation_reason,
            selection,
        )
    except Exception as exc:
        if not step_emitted:
            _emit_structured("STEP", step=1, reward=_format_score(0.0))
        _emit_structured("END", task=task_name, score=_format_score(_FALLBACK_SCORE), steps=1)
        print(f"{task_name} inference error: {exc}", file=sys.stderr, flush=True)


def main() -> None:
    for task_name, task_loader in TASK_LOADERS.items():
        _run_task(task_name, task_loader)


if __name__ == "__main__":
    main()
