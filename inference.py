from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from dotenv import load_dotenv

from app.agent.baseline_agent import choose_best_slot
from app.env.action import Action
from app.env.environment import MeetingEnv
from app.env.reward import compute_score
from app.env.state import MeetingState, Participant
from app.tasks.grader import find_best_slot, grade
from app.tasks.medium import load_medium_task

_DISPLAY_PRECISION = Decimal("0.0001")


@dataclass(frozen=True)
class SlotAnalysis:
    slot: str
    attending: tuple[Participant, ...]
    missing: tuple[Participant, ...]
    covered_priority: Decimal
    total_priority: Decimal
    required_satisfied: bool
    score: float


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _format_decimal(value: Decimal) -> str:
    quantized = value.quantize(_DISPLAY_PRECISION, rounding=ROUND_HALF_UP)
    return f"{quantized:.4f}"


def _format_score(value: float) -> str:
    return _format_decimal(_decimal(value))


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


def _render_agent_decision(chosen_slot: str, valid_action: bool, reward: float, final_score: float, reason: str | None) -> str:
    lines = [
        "==== AGENT DECISION ====",
        f"Chosen slot: {chosen_slot}",
        f"Slot validation: {'valid' if valid_action else 'invalid'}",
        f"Reward: {_format_score(reward)}",
        f"Final score: {_format_score(final_score)}",
    ]
    if reason:
        lines.append(f"Validation message: {reason}")
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


def main() -> None:
    load_dotenv()
    debug = True

    env = MeetingEnv(load_medium_task)
    current_state = env.reset()
    chosen_slot = choose_best_slot(current_state, validate=False)

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

    print(_render_state_summary(current_state))
    print()

    if debug:
        print(_render_slot_analysis(current_state))
        print()

    print(_render_ground_truth(best_slot, optimal_score))
    print()
    print(_render_agent_decision(chosen_slot, valid_action, reward, normalized_score, validation_reason))
    print()
    print("==== EXPLANATION ====")
    print(explain_decision(current_state, chosen_slot))
    print()
    print(_render_final_evaluation(reward, optimal_score, normalized_score))


if __name__ == "__main__":
    main()
