from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from app.env.state import MeetingState

DEFAULT_MODEL = "gpt-4o-mini"


class SlotDecision(BaseModel):
    chosen_slot: str = Field(min_length=1)


def _build_prompt(state: MeetingState) -> str:
    state_json = state.model_dump_json(indent=2)
    return (
        "Choose the single best meeting slot from all_slots.\n"
        "Rules:\n"
        "1. If any required participant is unavailable for a slot, that slot is invalid and should not be selected.\n"
        "2. Among valid slots, maximize the sum of participant priorities for people available in that slot.\n"
        "3. If multiple slots tie, choose the earliest slot in the order listed in all_slots.\n"
        "4. Return only the chosen slot using the required response schema.\n\n"
        f"Meeting state:\n{state_json}"
    )


def choose_best_slot(state: MeetingState, model: str | None = None, validate: bool = True) -> str:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your environment or .env file.")

    client = OpenAI()
    selected_model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    response = client.responses.parse(
        model=selected_model,
        instructions=(
            "You are a deterministic scheduling agent. "
            "Respect required participants and maximize total priority."
        ),
        input=_build_prompt(state),
        text_format=SlotDecision,
        max_output_tokens=32,
        temperature=0.0,
    )

    decision = response.output_parsed
    if decision is None:
        raise RuntimeError("The OpenAI response could not be parsed into a slot decision.")
    if validate and decision.chosen_slot not in state.all_slots:
        raise ValueError(
            f"Model returned an invalid slot '{decision.chosen_slot}'. Expected one of: {state.all_slots}"
        )

    return decision.chosen_slot
