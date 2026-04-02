from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Participant(BaseModel):
    name: str = Field(min_length=1)
    available_slots: list[str] = Field(default_factory=list)
    priority: float = Field(ge=0.0, le=1.0)
    required: bool = False


class MeetingState(BaseModel):
    participants: list[Participant] = Field(min_length=1)
    all_slots: list[str] = Field(min_length=1)
    meeting_duration: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_participant_slots(self) -> "MeetingState":
        valid_slots = set(self.all_slots)
        for participant in self.participants:
            unknown_slots = [slot for slot in participant.available_slots if slot not in valid_slots]
            if unknown_slots:
                raise ValueError(
                    f"Participant '{participant.name}' has availability outside all_slots: {unknown_slots}"
                )
        return self
