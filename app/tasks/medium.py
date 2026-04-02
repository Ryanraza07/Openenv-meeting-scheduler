from app.env.state import MeetingState, Participant


def load_medium_task() -> MeetingState:
    all_slots = [
        "2026-04-02T09:00",
        "2026-04-02T10:00",
        "2026-04-02T11:00",
        "2026-04-02T14:00",
    ]
    participants = [
        Participant(
            name="Alice",
            available_slots=["2026-04-02T09:00", "2026-04-02T10:00", "2026-04-02T14:00"],
            priority=0.4,
            required=False,
        ),
        Participant(
            name="Bob",
            available_slots=["2026-04-02T10:00", "2026-04-02T11:00"],
            priority=0.25,
            required=False,
        ),
        Participant(
            name="Carol",
            available_slots=["2026-04-02T10:00", "2026-04-02T14:00"],
            priority=0.2,
            required=False,
        ),
        Participant(
            name="Diego",
            available_slots=["2026-04-02T09:00", "2026-04-02T11:00", "2026-04-02T14:00"],
            priority=0.15,
            required=False,
        ),
    ]
    return MeetingState(participants=participants, all_slots=all_slots, meeting_duration=45)
