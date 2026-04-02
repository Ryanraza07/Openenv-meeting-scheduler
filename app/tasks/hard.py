from app.env.state import MeetingState, Participant


def load_hard_task() -> MeetingState:
    all_slots = [
        "2026-04-03T09:00",
        "2026-04-03T10:00",
        "2026-04-03T11:00",
        "2026-04-03T14:00",
    ]
    participants = [
        Participant(
            name="Nina",
            available_slots=["2026-04-03T09:00", "2026-04-03T10:00"],
            priority=0.3,
            required=True,
        ),
        Participant(
            name="Omar",
            available_slots=["2026-04-03T10:00", "2026-04-03T11:00"],
            priority=0.25,
            required=True,
        ),
        Participant(
            name="Priya",
            available_slots=["2026-04-03T09:00", "2026-04-03T11:00", "2026-04-03T14:00"],
            priority=0.2,
            required=False,
        ),
        Participant(
            name="Quinn",
            available_slots=["2026-04-03T10:00", "2026-04-03T14:00"],
            priority=0.15,
            required=False,
        ),
        Participant(
            name="Rosa",
            available_slots=["2026-04-03T09:00", "2026-04-03T14:00"],
            priority=0.1,
            required=False,
        ),
    ]
    return MeetingState(participants=participants, all_slots=all_slots, meeting_duration=60)
