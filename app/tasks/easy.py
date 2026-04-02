from app.env.state import MeetingState, Participant


def load_easy_task() -> MeetingState:
    all_slots = [
        "2026-04-01T09:00",
        "2026-04-01T10:00",
        "2026-04-01T11:00",
    ]
    participants = [
        Participant(
            name="Ava",
            available_slots=["2026-04-01T09:00", "2026-04-01T10:00"],
            priority=0.6,
            required=False,
        ),
        Participant(
            name="Ben",
            available_slots=["2026-04-01T10:00", "2026-04-01T11:00"],
            priority=0.4,
            required=False,
        ),
    ]
    return MeetingState(participants=participants, all_slots=all_slots, meeting_duration=30)
