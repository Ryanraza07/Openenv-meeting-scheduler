from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from .env.action import Action
    from .env.environment import MeetingEnv
    from .tasks.medium import load_medium_task
except ImportError:
    # Fallback for non-package path (for direct execution or legacy envs)
    from app.env.action import Action
    from app.env.environment import MeetingEnv
    from app.tasks.medium import load_medium_task


app = FastAPI(
    title="Meeting Scheduler API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

env = MeetingEnv(load_medium_task)


class ParticipantResponse(BaseModel):
    name: str
    available_slots: List[str]
    priority: float
    required: bool


class StateResponse(BaseModel):
    participants: List[ParticipantResponse]
    all_slots: List[str]
    meeting_duration: int


class StepRequest(BaseModel):
    """Request schema for /step endpoint"""

    chosen_slot: str = Field(min_length=1, description="The chosen time slot for the meeting")


class StepResponse(BaseModel):
    """Response schema for /step endpoint"""

    state: StateResponse
    reward: float
    done: bool
    info: dict


class HealthResponse(BaseModel):
    """Response schema for /health endpoint"""

    status: str = Field(description="Status message")


class RootResponse(BaseModel):
    """Response schema for / endpoint"""

    name: str = Field(description="API name")
    status: str = Field(description="Service status")
    docs_url: str = Field(description="OpenAPI docs URL")
    openapi_url: str = Field(description="OpenAPI schema URL")
    endpoints: List[str] = Field(description="Available API endpoints")


def _ensure_state() -> None:
    if env._current_state is None:
        env.reset()


def _serialize_state() -> dict[str, object]:
    _ensure_state()
    state = env.state()
    return {
        "participants": [participant.model_dump() for participant in state.participants],
        "all_slots": state.all_slots,
        "meeting_duration": state.meeting_duration,
    }


@app.on_event("startup")
async def initialize_environment() -> None:
    _ensure_state()


@app.get("/", response_model=RootResponse, tags=["Health"])
async def root() -> dict[str, object]:
    return {
        "name": "Meeting Scheduler API",
        "status": "ok",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
        "endpoints": ["/health", "/reset", "/step", "/state"],
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> dict[str, str]:
    """
    Health check endpoint.

    Returns:
        HealthResponse: Status of the API
    """
    return {"status": "ok"}


@app.post("/reset", response_model=StateResponse, tags=["Environment"])
async def reset() -> dict[str, object]:
    """
    Reset the meeting scheduler environment.

    Initializes a new scheduling task with participants and time slots.

    Returns:
        StateResponse: Initial environment state
    """
    env.reset()
    return _serialize_state()


@app.get("/state", response_model=StateResponse, tags=["Environment"])
async def state() -> dict[str, object]:
    """
    Return the current environment state.

    Returns:
        StateResponse: Current environment state
    """
    return _serialize_state()


@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def step(request: StepRequest) -> dict[str, object]:
    """
    Execute one step in the meeting scheduler environment.

    Takes a chosen time slot and computes the reward based on participant availability.

    Args:
        request: JSON payload containing the chosen slot

    Returns:
        StepResponse: Result of the step including state, reward, done, and info

    Raises:
        HTTPException: 400 if chosen_slot is invalid
        HTTPException: 500 if reset has not been called
    """
    _ensure_state()

    current_state = env.state()
    if request.chosen_slot not in current_state.all_slots:
        raise HTTPException(status_code=400, detail="chosen_slot is not a valid slot")

    try:
        action = Action(chosen_slot=request.chosen_slot)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {str(e)}")

    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    updated_state = env.state()

    return {
        "state": {
            "participants": [participant.model_dump() for participant in updated_state.participants],
            "all_slots": updated_state.all_slots,
            "meeting_duration": updated_state.meeting_duration,
        },
        "reward": result["reward"],
        "done": result["done"],
        "info": result["info"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
