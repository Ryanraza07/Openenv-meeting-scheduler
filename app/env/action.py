from pydantic import BaseModel, Field


class Action(BaseModel):
    chosen_slot: str = Field(min_length=1, description="Meeting slot selected by the agent.")
