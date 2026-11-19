import re

from pydantic import BaseModel, Field, field_validator

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class ChatRequest(BaseModel):
    session_id: str | None = Field(
        default=None, description="Omit on the first message; reuse the returned id afterwards."
    )
    message: str = Field(max_length=2000)

    @field_validator("session_id")
    @classmethod
    def _valid_session_id(cls, value: str | None) -> str | None:
        if value is not None and not SESSION_ID_RE.match(value):
            raise ValueError("session_id must be 8-64 characters: letters, digits, _ or -")
        return value


class ChatResponse(BaseModel):
    session_id: str
    messages: list[str]
    step: str
    qualified: bool
    done: bool
    closing: str | None = None
    degraded: bool = False


class SessionResponse(BaseModel):
    session_id: str
    messages: list[str]
