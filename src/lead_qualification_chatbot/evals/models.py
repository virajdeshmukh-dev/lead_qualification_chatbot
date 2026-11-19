from pydantic import BaseModel, Field


class Turn(BaseModel):
    say: str
    step: str | None = None  # expected step after this turn
    route: str | None = None
    contains: list[str] = Field(default_factory=list)
    not_contains: list[str] = Field(default_factory=list)
    contains_digits: list[str] = Field(default_factory=list)  # compared digits-only
    contains_any: list[str] = Field(default_factory=list)  # at least one must appear


class Scenario(BaseModel):
    id: str
    lane: str
    description: str
    turns: list[Turn]
    needs_llm: bool = False
    closing: str | None = None  # expected closing code after the last turn
    qualified: bool | None = None
    done: bool | None = None
