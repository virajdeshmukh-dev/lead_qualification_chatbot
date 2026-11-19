from datetime import UTC, date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lead_qualification_chatbot.domain.enums import Intent, ListingCategory, ListingStatus
from lead_qualification_chatbot.exceptions import InvalidInputError
from lead_qualification_chatbot.validation.postcode import normalize_postcode


def _to_postcode(value: object) -> str:
    try:
        return normalize_postcode(str(value))
    except InvalidInputError as exc:
        raise ValueError(exc.message) from exc  # pydantic only catches ValueError


class Property(BaseModel):
    model_config = ConfigDict(frozen=True)

    property_id: str
    property_name: str
    address: str
    town: str
    postcode: str
    listing_category: ListingCategory
    property_type: str
    bedrooms: int = Field(ge=0)
    bathrooms: int = Field(ge=0)
    price_gbp: int = Field(gt=0)
    status: ListingStatus
    description: str
    branch: str
    listed_date: date

    @field_validator("postcode", mode="before")
    @classmethod
    def _normalize_postcode(cls, value: object) -> str:
        return _to_postcode(value)

    @property
    def is_available(self) -> bool:
        return self.status is ListingStatus.AVAILABLE


class FaqEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    faq_id: str
    category: str
    question: str
    answer: str


class SellerPostcode(BaseModel):
    model_config = ConfigDict(frozen=True)

    postcode: str

    @field_validator("postcode", mode="before")
    @classmethod
    def _normalize_postcode(cls, value: object) -> str:
        return _to_postcode(value)


class Lead(BaseModel):
    """A qualified lead handed to the sales team."""

    lead_id: str
    session_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    name: str
    intent: Intent
    category: ListingCategory | None = None
    budget: int | None = None
    postcode: str | None = None
    property_id: str | None = None
    match_ids: tuple[str, ...] = ()
    phone: str
    email: str
