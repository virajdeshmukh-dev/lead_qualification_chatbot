from dataclasses import dataclass

from lead_qualification_chatbot.data.repositories import PropertyRepository, SellerCoverage
from lead_qualification_chatbot.domain.enums import ClosingCode, ListingCategory
from lead_qualification_chatbot.domain.models import Property

NEW_BUILD_MIN_BUDGET = 1_000_000
MAX_MATCHES = 5


@dataclass(frozen=True, slots=True)
class Outcome:
    closing: ClosingCode | None = None
    matches: tuple[Property, ...] = ()

    @property
    def qualified(self) -> bool:
        return self.closing is None


def new_build_budget_ok(budget: int) -> bool:
    """A2 (and C6)."""
    return budget >= NEW_BUILD_MIN_BUDGET


def _top(properties: list[Property]) -> tuple[Property, ...]:
    ordered = sorted(properties, key=lambda p: (p.price_gbp, p.property_id))
    return tuple(ordered[:MAX_MATCHES])


def qualify_new_build(budget: int, postcode: str, repo: PropertyRepository) -> Outcome:
    """A2 -> A4 -> A5."""
    if not new_build_budget_ok(budget):
        return Outcome(closing=ClosingCode.M1)
    listed = repo.at_postcode(postcode, ListingCategory.NEW_BUILD)
    if not listed:
        return Outcome(closing=ClosingCode.M2)
    within = [p for p in listed if p.is_available and p.price_gbp <= budget]
    if not within:
        return Outcome(closing=ClosingCode.M3)
    return Outcome(matches=_top(within))


def qualify_resale(postcode: str, repo: PropertyRepository) -> Outcome:
    """B3 -> B4. Budget is recorded elsewhere, never used as a filter."""
    listed = repo.at_postcode(postcode, ListingCategory.RESALE)
    if not listed:
        return Outcome(closing=ClosingCode.M2)
    available = [p for p in listed if p.is_available]
    if not available:
        return Outcome(closing=ClosingCode.M3)
    return Outcome(matches=_top(available))


def qualify_seller(postcode: str, sellers: SellerCoverage) -> Outcome:
    """D2."""
    return Outcome() if sellers.covers(postcode) else Outcome(closing=ClosingCode.M2)
