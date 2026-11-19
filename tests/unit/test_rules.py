from datetime import date
from types import SimpleNamespace

import pytest
from tenacity import wait_none

from lead_qualification_chatbot.data.repositories import PropertyRepository, SellerCoverage
from lead_qualification_chatbot.domain.enums import (
    ClosingCode,
    ListingCategory,
    ListingStatus,
)
from lead_qualification_chatbot.domain.models import Property
from lead_qualification_chatbot.exceptions import DataLoadError
from lead_qualification_chatbot.flow.rules import (
    qualify_new_build,
    qualify_resale,
    qualify_seller,
)
from lead_qualification_chatbot.llm.resilience import wait_retry_after

PC = "SW3 3RB"


def make(pid: str, **kw: object) -> Property:
    base: dict[str, object] = {
        "property_id": pid,
        "property_name": f"Home {pid}",
        "address": "1 Test St",
        "town": "London",
        "postcode": PC,
        "listing_category": ListingCategory.NEW_BUILD,
        "property_type": "Apartment",
        "bedrooms": 2,
        "bathrooms": 1,
        "price_gbp": 1_500_000,
        "status": ListingStatus.AVAILABLE,
        "description": "d",
        "branch": "Chelsea",
        "listed_date": date(2026, 1, 1),
    }
    base.update(kw)
    return Property(**base)  # type: ignore[arg-type]


def repo(*props: Property) -> PropertyRepository:
    return PropertyRepository(props)


# ---- lane A
def test_a_budget_below_1m_is_m1():
    assert qualify_new_build(999_999, PC, repo(make("1"))).closing is ClosingCode.M1


def test_a_budget_exactly_1m_passes_a2():
    out = qualify_new_build(1_000_000, PC, repo(make("1", price_gbp=1_000_000)))
    assert out.qualified


def test_a_no_new_build_at_postcode_is_m2():
    assert qualify_new_build(2_000_000, "GU1 3AD", repo(make("1"))).closing is ClosingCode.M2


def test_a_only_resale_at_postcode_is_m2():
    r = repo(make("1", listing_category=ListingCategory.RESALE))
    assert qualify_new_build(2_000_000, PC, r).closing is ClosingCode.M2


@pytest.mark.parametrize("status", [ListingStatus.UNDER_OFFER, ListingStatus.SOLD])
def test_a_unavailable_is_m3(status):
    out = qualify_new_build(2_000_000, PC, repo(make("1", status=status)))
    assert out.closing is ClosingCode.M3


def test_a_over_budget_is_m3():
    r = repo(make("1", price_gbp=3_000_000))
    assert qualify_new_build(2_000_000, PC, r).closing is ClosingCode.M3


def test_a_price_equal_to_budget_matches():
    r = repo(make("1", price_gbp=2_000_000))
    assert len(qualify_new_build(2_000_000, PC, r).matches) == 1


def test_a_matches_capped_at_5_sorted_available_only():
    props = [make(str(i), price_gbp=1_000_000 + i * 1000) for i in range(8, 0, -1)]
    props.append(make("sold", status=ListingStatus.SOLD, price_gbp=1_000_000))
    out = qualify_new_build(2_000_000, "sw3 3rb", repo(*props))
    assert [p.property_id for p in out.matches] == ["1", "2", "3", "4", "5"]


# ---- lane B
def test_b_no_minimum_budget_and_none_filter():
    r = repo(make("1", listing_category=ListingCategory.RESALE, price_gbp=260_000))
    assert qualify_resale(PC, r).qualified


def test_b_none_listed_is_m2_even_if_new_build_exists():
    assert qualify_resale(PC, repo(make("1"))).closing is ClosingCode.M2


def test_b_none_available_is_m3():
    r = repo(make("1", listing_category=ListingCategory.RESALE, status=ListingStatus.UNDER_OFFER))
    assert qualify_resale(PC, r).closing is ClosingCode.M3


# ---- lane D
def test_d_seller_coverage():
    cov = SellerCoverage(["GU5 7AQ"])
    assert qualify_seller("gu57aq", cov).qualified
    assert qualify_seller("GU5 7AR", cov).closing is ClosingCode.M2


def test_duplicate_ids_rejected():
    with pytest.raises(DataLoadError):
        repo(make("1"), make("1"))


def _state(exc):
    return SimpleNamespace(outcome=SimpleNamespace(exception=lambda: exc))


class _Err(Exception):
    def __init__(self, headers):
        super().__init__("429")
        self.response = SimpleNamespace(headers=headers)


def test_retry_after_header_is_honoured_and_capped():
    wait = wait_retry_after(wait_none(), cap_s=10)
    assert wait(_state(_Err({"retry-after": "3"}))) == 3.0
    assert wait(_state(_Err({"retry-after": "99"}))) == 10.0
    assert wait(_state(_Err({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}))) == 0
    assert wait(_state(ValueError())) == 0  # no header: fallback
