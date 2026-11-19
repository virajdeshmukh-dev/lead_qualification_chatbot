import pytest

from lead_qualification_chatbot.config import DEFAULT_DATA_DIR
from lead_qualification_chatbot.data.repositories import load_data
from lead_qualification_chatbot.domain.enums import ClosingCode, ListingCategory
from lead_qualification_chatbot.flow.rules import qualify_resale

DATA = DEFAULT_DATA_DIR


@pytest.fixture(scope="module")
def store():
    return load_data(DATA)


def test_counts(store):
    assert len(store.properties.all()) == 247
    assert len(store.sellers) == 100
    assert len(store.faq) == 47


def test_resale_outcome_matches_bruteforce(store):
    for pc in {p.postcode for p in store.properties.all()}:
        rows = [
            p
            for p in store.properties.all()
            if p.postcode == pc and p.listing_category is ListingCategory.RESALE
        ]
        out = qualify_resale(pc, store.properties)
        if not rows:
            assert out.closing is ClosingCode.M2
        elif not any(p.is_available for p in rows):
            assert out.closing is ClosingCode.M3
        else:
            assert out.qualified
            assert all(p.is_available for p in out.matches)
