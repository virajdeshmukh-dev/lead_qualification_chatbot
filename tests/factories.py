from datetime import date

from lead_qualification_chatbot.data.repositories import (
    DataStore,
    PropertyRepository,
    SellerCoverage,
)
from lead_qualification_chatbot.domain.enums import ListingCategory, ListingStatus
from lead_qualification_chatbot.domain.models import FaqEntry, Property


def make_property(
    pid: str = "P1", name: str = "Apartment 1, Aldermoor Place", **kw: object
) -> Property:
    base: dict[str, object] = {
        "property_id": pid,
        "property_name": name,
        "address": f"{name}, 1 Test Street",
        "town": "London",
        "postcode": "SW3 3RB",
        "listing_category": ListingCategory.NEW_BUILD,
        "property_type": "Apartment",
        "bedrooms": 2,
        "bathrooms": 1,
        "price_gbp": 1_500_000,
        "status": ListingStatus.AVAILABLE,
        "description": "A bright flat.",
        "branch": "Chelsea",
        "listed_date": date(2026, 1, 1),
    }
    base.update(kw)
    return Property(**base)  # type: ignore[arg-type]


def make_store() -> DataStore:
    props = [
        make_property("AR-1", "Apartment 1, Aldermoor Place"),
        make_property(
            "RS-1",
            "The Old Rectory",
            listing_category=ListingCategory.RESALE,
            postcode="GU5 7AQ",
            price_gbp=900_000,
        ),
    ]
    return DataStore(PropertyRepository(props), SellerCoverage(["GU5 7AQ"]), tuple(FAQ))


FAQ = [
    FaqEntry(
        faq_id="F1",
        category="hours",
        question="What are your opening hours?",
        answer="We open 9am to 5pm, Monday to Friday.",
    ),
    FaqEntry(
        faq_id="F2",
        category="fees",
        question="How much do you charge to sell a home?",
        answer="Our fee is 1.2% plus VAT.",
    ),
    FaqEntry(
        faq_id="F3",
        category="office",
        question="Where is your head office?",
        answer="In Chelsea, London.",
    ),
]
