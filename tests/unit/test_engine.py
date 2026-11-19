from datetime import date

import pytest

from lead_qualification_chatbot.data.repositories import (
    DataStore,
    PropertyRepository,
    SellerCoverage,
)
from lead_qualification_chatbot.domain.enums import (
    ClosingCode,
    Intent,
    ListingCategory,
    ListingStatus,
    Step,
)
from lead_qualification_chatbot.domain.messages import GOODBYE
from lead_qualification_chatbot.domain.models import Property
from lead_qualification_chatbot.flow.engine import FlowEngine
from lead_qualification_chatbot.flow.types import Extraction, FlowState

OFFICE = "020 7946 0100"
NB, RS = ListingCategory.NEW_BUILD, ListingCategory.RESALE
AV, SOLD, OFFER = ListingStatus.AVAILABLE, ListingStatus.SOLD, ListingStatus.UNDER_OFFER


def prop(
    pid: str, name: str, cat: ListingCategory, status: ListingStatus, price: int, postcode: str
) -> Property:
    return Property(
        property_id=pid,
        property_name=name,
        address=f"{name}, 1 Test Street",
        town="London",
        postcode=postcode,
        listing_category=cat,
        property_type="Apartment",
        bedrooms=2,
        bathrooms=1,
        price_gbp=price,
        status=status,
        description="d",
        branch="Chelsea",
        listed_date=date(2026, 1, 1),
    )


@pytest.fixture
def engine() -> FlowEngine:
    props = [
        prop("AR-1", "Apartment 1, Aldermoor Place", NB, AV, 1_500_000, "SW3 3RB"),
        prop("AR-2", "Apartment 2, Aldermoor Place", NB, SOLD, 1_200_000, "SW3 3RB"),
        prop("AR-3", "Apartment 3, Aldermoor Place", NB, AV, 1_400_000, "SW3 3RB"),
        prop("AR-13", "Apartment 13, Aldermoor Place", NB, AV, 1_600_000, "SW3 3RB"),
        prop("NB-9", "Harbour View", NB, OFFER, 2_000_000, "W8 4SB"),
        prop("RS-1", "The Old Rectory", RS, AV, 900_000, "GU5 7AQ"),
        prop("RS-2", "Rose Cottage", RS, OFFER, 700_000, "GU5 7AQ"),
        prop("RS-3", "Oak Lodge", RS, SOLD, 800_000, "OX1 2JD"),
    ]
    store = DataStore(PropertyRepository(props), SellerCoverage(["GU5 7AQ"]), ())
    return FlowEngine(store, OFFICE)


def converse(engine: FlowEngine, *texts: str) -> tuple[FlowState, list[str]]:
    result = engine.start()
    state, log = result.state, list(result.messages)
    for text in texts:
        result = engine.step(state, text)
        state = result.state
        log += result.messages
    return state, log


BUY_NO = ("I want to buy", "Sam", "no")


# ---- lane A
def test_lane_a_happy_path(engine):
    state, log = converse(
        engine,
        *BUY_NO,
        "new-build",
        "£2m",
        "sw3 3rb",
        "07911 123456",
        "sam@gmail.com",
        "no",
    )
    assert state.step is Step.DONE and state.qualified
    assert state.slots.phone == "+447911123456"
    assert state.slots.email == "sam@gmail.com"
    assert log[-1] == GOODBYE
    assert any("Apartment 1, Aldermoor Place" in m for m in log)


def test_lane_a_m1_no_contact_asked(engine):
    state, log = converse(engine, *BUY_NO, "new build", "900k")
    assert state.closing is ClosingCode.M1 and state.step is Step.DONE
    assert OFFICE in log[-1] and state.slots.phone is None
    assert not any("phone number" in m for m in log)


def test_lane_a_m2_postcode(engine):
    state, _ = converse(engine, *BUY_NO, "new-build", "2m", "GU5 7AQ")
    assert state.closing is ClosingCode.M2


def test_lane_a_m3_over_budget(engine):
    state, _ = converse(engine, *BUY_NO, "new-build", "1m", "SW3 3RB")
    assert state.closing is ClosingCode.M3


def test_lane_a_volunteered_category_skips_question(engine):
    state, log = converse(engine, "I want to buy a new build", "Sam", "no")
    assert state.step is Step.A1_BUDGET
    assert not any("new-build or a resale" in m for m in log)


# ---- lane B
def test_lane_b_happy_path(engine):
    state, _ = converse(engine, *BUY_NO, "resale", "£500k", "gu5 7aq")
    assert state.step is Step.E1_PHONE and state.slots.budget == 500_000


def test_lane_b_m2_and_m3(engine):
    assert converse(engine, *BUY_NO, "resale", "1m", "SW3 3RB")[0].closing is ClosingCode.M2
    assert converse(engine, *BUY_NO, "resale", "1m", "OX1 2JD")[0].closing is ClosingCode.M3


# ---- lane D
def test_lane_d_covered_and_not(engine):
    state, _ = converse(engine, "I want to sell", "Sam", "gu5 7aq")
    assert state.step is Step.E1_PHONE and state.qualified
    state, _ = converse(engine, "sell", "Sam", "SW3 3RB")
    assert state.closing is ClosingCode.M2


# ---- lane C
def test_lane_c_available(engine):
    state, log = converse(engine, "buy", "Sam", "yes", "Apartment 1 Aldermoor Place")
    assert state.step is Step.E1_PHONE
    assert any("£1,500,000" in m for m in log)


def test_lane_c_exact_beats_substring(engine):
    state, _ = converse(engine, "buy", "Sam", "yes", "Apartment 3 Aldermoor")
    assert state.step is Step.E1_PHONE and state.slots.property_id == "AR-3"


def test_lane_c_ambiguous_then_pick(engine):
    state, _ = converse(engine, "buy", "Sam", "yes", "Aldermoor Place")
    assert state.step is Step.C_PICK
    state, _ = converse(engine, "buy", "Sam", "yes", "Aldermoor Place", "1")
    assert state.step is Step.E1_PHONE and state.slots.property_id == "AR-1"


def test_lane_c_sold_new_build_low_budget_is_m1(engine):
    state, _ = converse(engine, "buy", "Sam", "yes", "Apartment 2 Aldermoor Place")
    assert state.step is Step.C5_BUDGET
    state, _ = converse(engine, "buy", "Sam", "yes", "Apartment 2 Aldermoor Place", "£900k")
    assert state.closing is ClosingCode.M1


def test_lane_c_sold_new_build_goes_to_a3(engine):
    state, _ = converse(engine, "buy", "Sam", "yes", "Apartment 2 Aldermoor Place", "£2m")
    assert state.step is Step.A3_POSTCODE


def test_lane_c_sold_resale_goes_to_b2_any_budget(engine):
    state, _ = converse(engine, "buy", "Sam", "yes", "Oak Lodge", "£100k")
    assert state.step is Step.B2_POSTCODE and state.slots.budget == 100_000


def test_lane_c_not_found_twice_falls_back(engine):
    state, _ = converse(engine, "buy", "Sam", "yes", "Zebra Mansion")
    assert state.step is Step.C_NOT_FOUND
    state, _ = converse(engine, "buy", "Sam", "yes", "Zebra Mansion", "Zebra Mansion")
    assert state.step is Step.NEWBUILD_OR_RESALE


# ---- input validation: step must not advance
@pytest.mark.parametrize(
    ("script", "step"),
    [
        ((*BUY_NO, "new-build", "lots"), Step.A1_BUDGET),
        (("sell", "Sam", "12345"), Step.D1_POSTCODE),
        (("sell", "Sam", "GU5 7AQ", "12345"), Step.E1_PHONE),
        (("sell", "Sam", "GU5 7AQ", "07911 123456", "nope"), Step.E2_EMAIL),
    ],
)
def test_invalid_input_stays_on_step(engine, script, step):
    state, log = converse(engine, *script)
    assert state.step is step


# ---- E: reassist keeps details
def test_reassist_keeps_details_and_skips_contact(engine):
    state, log = converse(engine, "sell", "Sam", "GU5 7AQ", "07911 123456", "sam@gmail.com", "yes")
    assert state.step is Step.BUY_OR_SELL and "How can I help you?" in log
    assert state.slots.name == "Sam" and state.slots.phone and state.slots.postcode is None
    state, _ = converse(
        engine,
        "sell",
        "Sam",
        "GU5 7AQ",
        "07911 123456",
        "sam@gmail.com",
        "yes",
        "sell",
        "GU5 7AQ",
    )
    assert state.step is Step.E3_ANYTHING_ELSE


# ---- change of mind and resume
def test_switch_buy_to_sell(engine):
    state, _ = converse(engine, *BUY_NO, "new-build")
    result = engine.step(state, "actually I want to sell", Extraction(intent=Intent.SELL))
    assert result.state.step is Step.D1_POSTCODE and result.state.slots.category is None


def test_switch_category(engine):
    state, _ = converse(engine, *BUY_NO, "new-build")
    result = engine.step(state, "actually resale", Extraction(category=RS))
    assert result.state.step is Step.B1_BUDGET


def test_prompt_resumes_current_step(engine):
    state, log = converse(engine, *BUY_NO, "new-build")
    assert engine.prompt(state) == log[-1] == "What is your budget?"


def test_done_state_is_terminal(engine):
    state, _ = converse(engine, *BUY_NO, "new-build", "900k")
    result = engine.step(state, "hello?")
    assert result.state.step is Step.DONE and OFFICE in result.messages[0]
