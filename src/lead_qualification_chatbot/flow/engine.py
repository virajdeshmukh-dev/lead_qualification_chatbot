from collections.abc import Callable

from lead_qualification_chatbot.data.repositories import DataStore
from lead_qualification_chatbot.domain import messages as msg
from lead_qualification_chatbot.domain.enums import ClosingCode, Intent, ListingCategory, Step
from lead_qualification_chatbot.domain.models import Property
from lead_qualification_chatbot.exceptions import InvalidInputError
from lead_qualification_chatbot.flow import parsing, rules
from lead_qualification_chatbot.flow.types import Extraction, FlowResult, FlowState, Slots
from lead_qualification_chatbot.logging_config import get_logger
from lead_qualification_chatbot.validation.budget import parse_budget
from lead_qualification_chatbot.validation.contact import (
    find_email_candidate,
    find_phone_candidate,
    validate_email_address,
    validate_phone,
)
from lead_qualification_chatbot.validation.postcode import extract_postcode, normalize_postcode

log = get_logger(__name__)


Handler = Callable[[FlowState, str, Extraction, list[str]], None]


class FlowEngine:
    """Flowchart lanes A-E. Pure and deterministic: no LLM, no I/O."""

    def __init__(self, data: DataStore, office_phone: str) -> None:
        self._data = data
        self._phone = office_phone
        self._handlers: dict[Step, Handler] = {
            Step.HOME: self._on_home,
            Step.ASK_NAME: self._on_name,
            Step.BUY_OR_SELL: self._on_buy_or_sell,
            Step.SPECIFIC_PROPERTY: self._on_specific,
            Step.NEWBUILD_OR_RESALE: self._on_category,
            Step.A1_BUDGET: self._on_budget,
            Step.B1_BUDGET: self._on_budget,
            Step.C5_BUDGET: self._on_budget,
            Step.A3_POSTCODE: self._on_postcode,
            Step.B2_POSTCODE: self._on_postcode,
            Step.D1_POSTCODE: self._on_postcode,
            Step.C1_PROPERTY: self._on_property,
            Step.C_NOT_FOUND: self._on_property,
            Step.C_PICK: self._on_pick,
            Step.E1_PHONE: self._on_phone,
            Step.E2_EMAIL: self._on_email,
            Step.E3_ANYTHING_ELSE: self._on_anything_else,
        }

    # ---- public API
    def start(self) -> FlowResult:
        return FlowResult(state=FlowState(), messages=(msg.PROMPTS[Step.HOME],))

    def prompt(self, state: FlowState) -> str:
        """The question the visitor is currently expected to answer (for resuming)."""
        if state.step is Step.DONE:
            return msg.ended(self._phone)
        if state.step is Step.C_PICK:
            return msg.pick_prompt(self._candidates(state))
        return msg.PROMPTS[state.step]

    def step(self, state: FlowState, text: str, extraction: Extraction | None = None) -> FlowResult:
        s = state.model_copy(deep=True)
        x = extraction or Extraction()
        out: list[str] = []
        before = s.step
        if s.step is Step.DONE:
            out.append(msg.ended(self._phone))
        elif not self._maybe_switch(s, x, out):
            self._handlers[s.step](s, text.strip(), x, out)
        log.info(
            "flow_step",
            step_from=before.value,
            step_to=s.step.value,
            qualified=s.qualified,
            closing=s.closing.value if s.closing else None,
        )
        return FlowResult(state=s, messages=tuple(out))

    # ---- helpers
    def _ask(self, s: FlowState, step: Step, out: list[str]) -> None:
        s.step = step
        out.append(self.prompt(s))

    def _retry(self, s: FlowState, out: list[str], reason: str = msg.RETRY) -> None:
        out.append(reason)
        self._ask(s, s.step, out)

    def _candidates(self, s: FlowState) -> tuple[Property, ...]:
        found = (self._data.properties.get(pid) for pid in s.candidate_ids)
        return tuple(p for p in found if p is not None)

    @staticmethod
    def _reset_journey(s: FlowState) -> None:
        keep = s.slots
        s.slots = Slots(name=keep.name, phone=keep.phone, email=keep.email)
        s.candidate_ids = ()
        s.not_found_count = 0
        s.closing = None
        s.qualified = False
        s.match_ids = ()
        s.lead_submitted = False

    @staticmethod
    def _enrich(x: Extraction, text: str) -> Extraction:
        """LLM values win; regex fills the gaps."""
        return x.model_copy(
            update={
                "intent": x.intent or parsing.parse_intent(text),
                "category": x.category or parsing.parse_category(text),
            }
        )

    @staticmethod
    def _apply(s: FlowState, x: Extraction) -> None:
        """Take anything the visitor volunteered, without overwriting known slots."""
        sl = s.slots
        sl.intent = sl.intent or x.intent
        if x.property_query:
            sl.property_query, sl.wants_specific = x.property_query, True
            sl.intent = sl.intent or Intent.BUY
        if x.wants_specific is not None and sl.wants_specific is None:
            sl.wants_specific = x.wants_specific
        sl.category = sl.category or x.category

    def _maybe_switch(self, s: FlowState, x: Extraction, out: list[str]) -> bool:
        """Visitor changed their mind mid-journey (needs LLM extraction)."""
        sl = s.slots
        if s.step in (Step.HOME, Step.ASK_NAME):
            return False
        if x.intent and sl.intent and x.intent is not sl.intent:
            self._reset_journey(s)
            self._apply(s, x)
            self._route(s, out)
            return True
        if x.category and sl.category and x.category is not sl.category:
            sl.category, sl.wants_specific, sl.property_query = x.category, False, None
            s.candidate_ids, s.not_found_count, s.qualified = (), 0, False
            s.lead_submitted = False
            self._route(s, out)
            return True
        return False

    # ---- routing: decides the next question from filled slots
    def _route(self, s: FlowState, out: list[str]) -> None:
        sl = s.slots
        if not sl.name:
            self._ask(s, Step.ASK_NAME, out)
        elif sl.intent is None:
            self._ask(s, Step.BUY_OR_SELL, out)
        elif sl.intent is Intent.SELL:
            self._route_seller(s, out)
        else:
            self._route_buyer(s, out)

    def _route_seller(self, s: FlowState, out: list[str]) -> None:
        if s.slots.postcode is None:
            self._ask(s, Step.D1_POSTCODE, out)
        else:
            self._finish(s, out, rules.qualify_seller(s.slots.postcode, self._data.sellers))

    def _route_buyer(self, s: FlowState, out: list[str]) -> None:
        sl = s.slots
        if sl.wants_specific is None:
            self._ask(s, Step.SPECIFIC_PROPERTY, out)
        elif sl.wants_specific:
            if sl.property_query:
                self._lookup(s, out)
            else:
                self._ask(s, Step.C1_PROPERTY, out)
        else:
            self._route_lane(s, out)

    def _route_lane(self, s: FlowState, out: list[str]) -> None:
        sl = s.slots
        repo = self._data.properties
        if sl.category is None:
            self._ask(s, Step.NEWBUILD_OR_RESALE, out)
        elif sl.category is ListingCategory.NEW_BUILD:
            if sl.budget is None:
                self._ask(s, Step.A1_BUDGET, out)
            elif not rules.new_build_budget_ok(sl.budget):
                self._close(s, out, ClosingCode.M1)
            elif sl.postcode is None:
                self._ask(s, Step.A3_POSTCODE, out)
            else:
                self._finish(s, out, rules.qualify_new_build(sl.budget, sl.postcode, repo))
        elif sl.budget is None:
            self._ask(s, Step.B1_BUDGET, out)
        elif sl.postcode is None:
            self._ask(s, Step.B2_POSTCODE, out)
        else:
            self._finish(s, out, rules.qualify_resale(sl.postcode, repo))

    def _finish(self, s: FlowState, out: list[str], outcome: rules.Outcome) -> None:
        if outcome.closing is not None:
            self._close(s, out, outcome.closing)
            return
        if outcome.matches:
            s.match_ids = tuple(p.property_id for p in outcome.matches)
            out.append(msg.format_matches(s.slots.name, outcome.matches))
        self._qualified(s, out)

    def _close(self, s: FlowState, out: list[str], code: ClosingCode) -> None:
        out.append(msg.closing_message(code, self._phone))
        s.closing, s.step = code, Step.DONE

    def _qualified(self, s: FlowState, out: list[str]) -> None:
        """Lane E. Contact details already collected are never asked twice."""
        s.qualified = True
        if not s.slots.phone:
            self._ask(s, Step.E1_PHONE, out)
        elif not s.slots.email:
            self._ask(s, Step.E2_EMAIL, out)
        else:
            self._ask(s, Step.E3_ANYTHING_ELSE, out)

    # ---- lane C
    def _lookup(self, s: FlowState, out: list[str]) -> None:
        found = self._data.properties.find(s.slots.property_query or "")
        self._resolve(s, found, out)

    def _resolve(self, s: FlowState, found: tuple[Property, ...], out: list[str]) -> None:
        if not found:
            s.not_found_count += 1
            s.candidate_ids, s.slots.property_query = (), None
            if s.not_found_count == 1:  # ask once more
                self._ask(s, Step.C_NOT_FOUND, out)
            else:  # continue as a general buyer
                s.slots.wants_specific, s.not_found_count = False, 0
                self._route(s, out)
        elif len(found) == 1:
            self._found(s, found[0], out)
        else:
            s.candidate_ids = tuple(p.property_id for p in found)
            self._ask(s, Step.C_PICK, out)

    def _found(self, s: FlowState, prop: Property, out: list[str]) -> None:
        sl = s.slots
        s.candidate_ids, s.not_found_count = (), 0
        sl.property_id, sl.property_query = prop.property_id, None
        if prop.is_available:  # C3 yes -> C4
            out.append(msg.property_details(prop))
            s.match_ids = (prop.property_id,)
            self._qualified(s, out)
            return
        out.append(msg.property_gone(prop))  # C5
        sl.category = prop.listing_category
        if sl.budget is None:
            self._ask(s, Step.C5_BUDGET, out)
        else:
            sl.wants_specific = False
            self._route(s, out)

    # ---- step handlers
    def _on_home(self, s: FlowState, text: str, x: Extraction, out: list[str]) -> None:
        x = self._enrich(x, text)
        s.slots.name = x.name or parsing.parse_name(text, strict=True)
        self._apply(s, x)
        self._route(s, out)

    def _on_name(self, s: FlowState, text: str, x: Extraction, out: list[str]) -> None:
        name = x.name or parsing.parse_name(text, strict=False)
        if not name:
            self._retry(s, out, "Sorry, I didn't catch your name.")
            return
        s.slots.name = name
        out.append(f"Nice to meet you, {name}.")
        self._apply(s, self._enrich(x, text))
        self._route(s, out)

    def _on_buy_or_sell(self, s: FlowState, text: str, x: Extraction, out: list[str]) -> None:
        x = self._enrich(x, text)
        if x.intent is None:
            self._retry(s, out)
            return
        self._apply(s, x)
        self._route(s, out)

    def _on_specific(self, s: FlowState, text: str, x: Extraction, out: list[str]) -> None:
        if x.property_query:
            s.slots.property_query, s.slots.wants_specific = x.property_query, True
        else:
            answer = x.wants_specific if x.wants_specific is not None else x.answer
            answer = answer if answer is not None else parsing.parse_yes_no(text)
            if answer is None:
                self._retry(s, out)
                return
            s.slots.wants_specific = answer
        self._route(s, out)

    def _on_category(self, s: FlowState, text: str, x: Extraction, out: list[str]) -> None:
        category = x.category or parsing.parse_category(text)
        if category is None:
            self._retry(s, out)
            return
        s.slots.category = category
        self._route(s, out)

    def _on_budget(self, s: FlowState, text: str, x: Extraction, out: list[str]) -> None:
        try:
            s.slots.budget = parse_budget(str(x.budget) if x.budget is not None else text)
        except InvalidInputError as exc:
            self._retry(s, out, exc.message)
            return
        if s.step is Step.C5_BUDGET:
            s.slots.wants_specific = False  # now re-qualified through lane A or B
        self._route(s, out)

    def _on_postcode(self, s: FlowState, text: str, x: Extraction, out: list[str]) -> None:
        try:
            s.slots.postcode = extract_postcode(text) or normalize_postcode(text)
        except InvalidInputError as exc:
            self._retry(s, out, exc.message)
            return
        self._route(s, out)

    def _on_property(self, s: FlowState, text: str, x: Extraction, out: list[str]) -> None:
        query = x.property_query or text
        if len(query) < 3:
            self._retry(s, out)
            return
        s.slots.property_query = query
        self._lookup(s, out)

    def _on_pick(self, s: FlowState, text: str, x: Extraction, out: list[str]) -> None:
        candidates = self._candidates(s)
        n = parsing.parse_ordinal(text)
        if n is not None and 1 <= n <= len(candidates):
            self._found(s, candidates[n - 1], out)
            return
        found = self._data.properties.find(x.property_query or text)
        narrowed = tuple(p for p in found if p.property_id in s.candidate_ids)
        self._resolve(s, narrowed or found, out)

    def _on_phone(self, s: FlowState, text: str, x: Extraction, out: list[str]) -> None:
        try:
            s.slots.phone = validate_phone(find_phone_candidate(text) or text)
        except InvalidInputError as exc:
            self._retry(s, out, exc.message)
            return
        self._qualified(s, out)

    def _on_email(self, s: FlowState, text: str, x: Extraction, out: list[str]) -> None:
        try:
            s.slots.email = validate_email_address(find_email_candidate(text) or text)
        except InvalidInputError as exc:
            self._retry(s, out, exc.message)
            return
        self._qualified(s, out)

    def _on_anything_else(self, s: FlowState, text: str, x: Extraction, out: list[str]) -> None:
        x = self._enrich(x, text)
        answer = x.answer if x.answer is not None else parsing.parse_yes_no(text)
        intent = x.intent
        if answer is None and intent is None:
            self._retry(s, out)
        elif intent is None and answer is False:
            out.append(msg.GOODBYE)
            s.step = Step.DONE
        else:  # reassist: back to "Buy or sell?", details kept
            self._reset_journey(s)
            if intent is None:
                out.append(msg.REASSIST)
                s.step = Step.BUY_OR_SELL
            else:
                self._apply(s, x)
                self._route(s, out)
