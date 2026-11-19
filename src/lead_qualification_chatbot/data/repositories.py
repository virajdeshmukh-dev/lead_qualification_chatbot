import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel, ValidationError
from rapidfuzz import fuzz

from lead_qualification_chatbot.domain.enums import ListingCategory
from lead_qualification_chatbot.domain.models import FaqEntry, Property, SellerPostcode
from lead_qualification_chatbot.exceptions import DataLoadError
from lead_qualification_chatbot.logging_config import get_logger
from lead_qualification_chatbot.validation.postcode import normalize_postcode

log = get_logger(__name__)

PROPERTIES_FILE = "properties_for_buyers.csv"
SELLERS_FILE = "seller_coverage.csv"
FAQ_FILE = "faq.csv"

M = TypeVar("M", bound=BaseModel)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_IGNORE = frozenset(
    {
        "the",
        "a",
        "an",
        "at",
        "in",
        "of",
        "on",
        "please",
        "property",
        "its",
        "it",
        "is",
        "that",
        "this",
        "one",
        "called",
        "apartment",
        "flat",
        "house",
        "home",
        "detached",
        "terraced",
        "semi",
        "townhouse",
        "cottage",
        "penthouse",
    }
)
_FUZZY_CUTOFF = 88


def _norm(text: str) -> str:
    return " ".join(_NON_ALNUM.sub(" ", text.lower()).split())


norm_text = _norm


@dataclass(frozen=True, slots=True)
class _Entry:
    prop: Property
    pid: str
    name: str
    address: str
    tokens: frozenset[str]


def _read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise DataLoadError(f"Data file not found: {path}") from exc
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Cannot parse {path.name}: {exc}") from exc
    df.columns = pd.Index([str(c).strip() for c in df.columns])
    missing = required - set(df.columns)
    if missing:
        raise DataLoadError(f"{path.name} is missing columns: {sorted(missing)}")
    return [{str(k): str(v).strip() for k, v in row.items()} for row in df.to_dict("records")]


def _build[M: BaseModel](model: type[M], rows: list[dict[str, str]], source: str) -> list[M]:
    items: list[M] = []
    errors: list[str] = []
    for line, row in enumerate(rows, start=2):  # line 1 is the header
        try:
            items.append(model.model_validate(row))
        except ValidationError as exc:
            first = exc.errors()[0]
            errors.append(f"line {line}: {first['loc']} {first['msg']}")
    if errors:
        raise DataLoadError(f"{source}: {len(errors)} invalid rows; " + "; ".join(errors[:5]))
    if not items:
        raise DataLoadError(f"{source}: no rows")
    return items


class PropertyRepository:
    """Exact lookups over listings. Callers may pass any postcode casing/spacing."""

    def __init__(self, properties: Iterable[Property]) -> None:
        items = tuple(properties)
        ids = [p.property_id for p in items]
        if len(ids) != len(set(ids)):
            raise DataLoadError("Duplicate property_id values in properties data")
        self._items = items
        self._by_id = {p.property_id: p for p in items}
        index: dict[tuple[ListingCategory, str], list[Property]] = defaultdict(list)
        for p in items:
            index[(p.listing_category, p.postcode)].append(p)
        self._by_postcode = {k: tuple(v) for k, v in index.items()}

        self._entries = tuple(
            _Entry(
                prop=p,
                pid=_norm(p.property_id),
                name=_norm(p.property_name),
                address=_norm(p.address),
                tokens=frozenset(_norm(f"{p.property_name} {p.address}").split()),
            )
            for p in items
        )

    @classmethod
    def from_csv(cls, path: Path) -> "PropertyRepository":
        rows = _read_rows(path, set(Property.model_fields))
        return cls(_build(Property, rows, path.name))

    def all(self) -> tuple[Property, ...]:
        return self._items

    def get(self, property_id: str) -> Property | None:
        return self._by_id.get(property_id)

    def at_postcode(self, postcode: str, category: ListingCategory) -> tuple[Property, ...]:
        """All listings (any status) of a category at a postcode."""
        return self._by_postcode.get((category, normalize_postcode(postcode)), ())

    def find(self, query: str, limit: int = 5) -> tuple[Property, ...]:
        """Resolve free text to listings: exact name/address/id, then token
        containment, then fuzzy. Several results means 'ask which one'."""
        q = _norm(query)
        if len(q) < 3:
            return ()

        def ordered(entries: list[_Entry]) -> tuple[Property, ...]:
            return tuple(sorted((e.prop for e in entries), key=lambda p: p.property_id)[:limit])

        exact = [e for e in self._entries if q in (e.name, e.address, e.pid)]
        if exact:
            return ordered(exact)
        # Generic words ("flat", "house") never identify a property on their own.
        wanted = frozenset(q.split()) - _IGNORE
        if not wanted:
            return ()
        contained = [e for e in self._entries if wanted <= e.tokens]
        if contained:
            return ordered(contained)
        scored = sorted(
            (
                (max(fuzz.partial_ratio(q, e.name), fuzz.partial_ratio(q, e.address)), e)
                for e in self._entries
            ),
            key=lambda t: (-t[0], t[1].prop.property_id),
        )
        return tuple(e.prop for score, e in scored[:limit] if score >= _FUZZY_CUTOFF)


class SellerCoverage:
    def __init__(self, postcodes: Iterable[str]) -> None:
        self._postcodes = frozenset(normalize_postcode(p) for p in postcodes)

    @property
    def postcodes(self) -> frozenset[str]:
        return self._postcodes

    @classmethod
    def from_csv(cls, path: Path) -> "SellerCoverage":
        rows = _read_rows(path, {"postcode"})
        return cls(p.postcode for p in _build(SellerPostcode, rows, path.name))

    def covers(self, postcode: str) -> bool:
        return normalize_postcode(postcode) in self._postcodes

    def __len__(self) -> int:
        return len(self._postcodes)


@dataclass(frozen=True, slots=True)
class DataStore:
    properties: PropertyRepository
    sellers: SellerCoverage
    faq: tuple[FaqEntry, ...]


def load_data(data_dir: Path) -> DataStore:
    """Load and validate every CSV once at startup. Raises DataLoadError."""
    properties = PropertyRepository.from_csv(data_dir / PROPERTIES_FILE)
    sellers = SellerCoverage.from_csv(data_dir / SELLERS_FILE)
    faq_rows = _read_rows(data_dir / FAQ_FILE, set(FaqEntry.model_fields))
    faq = tuple(_build(FaqEntry, faq_rows, FAQ_FILE))
    log.info(
        "data_loaded",
        properties=len(properties.all()),
        seller_postcodes=len(sellers),
        faq=len(faq),
    )
    return DataStore(properties=properties, sellers=sellers, faq=faq)
