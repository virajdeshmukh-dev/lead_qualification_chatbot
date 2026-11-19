import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from lead_qualification_chatbot.domain.models import FaqEntry
from lead_qualification_chatbot.exceptions import DataLoadError
from lead_qualification_chatbot.llm.resilience import call_with_retry
from lead_qualification_chatbot.logging_config import get_logger

log = get_logger(__name__)

_CLIENT_SETTINGS = ChromaSettings(anonymized_telemetry=False)


@dataclass(frozen=True, slots=True)
class FaqHit:
    entry: FaqEntry
    distance: float  # cosine distance: 0 identical, ~1 unrelated


class FaqStore:
    """Vector search over FAQ *questions*. Answers always come from the CSV rows."""

    def __init__(
        self,
        store: Chroma,
        entries: Mapping[str, FaqEntry],
        *,
        top_k: int,
        max_distance: float,
        attempts: int,
    ) -> None:
        self._store = store
        self._entries = entries
        self._top_k = top_k
        self._max_distance = max_distance
        self._attempts = attempts

    @classmethod
    async def create(
        cls,
        faq: Sequence[FaqEntry],
        embeddings: Embeddings,
        *,
        embedding_id: str,
        persist_dir: Path | None,
        top_k: int,
        max_distance: float,
        attempts: int,
    ) -> "FaqStore":
        entries = {e.faq_id: e for e in faq}
        if len(entries) != len(faq):
            raise DataLoadError("Duplicate faq_id values in faq data")
        # New questions or a new embedding model => new collection => re-index.
        fingerprint = "\n".join([embedding_id, *(f"{e.faq_id}|{e.question}" for e in faq)])
        digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
        store = Chroma(
            collection_name=f"faq_{digest}",
            embedding_function=embeddings,
            persist_directory=str(persist_dir) if persist_dir else None,
            collection_metadata={"hnsw:space": "cosine"},
            client_settings=_CLIENT_SETTINGS,
        )
        existing = set(store.get()["ids"])
        missing = [e for e in faq if e.faq_id not in existing]
        if missing:
            await call_with_retry(
                lambda: store.aadd_texts(
                    texts=[e.question for e in missing],
                    metadatas=[{"faq_id": e.faq_id} for e in missing],
                    ids=[e.faq_id for e in missing],
                ),
                attempts=attempts,
                op="faq_index",
            )
            log.info("faq_indexed", count=len(missing), collection=f"faq_{digest}")
        return cls(store, entries, top_k=top_k, max_distance=max_distance, attempts=attempts)

    async def search(self, query: str) -> list[FaqHit]:
        raw = await call_with_retry(
            lambda: self._store.asimilarity_search_with_score(query, k=self._top_k),
            attempts=self._attempts,
            op="faq_search",
        )
        hits = [
            FaqHit(entry, float(distance))
            for doc, distance in raw
            if (entry := self._entries.get(str(doc.metadata.get("faq_id")))) is not None
            and distance <= self._max_distance
        ]
        log.info("faq_search", results=len(hits), best=hits[0].distance if hits else None)
        return hits

    async def distances(self, query: str) -> list[tuple[str, float]]:
        """Unfiltered (faq_id, distance), for calibrating FAQ_MAX_DISTANCE."""
        raw = await self._store.asimilarity_search_with_score(query, k=self._top_k)
        return [(str(d.metadata.get("faq_id")), float(s)) for d, s in raw]
