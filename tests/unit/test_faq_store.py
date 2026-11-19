import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from lead_qualification_chatbot.domain.models import FaqEntry
from lead_qualification_chatbot.exceptions import DataLoadError
from lead_qualification_chatbot.rag.faq_store import FaqStore
from tests.factories import FAQ


class CountingEmbeddings(Embeddings):
    def __init__(self) -> None:
        self._inner = DeterministicFakeEmbedding(size=256)
        self.doc_calls = 0

    def embed_documents(self, texts):
        self.doc_calls += 1
        return self._inner.embed_documents(texts)

    def embed_query(self, text):
        return self._inner.embed_query(text)


async def make(tmp_path, emb, faq=FAQ, eid="fake:256"):
    return await FaqStore.create(
        faq, emb, embedding_id=eid, persist_dir=tmp_path, top_k=3, max_distance=0.6, attempts=1
    )


async def test_exact_question_is_top_hit(tmp_path):
    store = await make(tmp_path, CountingEmbeddings())
    hits = await store.search("What are your opening hours?")
    assert hits[0].entry.faq_id == "F1" and hits[0].distance < 0.01


async def test_unrelated_query_is_filtered_out(tmp_path):
    store = await make(tmp_path, CountingEmbeddings())
    assert await store.search("zebra quantum marmalade") == []


async def test_index_is_reused_and_rebuilt_only_when_questions_change(tmp_path):
    emb = CountingEmbeddings()
    await make(tmp_path, emb)
    assert emb.doc_calls == 1
    await make(tmp_path, emb)
    assert emb.doc_calls == 1  # same data: no re-embedding
    changed = [FAQ[0].model_copy(update={"question": "When do you open?"}), *FAQ[1:]]
    await make(tmp_path, emb, faq=changed)
    assert emb.doc_calls == 2


async def test_answer_edits_need_no_reindex(tmp_path):
    emb = CountingEmbeddings()
    await make(tmp_path, emb)
    edited = [FAQ[0].model_copy(update={"answer": "New answer."}), *FAQ[1:]]
    store = await make(tmp_path, emb, faq=edited)
    assert emb.doc_calls == 1
    hits = await store.search("What are your opening hours?")
    assert hits[0].entry.answer == "New answer."


async def test_duplicate_ids_rejected(tmp_path):
    dup = [FAQ[0], FaqEntry(faq_id="F1", category="x", question="q", answer="a")]
    with pytest.raises(DataLoadError):
        await make(tmp_path, CountingEmbeddings(), faq=dup)
