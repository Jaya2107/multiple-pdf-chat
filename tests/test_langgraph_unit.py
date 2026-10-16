
"""Fast unit tests for the self-correcting RAG graph (langgraph_rag.py).

Uses a FAKE LLM and FAKE vector store: no API key, no Streamlit needed.
"""
import pytest
from langchain_core.documents import Document

from langgraph_rag import get_langgraph_chain


class FakeLLM:
    def __init__(self, grade_reply="yes", answer="Quillon", grader_crashes=False):
        self.grade_reply = grade_reply
        self.answer = answer
        self.grader_crashes = grader_crashes
        self.grade_calls = 0

    def invoke(self, prompt):
        if prompt.startswith("You are grading"):
            self.grade_calls += 1
            if self.grader_crashes:
                raise RuntimeError("rate limit")
            return self.grade_reply
        return self.answer


class FakeVectorStore:
    def as_retriever(self, search_kwargs):
        k = search_kwargs["k"]

        class _R:
            def invoke(self, query):
                return [Document(page_content=f"chunk {i}") for i in range(k)]

        return _R()


def run(llm):
    graph = get_langgraph_chain(FakeVectorStore(), llm)
    return graph.invoke({"question": "capital?", "retry_count": 0, "documents": [], "answer": ""})


def test_sufficient_context_does_not_retry():
    result = run(FakeLLM(grade_reply="yes"))
    assert result["retry_count"] == 0
    assert len(result["documents"]) == 4
    assert result["answer"] == "Quillon"


def test_insufficient_context_widens_search_once():
    result = run(FakeLLM(grade_reply="no"))
    assert result["retry_count"] == 1
    assert len(result["documents"]) == 8
    assert result["answer"] == "Quillon"


def test_never_retries_more_than_once():
    llm = FakeLLM(grade_reply="no")
    result = run(llm)
    assert result["retry_count"] == 1
    assert llm.grade_calls == 2


def test_grader_failure_fails_open_and_still_answers():
    result = run(FakeLLM(grader_crashes=True))
    assert result["retry_count"] == 0
    assert result["answer"] == "Quillon"


@pytest.mark.parametrize("reply", ["Yes", "YES.", "yes, it is enough"])
def test_grader_reply_variants_count_as_yes(reply):
    assert run(FakeLLM(grade_reply=reply))["retry_count"] == 0