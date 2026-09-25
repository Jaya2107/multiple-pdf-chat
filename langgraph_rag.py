
"""
langgraph_rag.py

Self-correcting RAG flow built with LangGraph, as an OPTIONAL alternative
to the existing get_conversation_chain() in app.py.

This file does NOT touch your existing code. It reuses the same
vectorstore and get_llm() you already have, and just wraps the
retrieve -> grade -> (re-retrieve if needed) -> generate steps
into a LangGraph graph.

How it works:
1. retrieve_node   -> pulls chunks from FAISS (same as before)
2. grade_node      -> asks the LLM "is this context enough to answer?"
3. conditional edge-> if NOT enough and we haven't retried yet,
                      go back to retrieve with a wider search (k+4)
                      if enough (or already retried once), go to generate
4. generate_node   -> produces the final answer using your existing prompt style

Usage from app.py (does not replace anything, just adds an option):

    from langgraph_rag import get_langgraph_chain

    graph = get_langgraph_chain(vectorstore)
    result = graph.invoke({"question": query, "retry_count": 0})
    answer = result["answer"]
    sources = result["documents"]
"""

from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END


# ---------- Shared state passed between nodes ----------
class RAGState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    retry_count: int
    k: int


def build_graph(vectorstore, llm):
    """
    vectorstore: your existing FAISS vectorstore (from get_vectorstore())
    llm: your existing llm object (from get_llm())
    Both are passed in from app.py so this file has zero duplicate logic.
    """

    # ---- Node 1: Retrieve ----
    def retrieve_node(state: RAGState) -> RAGState:
        k = state.get("k", 4)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(state["question"])
        return {**state, "documents": docs}

    # ---- Node 2: Grade (is retrieved context good enough?) ----
    def grade_node(state: RAGState) -> RAGState:
        context = "\n\n".join(d.page_content for d in state["documents"])
        grading_prompt = (
            "You are grading whether the CONTEXT below is enough to answer "
            "the QUESTION. Reply with exactly one word: 'yes' or 'no'.\n\n"
            f"CONTEXT:\n{context[:3000]}\n\n"
            f"QUESTION:\n{state['question']}\n\n"
            "Is the context sufficient? (yes/no):"
        )
        try:
            verdict = llm.invoke(grading_prompt)
            verdict_text = getattr(verdict, "content", str(verdict)).strip().lower()
        except Exception:
            verdict_text = "yes"  # fail open, don't loop forever if grading errors out

        state["_sufficient"] = verdict_text.startswith("yes")
        return state

    # ---- Conditional edge: decide retrieve again or move to generate ----
    def route_after_grade(state: RAGState) -> str:
        sufficient = state.get("_sufficient", True)
        retry_count = state.get("retry_count", 0)

        if not sufficient and retry_count < 1:
            return "retry"
        return "generate"

    # ---- Node used only on retry: widen the search ----
    def widen_search_node(state: RAGState) -> RAGState:
        return {
            **state,
            "k": state.get("k", 4) + 4,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    # ---- Node 3: Generate final answer ----
    def generate_node(state: RAGState) -> RAGState:
        context = "\n\n".join(d.page_content for d in state["documents"])
        answer_prompt = (
            "You are a helpful assistant. Answer the user's question using "
            "only the context below. If the answer is not in the context, "
            "say \"Not in the provided documents\". Keep the answer short.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{state['question']}\n\n"
            "Answer:"
        )
        response = llm.invoke(answer_prompt)
        answer_text = getattr(response, "content", str(response))
        return {**state, "answer": answer_text}

    # ---- Wire the graph together ----
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("widen_search", widen_search_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"retry": "widen_search", "generate": "generate"},
    )
    graph.add_edge("widen_search", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


def get_langgraph_chain(vectorstore, llm):
    """
    Convenience wrapper: pass in your existing vectorstore and llm
    (both already created in your current app.py), get back a
    compiled LangGraph app you can call with .invoke({...}).
    """
    return build_graph(vectorstore, llm)