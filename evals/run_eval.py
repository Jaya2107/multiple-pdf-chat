"""Evaluate the RAG chatbot against a golden dataset and log results to MLflow.

Run from the project root (where app.py is):

    python -m evals.run_eval                                   # easy suite, keyword scoring
    python -m evals.run_eval --judge                           # + LLM-as-judge scores
    python -m evals.run_eval --suite hard --judge              # harder document + questions
    python -m evals.run_eval --suite hard --chunk-size 300 --k 2 --judge
    python -m evals.run_eval --suite hard --mode langgraph --judge

Then compare runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import argparse
import json
import re
import unicodedata
import sys
import textwrap
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REFUSAL_TEXT = "not in the provided documents"
PROMPT_VERSION = "v1"  # bump this when you change the prompt, MLflow will track it

ANSWER_PROMPT = """You are a helpful assistant. Answer the user's question using the context below.

Context:
{context}

Question:
{question}

Instructions:
- Only use information from context
- If not found, say "Not in the provided documents"
- Keep answers short and specific

Answer:"""

JUDGE_PROMPT = """You are a strict evaluator of a question-answering system.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Rate the ANSWER on two scales from 1 (worst) to 5 (best):
- faithfulness: is every claim in the answer supported by the CONTEXT? (a polite
  refusal like "Not in the provided documents" is faithful if the context truly lacks the answer)
- relevance: does the answer actually address the QUESTION?

Reply with ONLY JSON, like: {{"faithfulness": 4, "relevance": 5}}"""

FACTS = [
    "Zorland Country Fact Sheet",
    "The capital of Zorland is Quillon.",
    "The national currency of Zorland is the Zor.",
    "Zorland was founded in the year 1873 by Queen Marwa.",
    "The population of Zorland is approximately 4.2 million people.",
    "The official language of Zorland is Zorish, spoken by 90 percent of citizens.",
    "The highest mountain in Zorland is Mount Vellor, which is 3,450 meters tall.",
    "The main export of Zorland is blue silk.",
    "The river Elmar flows through the capital city Quillon.",
    "The national animal of Zorland is the silver fox.",
    "Independence Day in Zorland is celebrated on 14 March.",
]


def get_suite(name: str):
    """Return (dataset_path, pdf_path, facts) for a suite name."""
    here = Path(__file__).parent
    if name == "hard":
        from evals.hard_facts import HARD_FACTS
        return here / "hard_dataset.json", here / "data" / "zorland_hard.pdf", HARD_FACTS
    return here / "golden_dataset.json", here / "data" / "zorland.pdf", FACTS


# ---------------------------------------------------------------- scoring
def _norm(text: str) -> str:
    """Lowercase, unify odd unicode spaces (e.g. no-break spaces) and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def score_keywords(answer: str, expected: list, should_refuse: bool) -> bool:
    """True if the answer is correct (contains all expected words, or refuses)."""
    text = _norm(answer)
    if should_refuse:
        return REFUSAL_TEXT in text
    return all(_norm(word) in text for word in expected)


def parse_judge_reply(reply: str):
    """Pull (faithfulness, relevance) out of the judge's reply. None if unparsable."""
    try:
        faith = int(re.search(r'"faithfulness"\s*:\s*(\d)', reply).group(1))
        rel = int(re.search(r'"relevance"\s*:\s*(\d)', reply).group(1))
        if 1 <= faith <= 5 and 1 <= rel <= 5:
            return faith, rel
    except (AttributeError, ValueError):
        pass
    return None


def evaluate(dataset, answer_fn, judge_fn=None):
    """answer_fn(question) -> (answer, contexts).  judge_fn(q, contexts, answer) -> reply text."""
    rows = []
    for item in dataset:
        start = time.time()
        try:
            answer, contexts = answer_fn(item["question"])
            error = ""
        except Exception as e:  # e.g. rate limit; don't kill the whole run
            answer, contexts, error = "", [], str(e)
        latency = time.time() - start

        correct = score_keywords(answer, item["expected"], item["should_refuse"])
        faith = rel = None
        if judge_fn and not error:
            try:
                scores = parse_judge_reply(judge_fn(item["question"], contexts, answer))
                if scores:
                    faith, rel = scores
            except Exception:
                pass
        rows.append({
            "id": item["id"], "question": item["question"], "answer": answer,
            "should_refuse": item["should_refuse"], "correct": correct,
            "faithfulness": faith, "relevance": rel,
            "latency_s": round(latency, 2), "error": error,
        })

    def avg(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 3) if values else None

    answerable = [r for r in rows if not r["should_refuse"]]
    refusals = [r for r in rows if r["should_refuse"]]
    summary = {
        "answer_accuracy": avg([float(r["correct"]) for r in answerable]),
        "refusal_accuracy": avg([float(r["correct"]) for r in refusals]),
        "avg_faithfulness": avg([r["faithfulness"] for r in rows]),
        "avg_relevance": avg([r["relevance"] for r in rows]),
        "avg_latency_s": avg([r["latency_s"] for r in rows]),
        "error_count": sum(1 for r in rows if r["error"]),
    }
    return rows, summary


# ---------------------------------------------------------------- MLflow
def log_run(params: dict, summary: dict, rows: list, run_name: str,
            tracking_uri: str = "sqlite:///mlflow.db"):
    import mlflow  # imported here so unit tests don't need mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("rag-chatbot-eval")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in summary.items() if v is not None})
        mlflow.log_dict({"results": rows}, "results.json")
        return run.info.run_id


# ---------------------------------------------------------------- pipeline
def make_pdf(path: Path, lines=None):
    """Write the facts into a PDF. Long lines are wrapped, new pages added as needed."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    lines = FACTS if lines is None else lines
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    y = 800
    for line in lines:
        for part in (textwrap.wrap(line, 90) or [""]):
            if y < 60:
                c.showPage()
                y = 800
            c.drawString(60, y, part)
            y -= 18 if len(lines) > 15 else 28
    c.save()


def build_vectorstore(pdf_path: Path, chunk_size: int, chunk_overlap: int):
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader

    from app import DirectSentenceTransformerEmbeddings

    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(pdf_path)).pages)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    chunks = [c for c in splitter.split_text(text) if len(c.strip()) > 20]
    return FAISS.from_texts(chunks, DirectSentenceTransformerEmbeddings()), len(chunks)


def make_answer_fn(mode: str, vectorstore, llm, k: int):
    if mode == "langgraph":
        from langgraph_rag import get_langgraph_chain
        graph = get_langgraph_chain(vectorstore, llm)

        def answer_fn(question):
            res = graph.invoke({"question": question, "retry_count": 0,
                                "documents": [], "answer": "", "k": k})
            return res["answer"], [d.page_content for d in res["documents"]]
        return answer_fn

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def answer_fn(question):
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        resp = llm.invoke(ANSWER_PROMPT.format(context=context, question=question))
        return getattr(resp, "content", str(resp)), [d.page_content for d in docs]
    return answer_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["easy", "hard"], default="easy")
    parser.add_argument("--mode", choices=["standard", "langgraph"], default="standard")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--judge", action="store_true", help="also run LLM-as-judge scoring")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    from app import get_llm

    dataset_path, pdf_path, facts = get_suite(args.suite)
    make_pdf(pdf_path, facts)  # always rebuild so the PDF matches the facts
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    llm, llm_label = get_llm()
    vectorstore, n_chunks = build_vectorstore(pdf_path, args.chunk_size, args.chunk_overlap)
    answer_fn = make_answer_fn(args.mode, vectorstore, llm, args.k)

    judge_fn = None
    if args.judge:
        def judge_fn(question, contexts, answer):
            resp = llm.invoke(JUDGE_PROMPT.format(
                context="\n\n".join(contexts)[:3000], question=question, answer=answer))
            return getattr(resp, "content", str(resp))

    print(f"Running {len(dataset)} questions | suite={args.suite} | {llm_label} | mode={args.mode} "
          f"| chunk_size={args.chunk_size} | k={args.k} | judge={args.judge} | chunks={n_chunks}")
    rows, summary = evaluate(dataset, answer_fn, judge_fn)

    params = {
        "suite": args.suite, "mode": args.mode, "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap, "k": args.k,
        "embedding_model": "all-MiniLM-L6-v2", "llm": llm_label,
        "prompt_version": PROMPT_VERSION, "judge_enabled": args.judge, "n_chunks": n_chunks,
    }
    run_name = f"{args.suite}-{args.mode}-cs{args.chunk_size}-k{args.k}"
    run_id = log_run(params, summary, rows, run_name)

    print("\n--- RESULTS ---")
    for key, value in summary.items():
        print(f"{key:18}: {value}")
    failed = [r for r in rows if not r["correct"]]
    if failed:
        print("\nWrong answers:")
        for r in failed:
            print(f"  {r['id']}: {r['question']}\n      -> {r['answer'][:120]}")
    print(f"\nLogged to MLflow (run {run_id[:8]}). Compare runs with:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()