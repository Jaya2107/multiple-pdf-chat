import os
import re

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings

# Groq retires models regularly. If this one stops working, set GROQ_MODEL
# in Streamlit secrets (no code change needed). Check console.groq.com/docs/models
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


# ---------- Config helper ----------
def get_secret(name, default=None):
    """Read from Streamlit secrets (cloud) or env/.env (local)."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass  # no secrets.toml locally -> fall through to env vars
    return os.getenv(name, default)


# ---------- Embeddings ----------
class DirectSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, query):
        return self.model.encode([query], convert_to_numpy=True)[0].tolist()


@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def load_embeddings():
    # Cached so the model loads once per server, not on every "Process PDFs" click
    return DirectSentenceTransformerEmbeddings()


# ---------- PDF handling ----------
def clean_text(text):
    # Remove control chars, collapse spaces, but KEEP symbols like % / @ $ + =
    # (important for things like lab reports: "5.2 mg/dL", "98%")
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            pdf_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    pdf_text += f"\n--- Page {page_num + 1} ---\n{clean_text(page_text)}\n"

            if pdf_text.strip():
                text += f"\n=== Document: {pdf.name} ===\n{pdf_text}\n"
                st.info(f"Extracted from {pdf.name}: {len(pdf_text)} characters")
            else:
                st.warning(f"No text extracted from {pdf.name} (scanned PDF?)")
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {str(e)}")
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    return [c for c in splitter.split_text(text) if len(c.strip()) > 50]


def get_vectorstore(text_chunks):
    try:
        embeddings = load_embeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        st.success("Vector store created")
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


# ---------- LLM + chain ----------
def get_llm():
    """Groq if an API key is present (deployed), else local Ollama."""
    groq_key = get_secret("GROQ_API_KEY")
    if groq_key:
        from langchain_groq import ChatGroq
        model = get_secret("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        return ChatGroq(model=model, groq_api_key=groq_key, temperature=0.1), f"Groq ({model})"

    from langchain_community.llms import Ollama
    return Ollama(model="llama3:8b", temperature=0.1), "Ollama llama3:8b (local)"


def get_conversation_chain(vectorstore):
    try:
        llm, llm_label = get_llm()

        qa_prompt = PromptTemplate(
            template="""
You are a helpful assistant. Answer the user's question using the context below.

Context:
{context}

Question:
{question}

Instructions:
- Only use information from context
- If not found, say "Not in the provided documents"
- Keep answers short and specific

Answer:""",
            input_variables=["context", "question"],
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
        )
        return chain, llm_label
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None, None


# ---------- UI ----------
def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
    st.title("Multi-PDF Chatbot")
    st.caption("Upload one or more PDFs, click Process, then ask questions. Answers come only from your documents.")

    st.session_state.setdefault("conversation", None)
    st.session_state.setdefault("messages", [])

    pdf_docs = st.file_uploader("Upload PDF(s)", accept_multiple_files=True, type=["pdf"])

    if st.button("Process PDFs"):
        if not pdf_docs:
            st.warning("Please upload PDF(s) first.")
            return
        with st.spinner("Processing..."):
            text = get_pdf_text(pdf_docs)
            if not text:
                st.error("No text found.")
                return
            chunks = get_text_chunks(text)
            vectorstore = get_vectorstore(chunks)
            if vectorstore is None:
                return
            chain, llm_label = get_conversation_chain(vectorstore)
            if chain is None:
                return
            st.session_state.conversation = chain
            st.session_state.messages = []  # fresh chat for new documents
            st.success(f"Ready to chat! LLM: {llm_label}")

    if st.session_state.conversation:
        # Show past messages (chat_input avoids re-running the LLM on every rerun)
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("sources"):
                    with st.expander("Source Chunks"):
                        for s in msg["sources"]:
                            st.markdown(s)

        query = st.chat_input("Ask something from the PDF(s)")
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.spinner("Answering..."):
                    try:
                        response = st.session_state.conversation.invoke({"question": query})
                    except Exception as e:
                        st.error(
                            "Could not get an answer. This can be an API rate limit, "
                            "a missing/invalid API key, or a network issue. "
                            f"Details: {e}"
                        )
                        return

                answer = response["answer"]
                sources = [d.page_content[:500] + "..." for d in response["source_documents"]]
                st.write(answer)
                with st.expander("Source Chunks"):
                    for s in sources:
                        st.markdown(s)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )


if __name__ == "__main__":
    main()