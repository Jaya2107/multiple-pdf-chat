import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings
import re

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from langchain_community.embeddings import OllamaEmbeddings
    OLLAMA_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OLLAMA_EMBEDDINGS_AVAILABLE = False

# ✅ SentenceTransformer wrapper
class DirectSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, query):
        return self.model.encode([query], convert_to_numpy=True)[0].tolist()

# ✅ Clean PDF text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\"\'\(\)\-]', ' ', text)
    return text.strip()

# ✅ Extract PDF text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            pdf_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    cleaned = clean_text(page_text)
                    pdf_text += f"\n--- Page {page_num+1} ---\n{cleaned}\n"

            if pdf_text.strip():
                text += f"\n=== Document: {pdf.name} ===\n{pdf_text}\n"
                st.info(f"✅ Extracted from {pdf.name}: {len(pdf_text)} characters")
            else:
                st.warning(f"⚠️ No text extracted from {pdf.name}")
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {str(e)}")
    return text

# ✅ Chunk the text
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    return [chunk for chunk in splitter.split_text(text) if len(chunk.strip()) > 50]

# ✅ Create vectorstore
def get_vectorstore(text_chunks):
    try:
        st.write("🔍 Creating embeddings...")
        try:
            embeddings = DirectSentenceTransformerEmbeddings()
            st.write("✅ Using SentenceTransformers")
        except:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
            st.write("✅ Using HuggingFace embeddings")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        st.success("✅ Vector store created")
        return vectorstore
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        return None

# ✅ Build Conversation chain
def get_conversation_chain(vectorstore):
    try:
        model = "llama3:8b"
        llm = Ollama(model=model, temperature=0.1)
        Prompt = PromptTemplate(
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
            input_variables=["context", "question"]
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": Prompt}
        )
        return chain
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None

# ✅ Streamlit UI
def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
    st.title("📚 Multi-PDF Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

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
            chain = get_conversation_chain(vectorstore)
            if chain is None:
                return
            st.session_state.conversation = chain
            st.success("✅ Ready to chat!")

    if st.session_state.conversation:
        query = st.text_input("Ask something from the PDF(s):")
        if query:
            with st.spinner("Answering..."):
                response = st.session_state.conversation({"question": query})
                st.markdown("### 🧠 Answer")
                st.write(response["answer"])
                with st.expander("🔍 Source Chunks"):
                    for doc in response["source_documents"]:
                        st.markdown(doc.page_content[:500] + "...")

if __name__ == "__main__":
    main()
