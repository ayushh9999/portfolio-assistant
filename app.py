# ─────────────────────────────────────────────
#  CV Chatbot  |  Streamlit UI
#  Run with: streamlit run app.py
# ─────────────────────────────────────────────

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Load API key ───────────────────────────────
load_dotenv()


def get_mistral_api_key():
    """Read the key from Streamlit Cloud secrets or local environment variables."""
    try:
        cloud_key = st.secrets.get("MISTRAL_API_KEY")
    except FileNotFoundError:
        cloud_key = None

    return cloud_key or os.getenv("MISTRAL_API_KEY")


MISTRAL_API_KEY = get_mistral_api_key()

# ── Page config ────────────────────────────────
st.set_page_config(page_title="CV Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 CV Chatbot")
st.caption("Upload your CV and let recruiters ask anything about you.")

# ── Helper functions (same logic as chatbot.py) ─

def load_cv(uploaded_file):
    """Save uploaded file temporarily and load it."""
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(temp_path)
    else:
        loader = TextLoader(temp_path, encoding="utf-8")

    documents = loader.load()
    os.remove(temp_path)   # clean up temp file
    return documents


def build_chain(documents):
    """Takes CV documents and returns a ready-to-use chain."""

    if not MISTRAL_API_KEY:
        raise RuntimeError(
            "MISTRAL_API_KEY is not configured. Add it to Streamlit Cloud "
            "Settings > Secrets or to the local .env file."
        )

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Embeddings + vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Prompt
    prompt = PromptTemplate.from_template("""
You are a professional assistant representing a job candidate.
Use ONLY the CV information below to answer the recruiter's question.
Be clear, specific, and professional. If the answer isn't in the CV, say so honestly.

CV Information:
{context}

Recruiter's Question: {question}

Answer:""")

    # Mistral LLM
    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=MISTRAL_API_KEY,
        temperature=0.2,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ── Session state ──────────────────────────────
# st.session_state persists values across reruns (Streamlit reruns on every interaction)

if "messages" not in st.session_state:
    st.session_state.messages = []   # chat history

if "chain" not in st.session_state:
    st.session_state.chain = None    # the RAG chain


# ── Sidebar — CV upload ────────────────────────
with st.sidebar:
    st.header("📄 Upload Your CV")
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file:
        if st.button("⚡ Process CV", use_container_width=True):
            with st.spinner("Reading CV and building vector store..."):
                documents = load_cv(uploaded_file)
                st.session_state.chain = build_chain(documents)
                st.session_state.messages = []   # reset chat on new CV
            st.success(f"✅ CV processed! ({len(documents)} page(s))")

    st.divider()
    st.caption("Built with LangChain + Mistral + FAISS")


# ── Main area — Chat ───────────────────────────
if not st.session_state.chain:
    # No CV uploaded yet — show instructions
    st.info("👈 Upload your CV in the sidebar to get started.")
else:
    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input box (appears at the bottom)
    question = st.chat_input("Ask something about the candidate...")

    if question:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get and show bot answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.chain.invoke(question)
                except Exception as error:
                    response = getattr(error, "response", None)
                    status_code = getattr(response, "status_code", None)
                    if status_code == 401:
                        st.error(
                            "Mistral rejected the API key. Rotate the exposed key "
                            "and update Streamlit Cloud Secrets."
                        )
                    elif status_code == 429:
                        st.error("Mistral rate limit or quota reached. Try again later.")
                    else:
                        st.error(
                            "Mistral could not answer this request. Verify the "
                            "deployment secret and model availability."
                        )
                    st.stop()
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})