# ─────────────────────────────────────────────
#  CV Chatbot  |  LangChain + Mistral + FAISS
# ─────────────────────────────────────────────
#
#  HOW IT WORKS:
#  1. Load your CV (PDF or TXT)
#  2. Split it into small chunks
#  3. Turn chunks into vectors using a LOCAL embedding model (no API needed)
#  4. When a question comes in → find the most relevant chunks
#  5. Send those chunks + the question to Mistral → get answer
#
# ─────────────────────────────────────────────

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings        # free, runs locally
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── 1. Load environment variables (.env file) ──
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("❌ MISTRAL_API_KEY not found. Please add it to your .env file.")


# ── 2. Load your CV ────────────────────────────
def load_cv(file_path: str):
    """Loads a PDF or TXT file and returns a list of document pages."""
    print(f"📄 Loading CV from: {file_path}")

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("❌ Only .pdf and .txt files are supported.")

    documents = loader.load()
    print(f"   ✅ Loaded {len(documents)} page(s)")
    return documents


# ── 3. Split CV into chunks ────────────────────
def split_into_chunks(documents):
    """
    Splits the CV text into smaller overlapping chunks.
    - chunk_size: how many characters per chunk
    - chunk_overlap: characters shared between chunks (avoids cutting mid-sentence)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)
    print(f"   ✅ Split into {len(chunks)} chunks")
    return chunks


# ── 4. Create vector store ─────────────────────
def create_vector_store(chunks):
    """
    Converts text chunks into embeddings using a LOCAL model (no API key needed).
    'all-MiniLM-L6-v2' is small, fast, and great for semantic search.
    Downloads once (~90MB), then cached on your machine.
    """
    print("🔢 Loading embedding model (downloads once ~90MB, then cached)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("   ✅ Embedding model ready")
    print("📦 Building vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("   ✅ Vector store ready")
    return vector_store


# ── 5. Build the QA chain ──────────────────────
def build_qa_chain(vector_store):
    """
    Connects everything using LangChain's LCEL (pipe) syntax:
      retriever | prompt | llm | output_parser
    """

    prompt = PromptTemplate.from_template("""
You are a professional assistant representing a job candidate.
Use ONLY the CV information below to answer the recruiter's question.
Be clear, specific, and professional. If the answer isn't in the CV, say so honestly.

CV Information:
{context}

Recruiter's Question: {question}

Answer:""")

    # mistral-small-latest is free tier and very capable
    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=MISTRAL_API_KEY,
        temperature=0.2,
    )

    # Retriever: fetch top 3 most relevant CV chunks per question
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Format the retrieved documents into a single string for the prompt
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ── 6. Chat loop ───────────────────────────────
def run_chat(chain):
    """Simple terminal chat loop. Type 'exit' to quit."""
    print("\n" + "═" * 50)
    print("  🤖 CV Chatbot is ready!")
    print("  Ask anything about the candidate.")
    print("  Type 'exit' to quit.")
    print("═" * 50 + "\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye! 👋")
            break

        print("Bot: Thinking...", end="\r")
        answer = chain.invoke(question)
        print(f"Bot: {answer}\n")


# ── 7. Main entry point ────────────────────────
if __name__ == "__main__":
    CV_FILE = "abcde (6) (1).pdf"   # ← Change this to your CV filename

    documents    = load_cv(CV_FILE)
    chunks       = split_into_chunks(documents)
    vector_store = create_vector_store(chunks)
    chain        = build_qa_chain(vector_store)

    run_chat(chain)