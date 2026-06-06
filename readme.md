# 🤖 Portfolio Assistant

A RAG (Retrieval-Augmented Generation) chatbot that answers recruiter questions based on your CV. Upload your CV once, ask anything — the bot finds the relevant parts and answers professionally using Mistral AI.

---

## ✨ Features

- 📄 Supports PDF and TXT CV formats
- 🔍 Semantic search — finds the *meaning* behind questions, not just keywords
- 💾 Saves embeddings to disk so startup is instant after the first run
- 🖥️ Two interfaces — terminal for quick testing, Streamlit for a proper chat UI
- 🆓 Fully free — local embeddings (no API needed) + Mistral free tier

---

## 🏗️ Tech Stack

| Tool | Purpose |
|------|---------|
| [LangChain](https://python.langchain.com/) | Orchestrates the RAG pipeline |
| [Mistral AI](https://mistral.ai/) | LLM that generates the answers |
| [FAISS](https://github.com/facebookresearch/faiss) | Local vector store for semantic search |
| [HuggingFace sentence-transformers](https://www.sbert.net/) | Free local embedding model |
| [Streamlit](https://streamlit.io/) | Web UI |

---

## 📂 Project Structure

```
cv-ai/
├── chatbot.py        # Terminal version
├── app.py            # Streamlit web UI
├── requirements.txt  # Python dependencies
├── .env              # Your API key (never commit this!)
└── cv_vectorstore/   # Auto-created after first run — saved embeddings
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/cv-ai.git
cd cv-ai
```

### 2. Create and activate a virtual environment

```bash
uv venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt --link-mode=copy
```

### 4. Get a free Mistral API key

Go to [console.mistral.ai](https://console.mistral.ai), sign up, and create an API key.

### 5. Add your API key

Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your_api_key_here
```

### 6. Add your CV

Drop your CV (PDF or TXT) into the project folder and update the filename in `chatbot.py`:

```python
CV_FILE = "your_cv.pdf"
```

---

## 💬 Usage

### Terminal version

```bash
python chatbot.py
```

```
📄 Loading CV from: your_cv.pdf
   ✅ Loaded 1 page(s)
   ✅ Split into 8 chunks
🔢 Loading embedding model...
💾 Saved vector store to 'cv_vectorstore' folder

══════════════════════════════════════════════════
  🤖 CV Chatbot is ready!
  Ask anything about the candidate.
  Type 'exit' to quit.
══════════════════════════════════════════════════

You: Does the candidate know Python?
Bot: Yes, the candidate has 3 years of Python experience...

You: exit
Bot: Goodbye! 👋
```

> **Note:** The first run downloads the embedding model (~90MB) and builds the vector store. Every run after that loads instantly from disk.

### Streamlit UI

```bash
streamlit run app.py
```

Opens in your browser at `http://localhost:8501`. Upload your CV in the sidebar, click **Process CV**, then start chatting.

---

## 🧠 How It Works

```
Your CV
   ↓
Split into chunks (~500 chars each)
   ↓
Each chunk converted to a vector (list of numbers representing meaning)
   ↓
Vectors saved in FAISS (local search index)

--- at query time ---

Recruiter asks a question
   ↓
Question converted to a vector
   ↓
FAISS finds the 3 most similar CV chunks
   ↓
Chunks + question sent to Mistral
   ↓
Mistral answers based only on those chunks
```

---

## ⚙️ Configuration

You can tweak these values in `chatbot.py` to tune performance:

| Setting | Default | What it does |
|---------|---------|-------------|
| `chunk_size` | `500` | Characters per chunk — smaller = more precise retrieval |
| `chunk_overlap` | `100` | Overlap between chunks — prevents cutting sentences |
| `k` in `search_kwargs` | `3` | Number of chunks retrieved per question |
| `temperature` | `0.2` | LLM creativity — lower = more factual |

---

## 🔒 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MISTRAL_API_KEY` | ✅ Yes | Your Mistral API key from console.mistral.ai |

---