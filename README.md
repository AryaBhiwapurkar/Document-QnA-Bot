# Document Q&A Bot

A local RAG (Retrieval-Augmented Generation) application that lets you upload any PDF and ask questions about it — powered by Groq LLM and FAISS vector search.

---

## Demo

> Upload a PDF → Ask questions → Get answers grounded in the document.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Gradio |
| LLM | Groq (`llama-3.1-8b-instant`) |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS |
| Framework | LangChain |

---

## Architecture

```
PDF Upload → PyPDF Loader → Text Splitter → HuggingFace Embeddings → FAISS Vector Store
                                                                              ↓
User Question → Embeddings → FAISS Retrieval → LangChain RAG Chain → Groq LLM → Answer
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/AryaBhiwapurkar/Document-QnA-Bot.git
cd Document-QnA-Bot
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 5. Run the app

```bash
python ui.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## Usage

1. Upload a PDF using the file input
2. Wait for the status to show **"Document ingested successfully!"**
3. Type your question and click **Ask**
4. Get an answer grounded in your document

---

## Project Structure

```
Document-QnA-Bot/
├── ui.py           # Gradio frontend
├── chat.py         # RAG chain setup
├── ingest.py       # PDF loading, chunking, embedding
├── llm.py          # LLM and embeddings configuration
├── requirements.txt
├── .gitignore
└── .env            # Not committed - add your own
```

---

## Key Design Decisions

- **Answers only from document** — the prompt strictly instructs the LLM to use only retrieved context, preventing hallucinations
- **Single embeddings load** — embeddings model is loaded once per session and reused across ingestion and retrieval
- **Graceful error handling** — all errors are caught and shown as friendly messages in the UI

---

## License

MIT
