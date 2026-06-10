# 📄 Chat with Your PDF

A RAG (Retrieval Augmented Generation) app that lets you upload any PDF and ask questions about it. Built with LangChain, OpenAI, FAISS, and Streamlit.

## Features

- Upload any PDF file
- Asks questions in natural language
- Answers grounded in the document (no hallucinations from training data)
- Cites the exact page numbers where the answer was found
- Clean chat UI with conversation history

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Gemini 2.5 Flash (Google) |
| Orchestration | LangChain |
| Embeddings | `all-MiniLM-L6-v2` (local, via sentence-transformers) |
| Vector store | FAISS (local, in-memory) |
| UI | Streamlit |

## How It Works

```
PDF → split into chunks → embed chunks → store in FAISS
Question → embed question → retrieve top-4 similar chunks → feed to LLM → answer
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/chat-with-pdf.git
cd chat-with-pdf
```

**2. Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Gemini API key**
```bash
cp .env.example .env
# Edit .env and add your key: GOOGLE_API_KEY=...
```

**5. Run the app**
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Project Structure

```
chat-with-pdf/
├── app.py            # Streamlit UI
├── rag_chain.py      # LangChain RAG logic
├── requirements.txt
├── .env.example      # API key template
└── .gitignore
```

## What I Learned

- How RAG (Retrieval Augmented Generation) works in practice
- LangChain document loaders, text splitters, and retrieval chains
- Vector embeddings and similarity search with FAISS
- Building a stateful chat UI with Streamlit session state
