# Eshu

**Local · Private · Multi-Vault · Multimodal RAG System**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


ESHU -- A fully local AI knowledge management system. All models run on your machine via [Ollama](https://ollama.ai). **No data ever leaves your computer.**

---

## ✨ Features

- **Multi-vault knowledge base** — Organize documents into topic-specific vaults (e.g., Legal, Recipes, Travel)
- **Smart query routing** — Phi model automatically routes questions to the right vault(s)
- **Multimodal ingestion** — PDF, images (JPG/PNG/WebP), audio (MP3/WAV/M4A), and text (TXT/MD)
- **Local-first** — 100% offline; documents, embeddings, and answers stay on your machine
- **Conflict detection** — Flags when sources contradict each other
- **Learning system** — Override Phi’s routing; after 3+ corrections, auto-generates rules
- **Streamlit UI** — Chat, Upload, Vaults, Health, and Settings panels

---

## 📋 Prerequisites

| Requirement | Purpose | Windows | macOS | Linux |
|-------------|---------|---------|-------|-------|
| **Python 3.11+** | Runtime | [python.org](https://www.python.org) | `brew install python@3.11` | `apt install python3.11` |
| **Ollama** | Local LLM inference | [ollama.ai](https://ollama.ai) | `brew install ollama` | [ollama.ai](https://ollama.ai) |
| **Tesseract OCR** | PDF/image text extraction | [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) | `brew install tesseract` | `apt install tesseract-ocr` |
| **ffmpeg** | Audio transcription | [ffmpeg.org](https://ffmpeg.org) | `brew install ffmpeg` | `apt install ffmpeg` |

> **Note:** Tesseract and ffmpeg are optional. Without them, PDF/image OCR and audio transcription won’t work, but text ingestion and basic PDF parsing will.

---

## 🚀 Installation & Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/habeeb2023/eshu.git
cd eshu
```

### Step 2: Create a virtual environment (recommended)

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python dependencies

```bash
pip install -r requirements.txt
```

**If `sqlite-vec` fails to install:**

```bash
pip install sqlite-vec --pre
```

### Step 4: Install and run Ollama

1. Download Ollama from [ollama.ai](https://ollama.ai) and install.
2. Start Ollama (it usually runs in the background on startup).
3. Pull the required models:

```bash
ollama pull llama3.1      # Main chat model (answer synthesis)
ollama pull phi3         # Router model (classification)
ollama pull nomic-embed-text   # Embeddings
ollama pull llava        # Vision (for image classification)
```

> **Alternative models:** You can use `mistral`, `llama3.2`, etc. for the main model. Configure in `.env` (see Step 5).

### Step 5: Configure environment

```bash
# Windows (PowerShell)
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Edit `.env` if needed. Defaults work for most setups:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `MAIN_MODEL` | `llama3.1` | Chat model for answer synthesis |
| `ROUTER_MODEL` | `phi3` | Model for vault classification |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `VISION_MODEL` | `llava` | Vision model for images |

### Step 6: Install Tesseract & ffmpeg (optional)

- **Windows:** Install Tesseract from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add it to PATH. Install ffmpeg from [ffmpeg.org](https://ffmpeg.org).
- **macOS:** `brew install tesseract ffmpeg`
- **Linux:** `sudo apt install tesseract-ocr ffmpeg`

---

## ▶️ Running the app

### Option A: Using batch files (Windows)

**Terminal 1 — API backend:**
```bash
start_api.bat
```

**Terminal 2 — UI:**
```bash
start_ui.bat
```

### Option B: Manual commands (all platforms)

**Terminal 1 — API backend:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — UI:**
```bash
streamlit run ui/app.py
```

### Option C: UI only (standalone)

The Streamlit UI can run without the FastAPI backend for basic chat and uploads:

```bash
streamlit run ui/app.py
```

### Access

- **Streamlit UI:** http://localhost:8501  
- **API docs:** http://localhost:8000/docs  

---

## 📁 Project structure

```
eshu/
├── app/
│   ├── config.py           # Central Pydantic settings (.env driven)
│   ├── main.py             # FastAPI application (all endpoints)
│   ├── db/
│   │   ├── init.py         # SQLite + sqlite-vec schema creation
│   │   └── queries.py      # All SQL helpers (insert, search, delete)
│   ├── models/
│   │   └── schemas.py      # Pydantic models (API + Phi output)
│   └── services/
│       ├── ingestion.py    # Parse, chunk, embed, store (PDF/image/audio)
│       ├── router.py       # Phi: vault classification + query routing
│       ├── retrieval.py    # Filter-then-search + cross-vault re-ranking
│       ├── generation.py   # Answer synthesis + citations
│       ├── learning.py     # Correction logging + rule extraction
│       ├── session.py      # In-memory conversation history
│       └── vault_registry.py # registry.json / fingerprint.json / meta.json
├── ui/
│   └── app.py              # Streamlit UI (Chat, Upload, Vaults, Health)
├── data/                   # Created at runtime (not in Git)
│   ├── vault.db            # SQLite database (all vaults)
│   ├── registry.json       # Vault index
│   └── vaults/             # Per-vault fingerprint.json + meta.json
├── logo/
│   └── eshu-logo.svg       # App logo
├── tests/
│   └── test_core.py        # pytest test suite
├── .env.example            # Environment template (copy to .env)
├── requirements.txt        # Python dependencies
├── start_api.bat           # Launch FastAPI backend (Windows)
└── start_ui.bat            # Launch Streamlit UI (Windows)
```

---

## 🏗 Architecture

```
USER  →  Streamlit UI  →  FastAPI  →  Phi (Router)
                                   →  nomic-embed-text (Embedder)
                                   →  SQLite + sqlite-vec (Vector DB)
                                   →  Llama 3.1 (Generator)
                                   →  LLaVA (Vision)
                                   →  Whisper (Audio)
```

| Layer | Model | Responsibility |
|-------|-------|----------------|
| Generation | `MAIN_MODEL` | Answer synthesis, citations, conflict detection |
| Retrieval | `sqlite-vec` | Metadata-filtered cosine similarity search |
| Routing | `phi3` | Vault classification, query routing, rule application |
| Ingestion | `nomic-embed-text` + `llava` + `whisper` | Embed / parse content |

---

## ⚙️ Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `MAIN_MODEL` | `llama3.1` | Generator model |
| `ROUTER_MODEL` | `phi3` | Router/classifier model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `VISION_MODEL` | `llava` | Vision model for images |
| `CHUNK_SIZE` | `512` | Token size per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per vault |
| `ROUTER_CONFIDENCE_HIGH` | `0.85` | High-confidence threshold |
| `ROUTER_CONFIDENCE_MED` | `0.60` | Medium-confidence threshold |
| `MULTI_VAULT_THRESHOLD` | `0.35` | Min confidence for multi-vault search |
| `RULE_EXTRACTION_THRESHOLD` | `3` | Corrections needed to create a rule |
| `DATA_DIR` | `./data` | Database and metadata root |

---

## 📡 API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `POST` | `/query` | Ask a question (auto-routed) |
| `POST` | `/upload` | Upload + ingest a document |
| `GET` | `/vaults` | List all vaults |
| `POST` | `/vaults/create` | Create a new vault |
| `DELETE` | `/vaults/delete` | Delete a vault and all its data |
| `GET` | `/vaults/{name}/stats` | Get stats for a specific vault |
| `POST` | `/correction` | Log a routing correction |

Interactive docs: **http://localhost:8000/docs**

---

## 🧪 Running tests

```bash
pytest tests/ -v
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `Ollama is not running` | Start Ollama: `ollama serve` or launch the Ollama app |
| `sqlite-vec` install fails | Try: `pip install sqlite-vec --pre` |
| `ModuleNotFoundError` | Ensure you're in the project root and have activated the virtual environment |
| PDF/image not parsing | Install Tesseract and add it to PATH |
| Audio not transcribing | Install ffmpeg and add it to PATH |
| Slow first load | Phi and nomic-embed-text load on first use; subsequent requests are faster |

---

## 🔒 Privacy

Everything runs locally. No model calls leave your machine. Your documents, queries, and answers stay on your hardware.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*eshu · Built locally · Owned completely.*
