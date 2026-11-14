# KrishiNyay AI

A multilingual, source-grounded assistant for Indian farmers, with FastAPI, MiniLM + Chroma hybrid retrieval, live weather and mandi routing, and a browser interface.

## Run locally

Use Python 3.12 and Node 22 or newer. The repository includes the 1,748-chunk demo index; no ingestion is needed to start.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. The first retrieval downloads the multilingual MiniLM model unless it is already cached. `build_model.py` bundles the pinned model for deployments. Existing `chroma_db/` and `data/chunks/` directories take precedence locally; otherwise the packaged demo is used. Explicit `CHROMA_PATH` and `CHUNKS_DIR` override those defaults.

Configure `GEMINI_API_KEY` in `.env` for Gemini generation, and `DATA_GOV_IN_API_KEY` for live Agmarknet prices. With no generation key or local Ollama model, the app returns clearly labelled source-grounded template answers. PM-KISAN personal status is checked through the official portal; this app does not retrieve private beneficiary records.

For scanned PDFs, install Tesseract and its English, Hindi, Marathi, and Punjabi language packs. The Python wrapper alone does not provide an OCR executable.

## Mandi price reliability

Hosted deployments prefer the official AGMARKNET public market reports (`MANDI_PRIMARY_PROVIDER=agmarknet_public`). The configured Data.gov.in API remains a backup. Public directory/report requests never receive your Data.gov.in key. Metadata is cached for one hour and dated price reports for five minutes per instance.

A price query must identify a specific mandi. If the market is missing or ambiguous, the chat asks for it and preserves the crop across the follow-up. Reports show the actual market, variety, observation date and unit. An older report is never described as today's quote, and a failed lookup is not marked as verified price evidence. Example: ask “Aaj soybean ka mandi bhav kya hai?”, then reply “Indore APMC, Madhya Pradesh”.

## Docker and monitoring

```bash
docker compose up --build -d
```

This starts the app on http://localhost:8000, Prometheus on http://localhost:9090, and Grafana on http://localhost:3000. Grafana provisions the Prometheus datasource and the KrishiNyay AI dashboard automatically. Monitoring ports bind to localhost. Grafana and Prometheus data persist in named volumes. The Docker image includes Tesseract and Indic language packs. Optional secrets come from `.env`; never commit that file.

## Vercel

Import this repository with the FastAPI preset, or run:

```bash
vercel link
vercel env add GEMINI_API_KEY production
# Optional, for live mandi prices:
vercel env add DATA_GOV_IN_API_KEY production
vercel --prod
```

`vercel.json` enables public mode, disables ingestion, and uses the packaged Chroma index with a writable temporary copy. `pyproject.toml` selects Python 3.12 and Linux CPU-only PyTorch. The build bundles a pinned 384-dimensional MiniLM model so requests do not need to download it. The model and ML libraries may require Vercel Large Functions; enable `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` if the project requests it. See [Vercel FastAPI deployment](https://vercel.com/docs/frameworks/backend/fastapi) and [function limits](https://vercel.com/docs/functions/limitations).

Vercel runs the web app and API. Use the Docker deployment for the full Prometheus/Grafana stack and scanned-PDF ingestion with system Tesseract. In-memory query limits and metrics are per function instance on Vercel and reset on cold starts; use shared infrastructure for a global production quota or continuous metrics collection. Public ingestion stays disabled, so the packaged index is not modified by visitors.

## Verification

```bash
python build_model.py
LLM_PROVIDER=template python run_regression_suite.py --skip-generation
python -m unittest discover -s tests
node --check web/app.js
docker compose config --quiet
```

GitHub Actions runs the regression suite on Python 3.12 with Tesseract installed. Offline validators mock external services where appropriate; passing them does not verify live provider credentials. Run `python validate_public_demo.py --require-gemini` after configuring Gemini to check the hosted generation setup. Run the optional Ollama generation gate only when the local model is available.

`GET /health` reports index, language model, API, and OCR readiness. `GET /metrics` exposes Prometheus metrics. Check a real `/query` response after every deployment. See [PUBLIC_DEMO.md](PUBLIC_DEMO.md) for Hugging Face Docker hosting and the farmer demo script.

## 🏗️ Architecture

KrishiNyay AI follows a modular Retrieval-Augmented Generation (RAG) architecture designed to provide accurate, source-grounded answers for Indian farmers. The system combines semantic retrieval, workflow-based routing, live government APIs, and LLM-based generation while ensuring every response is backed by trusted evidence.



### Client Layer

The application provides a lightweight web interface built using HTML, CSS, and JavaScript. Farmers can ask questions in English, Hindi, or Hinglish. Every request is sent to the FastAPI backend, which returns the generated answer along with supporting sources and workflow metadata.

---

### Backend & Workflow Engine

FastAPI acts as the central controller of the system.

Each incoming request passes through a guarded workflow that performs:

- Language Detection
- Query Normalization
- Intent Classification
- Slot Extraction
- Missing Information Detection
- Dynamic Route Selection
- Evidence Verification

Based on the detected intent, the request is routed to one of the following pipelines:

- Static RAG Retrieval
- Live API Services
- System Information
- Clarification Workflow

This routing layer prevents incorrect retrievals and unnecessary LLM calls.

---

### Knowledge Base

The knowledge base is built entirely from trusted agricultural resources, including government schemes, official PDFs, legal documents, crop insurance policies, and farmer welfare programs.

The ingestion pipeline performs:

- PDF Extraction
- OCR for scanned documents
- Text Cleaning
- Hindi-aware Normalization
- Chunk Generation
- Metadata Creation
- Vector Embedding

Each processed chunk is stored together with its source information for citation.

---

### Retrieval Engine

KrishiNyay uses multilingual MiniLM embeddings to perform semantic search across the indexed corpus.

The retrieval pipeline includes:

- MiniLM Embeddings
- ChromaDB Vector Storage
- Hybrid Semantic + Lexical Retrieval
- Evidence Verification
- Source Ranking

Only the highest-quality evidence is forwarded to the language model.

---

### Live Data Routing

Questions containing rapidly changing information bypass the RAG pipeline.

Instead, they are routed to trusted live services.

Supported routes include:

- Weather Forecasts (Open-Meteo)
- Mandi Prices (Data.gov.in / Agmarknet)
- PM-KISAN Beneficiary Status (Official Portal)

This ensures responses remain current instead of relying on outdated indexed documents.

---

### Answer Generation

After retrieval, the verified evidence is passed to the configured language model.

Supported providers include:

- Gemini
- Ollama (Local)
- Template-based Fallback

Every generated response includes:

- Final Answer
- Supporting Sources
- Retrieval Metadata
- Workflow Information
- Route Details

This keeps responses transparent and source-grounded.

---

### Monitoring & Observability

The application integrates Prometheus and Grafana to monitor backend performance in real time.

Collected metrics include:

- API Requests
- Farmer Queries
- Query Latency
- Gemini API Usage
- RAG vs Live API Requests
- Backend Health Metrics

These metrics help identify bottlenecks, monitor system usage, and evaluate overall application performance.

---

### Deployment

The application is containerized using Docker for consistent deployment across environments.

Deployment includes:

- FastAPI Backend
- Prometheus Monitoring
- Grafana Dashboards

The project can be deployed locally or on cloud platforms such as Hugging Face Spaces or any Docker-compatible server.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|--------|--------------|
| Frontend | HTML, CSS, JavaScript |
| Backend | FastAPI, Python |
| Workflow | Intent Classification, Slot Filling, Query Routing |
| Embeddings | MiniLM Multilingual |
| Vector Database | ChromaDB |
| Retrieval | Hybrid Semantic + Lexical Search |
| OCR | Tesseract OCR |
| Live APIs | Open-Meteo, Data.gov.in Agmarknet |
| LLM | Gemini, Ollama |
| Monitoring | Prometheus, Grafana |
| Deployment | Docker |

<img width="781" height="519" alt="Screenshot 2026-07-30 at 2 10 34 AM" src="https://github.com/user-attachments/assets/75ff6aad-3cf1-44bb-b955-1201727ba9d7" />
