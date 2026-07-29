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
