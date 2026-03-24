<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-0.2-1C3C3C?logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/Qdrant-1.12-DC244C?logo=data:image/svg+xml;base64,&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/LangSmith-Tracing-FF6600?logo=langchain&logoColor=white" />
</p>

# AURA — Agentic RAGOps 🎓

**Enterprise-Grade Academic Chatbot | Multi-Agent Finite State Machine | LLMOps**

> Hệ thống Trợ lý Học thuật Thông minh sử dụng kiến trúc **Đa tác nhân (Multi-Agent FSM)** với LangGraph, kết hợp **Hybrid Search**, **HyDE**, **Self-Reflection (Critic Agent)**, và pipeline **RAGOps** đầy đủ — từ Ingestion → Retrieval → Generation → Evaluation → CI/CD → Docker.

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Tech Stack](#-tech-stack)
- [Tính năng chi tiết](#-tính-năng-chi-tiết)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cài đặt &amp; Chạy](#-cài-đặt--chạy)
- [API Endpoints](#-api-endpoints)
- [Roadmap](#-roadmap-6-tuần)
- [Tài liệu](#-tài-liệu)

---

## 🎯 Tổng quan

AURA không phải chatbot hỏi-đáp đơn giản. Đây là một **hệ thống AI có khả năng tự suy luận, tự sửa sai**, được giám sát chặt chẽ bằng LLMOps, và có thể vận hành thực tế.

### Điểm khác biệt so với RAG truyền thống

| RAG Truyền thống                      | AURA (Agentic RAGOps)                                                                                             |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Luồng thẳng: Retrieve → Generate     | **Đa nhánh, tuần hoàn**: Router → Researcher → Relevancy → Generator → Critic → (retry hoặc pass) |
| Không kiểm tra chất lượng đầu ra | **Critic Agent** phát hiện hallucination, ép retry nếu bịa                                             |
| Vector Search đơn thuần              | **Hybrid Search**: Dense (Qdrant) + Sparse (BM25) + HyDE + Compression + Reorder                            |
| Không trích dẫn nguồn               | **Citation Engine**: Mỗi câu trả lời kèm footnote `[1], [2]` trỏ về section + page                 |
| Không giám sát                       | **LangSmith** tracing mọi token, latency, luồng suy luận                                                 |
| Sập 1 model = sập hệ thống          | **LiteLLM Gateway**: Auto-failover Gemini → Gemini Lite → Groq Llama                                      |
| Không đo lường chất lượng        | **RAGAS CI/CD**: Auto-evaluate 20 câu hỏi mỗi lần push, block deploy nếu điểm giảm                  |

---

## 🏗 Kiến trúc hệ thống

### Luồng Truy vấn (Agentic Query Pipeline)

```
User Question
      │
      ▼
┌─────────────┐     "chat"     ┌───────────┐
│   Router    │───────────────▶│   Chat    │──▶ END
│   Agent     │                │   Node    │
└──────┬──────┘                └───────────┘
       │ "search"
       ▼
┌──────────────┐     ┌────────────────┐     "not_relevant"    ┌──────────┐
│  Researcher  │────▶│   Relevancy    │─────────────────────▶│ Fallback │
│  (Hybrid     │     │   Grader       │                       │ Node     │──▶ Formatter ──▶ END
│   Search)    │     └───────┬────────┘                       └──────────┘
└──────────────┘             │ "relevant"                           ▲
       ▲                     ▼                                     │
       │            ┌──────────────┐     ┌────────────────┐        │
       │            │  Generator   │────▶│   Critic       │        │
       │            │  (Writer)    │     │   (Judge)      │        │
       │            └──────────────┘     └───────┬────────┘        │
       │                                         │                 │
       │              "not_useful"               │ "pass"          │ "max_retries"
       └─────────────(retry, max 3)──────────────┤                 │
                                                  │                 │
                                                  ▼                 │
                                          ┌──────────────┐          │
                                          │  Formatter   │──▶ END   │
                                          │  (Citation)  │          │
                                          └──────────────┘          │
```

### Luồng Nạp liệu (Ingestion Pipeline)

```
PDF/DOCX/Scan ──▶ Parsing ──▶ OCR (nếu scan) ──▶ Academic Chunking ──▶ Embedding
                                                   (giữ heading,        (Gemini)
                                                    không cắt công  
                                                    thức toán)      
                                                        │
                                                        ├──▶ Qdrant (Dense Vector)
                                                        └──▶ BM25 Index (Sparse Keyword)
```

### Kiến trúc Docker

```
┌──────────────────────────────── docker compose ────────────────────────────────┐
│                                                                                │
│  ┌─────────┐     ┌─────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │  Qdrant │     │  Redis  │     │   FastAPI    │     │  Streamlit   │        │
│  │  :6333  │◄────│  :6379  │◄────│   API :8000  │◄────│   UI :8501   │        │
│  │ VectorDB│     │  Cache  │     │   Backend    │     │  Frontend    │        │
│  └─────────┘     └─────────┘     └──────────────┘     └──────────────┘        │
│                                                                                │
│  ┌──────────────┐  (Optional, --profile full)                                  │
│  │   LiteLLM   │                                                               │
│  │   :4000     │  LLM Gateway / Failover                                       │
│  └──────────────┘                                                              │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Tech Stack

### Lớp Ứng dụng & Trí tuệ (Application & Orchestration)

| Công nghệ                | Vai trò                                                     |
| -------------------------- | ------------------------------------------------------------ |
| **LangGraph**        | Kiến trúc Multi-Agent FSM (Router → Researcher → Critic) |
| **LangChain**        | LCEL chains, Retriever abstractions, Prompt Management       |
| **LlamaIndex**       | File readers, Document loading                               |
| **Gemini 2.5 Flash** | LLM chủ lực (Generation, Routing, Judging)                 |

### Lớp Dữ liệu & Tìm kiếm (Data & Retrieval)

| Công nghệ                | Vai trò                                |
| -------------------------- | --------------------------------------- |
| **Qdrant**           | Vector Database (Dense Retrieval, HNSW) |
| **BM25 (rank-bm25)** | Sparse Keyword Retrieval                |
| **Redis**            | Semantic Cache + Chat Message History   |
| **RapidOCR**         | OCR cho tài liệu scan/hình ảnh      |

### Lớp Giám sát & Đánh giá (RAGOps & LLMOps)

| Công nghệ              | Vai trò                                                 |
| ------------------------ | -------------------------------------------------------- |
| **LangSmith**      | Tracing toàn bộ luồng Agent (Token, Latency, Steps)   |
| **RAGAS**          | Đánh giá tự động (Faithfulness, Context Precision) |
| **LiteLLM**        | LLM Gateway — Auto-failover (Gemini → Groq Llama)      |
| **GitHub Actions** | CI/CD — Tự động evaluate 20 câu hỏi mỗi lần push |

### Lớp Giao diện & Hạ tầng (Serving & Infrastructure)

| Công nghệ                 | Vai trò                                                   |
| --------------------------- | ---------------------------------------------------------- |
| **FastAPI + Uvicorn** | Backend API (Upload, Search, Ask, Summarize)               |
| **Streamlit**         | Chat UI với nút 👍/👎 Feedback (Human-in-the-Loop)       |
| **Gradio**            | Giao diện chat thay thế                                  |
| **Docker Compose**    | Đóng gói 4+1 services (API, UI, Qdrant, Redis, LiteLLM) |

---

## ✨ Tính năng chi tiết

### 1. Multi-Agent FSM (LangGraph)

- **Router Agent** — Phân loại câu hỏi: Tán gẫu → Chat Node; Cần kiến thức → Researcher
- **Researcher Agent** — Tổng hợp Hybrid Search + HyDE + Compression + Reorder
- **Relevancy Grader** — Kiểm tra tài liệu có liên quan câu hỏi không (trước Generator)
- **Generator Agent** — Viết câu trả lời dựa trên context (temperature=0.3)
- **Critic Agent** — "LLM-as-a-Judge" kiểm tra hallucination (temperature=0). Fail → retry (max 3 vòng)
- **Formatter Agent** — Chuẩn format Markdown + chèn trích dẫn [1], [2]
- **Fallback Agent** — Graceful degradation (3 loại lỗi: no_session, not_relevant, max_retries)

### 2. Hybrid Retrieval Pipeline

- **Dense Search**: Qdrant cosine similarity (text-embedding-004)
- **Sparse Search**: BM25 keyword matching
- **Ensemble**: Weighted fusion (60% Dense, 40% BM25) — configurable
- **HyDE**: Hypothetical Document Embeddings cho câu hỏi mập mờ
- **Context Compression**: Regex lọc rác (header/footer), giữ công thức + định nghĩa
- **Reorder**: Short-to-long grouped by heading — chống "Lost in the Middle"

### 3. Academic Ingestion

- Đa format: PDF text, PDF scan (OCR), Word (.docx)
- **AcademicTextSplitter**: Nhận diện Heading (I, II, Abstract, Method), không cắt đứt công thức
- **Metadata enrichment**: `{doc_id, section, page, heading, contains_math}`
- **RapidOCR**: Bóc tách chữ từ tài liệu scan/hình ảnh

### 4. LLMOps & Observability

- **LangSmith Tracing**: Theo dõi token, latency, luồng suy luận của mọi Agent
- **PromptOps**: Prompt quản lý tập trung tại `lc/prompts/templates.py`, hỗ trợ LangSmith Hub
- **Feedback Loop**: Nút 👍/👎 → Thumbs Down tự động log state snapshot vào JSON
- **LiteLLM Gateway**: Proxy chống vendor lock-in, auto-failover 3 tầng model

### 5. CI/CD Evaluation

- **GitHub Actions**: Trigger mỗi lần push main/develop
- **Golden Dataset**: 20 câu hỏi + ground truth answers
- **Automated Scoring**: Context Hit Rate, Answer Similarity, Answer Completeness
- **Baseline Comparison**: Block deploy nếu điểm giảm dưới threshold
- **Report Artifact**: Upload evaluation report cho mỗi CI run

---

## 📂 Cấu trúc dự án

```
AURA/
├── app/                          # 🟢 FastAPI Application
│   ├── main.py                   #    Entry point — mount routers + middleware
│   ├── settings.py               #    Load config từ YAML + env vars
│   ├── deps.py                   #    Dependencies (JSON size limiter)
│   ├── middleware_rate_limit.py   #    Rate limiting middleware
│   └── api/
│       ├── routes_healthz.py     #    GET /healthz
│       ├── routes_session.py     #    POST /session/upload, /session/search
│       ├── routes_ask.py         #    POST /ask, /summarize_*
│       └── routes_toy.py         #    POST /toy (demo endpoint)
│
├── core/                         # 🔵 Shared Utilities
│   ├── chunking/                 #    Academic text splitting (heading-aware)
│   ├── citation/                 #    Citation footnote engine [1], [2]
│   ├── embedding/                #    Gemini text-embedding-004
│   ├── guardrails/               #    Min docs/tokens check before LLM call
│   ├── parsing/                  #    PDF/DOCX parsing (PyMuPDF, pdfminer)
│   ├── ocr.py                    #    RapidOCR wrapper
│   ├── retry.py                  #    Retry decorator with exponential backoff
│   └── telemetry/                #    Custom telemetry utilities
│
├── lc/                           # 🤖 LangChain & LangGraph Logic
│   ├── agents/
│   │   └── graph.py              #    ★ Multi-Agent FSM (7 nodes, 504 lines)
│   ├── chains/
│   │   ├── context_build.py      #    advanced_retrieve() — Hybrid+HyDE+Compress+Reorder
│   │   ├── qa_chain.py           #    LCEL QA chain with Redis cache + memory
│   │   ├── compress.py           #    Regex context compressor
│   │   ├── reorder.py            #    Document reorder strategy
│   │   └── summarize_chain.py    #    TL;DR / Executive / QFS summarization
│   ├── prompts/
│   │   └── templates.py          #    PromptOps — centralized prompt management
│   ├── retrievers/
│   │   ├── bm25.py               #    BM25 sparse retriever
│   │   ├── ensemble.py           #    Weighted ensemble (Dense + BM25)
│   │   ├── hyde.py               #    HyDE query expansion
│   │   └── compressor.py         #    Document compression retriever
│   └── vectordb/
│       └── qdrant_store.py       #    Qdrant client singleton + CRUD
│
├── ops/                          # 📊 Operations & LLMOps
│   ├── ingest/                   #    Ingestion orchestrator
│   ├── loaders/
│   │   ├── academic_loader.py    #    Standard document loader
│   │   └── legacy_ocr_loader.py  #    Custom OCR loader (kế thừa BaseLoader)
│   ├── splitters/
│   │   └── academic_splitter.py  #    AcademicTextSplitter
│   ├── observability/
│   │   └── langsmith_setup.py    #    LangSmith initialization
│   ├── feedback/
│   │   └── collector.py          #    👍/👎 feedback → JSON files
│   └── evaluation/               #    RAGAS evaluation utilities
│
├── ui/                           # 🟣 Frontend
│   ├── streamlit_app.py          #    Streamlit chat UI (feedback, upload, stats)
│   └── gradio_app.py             #    Gradio chat interface
│
├── eval/                         # 📊 Evaluation
│   ├── golden_dataset.json       #    20 Q&A pairs for automated testing
│   ├── baseline_scores.json      #    Threshold scores for CI/CD gates
│   └── run_baseline.py           #    Baseline evaluation script (Day 5)
│
├── tests/                        # 🧪 Test Suite
│   ├── test_day3_splitter.py     #    Text Splitter tests
│   ├── test_day4_api.py          #    FastAPI route tests
│   ├── test_day6_hybrid.py       #    Hybrid Search tests
│   ├── test_day7_optimize.py     #    HyDE + Compression tests
│   ├── test_day8_lcel.py         #    LCEL chain tests
│   ├── test_day9_10_agents.py    #    Router + Researcher tests
│   ├── test_day11_12_critic.py   #    Generator + Critic tests
│   ├── test_day13_14_formatter.py#    Formatter + E2E tests
│   ├── test_day17_18_promptops_feedback.py  # PromptOps + Feedback tests
│   └── test_day19_20_ragas_ci.py #    CI/CD RAGAS evaluation tests
│
├── configs/                      # ⚙️ Configuration
│   ├── app.yaml                  #    App settings (chunk size, retriever weights, etc.)
│   └── ops.yaml                  #    Operations config
│
├── docker/                       # 🐳 Docker
│   ├── Dockerfile.api            #    Multi-stage build for FastAPI
│   └── Dockerfile.ui             #    Build for Streamlit
│
├── .github/workflows/
│   └── evaluate.yml              #    CI/CD — RAGAS evaluation pipeline
│
├── docker-compose.yaml           #    4+1 services orchestration
├── litellm_config.yaml           #    LiteLLM model gateway config
├── requirements.txt              #    Python dependencies
├── .env.example                  #    Environment variables template
├── .dockerignore                 #    Docker build context exclusions
└── .gitignore                    #    Git ignore rules
```

---

## 🚀 Cài đặt & Chạy

### Cách 1: Docker Compose (Khuyến nghị — Production)

> **Yêu cầu**: Docker Desktop đã cài đặt.

```bash
# 1. Clone repo
git clone https://github.com/AnhVan17/AURA-Agentic-RAGOps.git
cd AURA-Agentic-RAGOps

# 2. Tạo file .env từ template
cp .env.example .env
# Mở .env → điền GOOGLE_API_KEY=AIza...

# 3. Khởi động 4 services (Qdrant, Redis, API, UI)
docker compose up -d

# 4. Kiểm tra trạng thái
docker compose ps
```

| Service                    | URL                             | Mô tả                      |
| -------------------------- | ------------------------------- | ---------------------------- |
| **Streamlit UI**     | http://localhost:8501           | 💬 Giao diện chat chính    |
| **FastAPI Docs**     | http://localhost:8000/docs      | 📖 Swagger API documentation |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | 🔷 Quản lý Vector Database |
| **API Health**       | http://localhost:8000/healthz   | ❤️ Health check            |

```bash
# Xem logs
docker compose logs -f api       # Logs của API
docker compose logs -f            # Tất cả logs

# Dừng hệ thống
docker compose down               # Dừng (giữ data)
docker compose down -v            # Dừng + XÓA data

# Chạy với LiteLLM Gateway (optional)
docker compose --profile full up -d
```

---

### Cách 2: Chạy Local (Development)

> **Yêu cầu**: Python 3.11+, (Optional) Docker cho Qdrant & Redis.

```bash
# 1. Clone & setup
git clone https://github.com/AnhVan17/AURA-Agentic-RAGOps.git
cd AURA-Agentic-RAGOps

# 2. Tạo môi trường ảo
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Cài đặt thư viện
pip install -r requirements.txt

# 4. Cấu hình env
cp .env.example .env
# Sửa .env → điền GOOGLE_API_KEY

# 5. (Optional) Khởi động Qdrant & Redis bằng Docker
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:v1.12.4
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

#### Khởi động Backend API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

→ Mở http://localhost:8000/docs

#### Khởi động UI (Streamlit)

```bash
streamlit run ui/streamlit_app.py
```

→ Mở http://localhost:8501

#### Chạy Tests

```bash
# Toàn bộ test suite
python -m pytest tests/ -v

# Theo ngày cụ thể
python -m pytest tests/test_day9_10_agents.py -v -s      # Agents
python -m pytest tests/test_day11_12_critic.py -v -s      # Critic
python -m pytest tests/test_day19_20_ragas_ci.py -v -s    # CI/CD Eval
```

#### Chạy Baseline Evaluation

```bash
python eval/run_baseline.py
```

---

## 📡 API Endpoints

| Method   | Endpoint            | Mô tả                                                                      |
| -------- | ------------------- | ---------------------------------------------------------------------------- |
| `GET`  | `/healthz`        | Health check                                                                 |
| `POST` | `/session/upload` | Upload file PDF/DOCX → Ingest vào Qdrant + BM25                            |
| `POST` | `/session/search` | Hybrid Search (Dense + BM25 + HyDE)                                          |
| `POST` | `/ask`            | Hỏi đáp — chạy full Agent pipeline (Router→Researcher→Critic→Format) |
| `POST` | `/summarize_tldr` | Tóm tắt TL;DR                                                              |
| `POST` | `/summarize_exec` | Tóm tắt Executive Summary                                                  |
| `POST` | `/summarize_qfs`  | Query-Focused Summarization                                                  |
| `POST` | `/toy`            | Demo endpoint (test LLM connection)                                          |

**Ví dụ gọi API:**

```bash
# Upload paper
curl -X POST "http://localhost:8000/session/upload?session_id=S1" \
  -F "file=@paper.pdf"

# Hỏi đáp
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "S1", "question": "Transformer model hoạt động như thế nào?", "k": 8}'
```

---

## 🗺 Roadmap 6 Tuần

### Giai đoạn 1: Nền Móng Dữ Liệu & RAGOps (Tuần 1)

| Ngày | Chủ đề                                                  | Trạng thái    |
| ----- | ---------------------------------------------------------- | --------------- |
| 1     | Observability — LangSmith Tracing                         | ✅ Hoàn thành |
| 2     | Hybrid Ingestion — PDF + OCR + LegacyLoader               | ✅ Hoàn thành |
| 3     | Academic Chunking — Heading-aware, bảo toàn công thức | ✅ Hoàn thành |
| 4     | Vector Database & FastAPI — Qdrant, Upload, Search        | ✅ Hoàn thành |
| 5     | Baseline Evaluation — Golden Dataset + RAGAS              | ✅ Hoàn thành |

### Giai đoạn 2: Tối Ưu Truy Xuất (Tuần 2)

| Ngày | Chủ đề                                            | Trạng thái    |
| ----- | ---------------------------------------------------- | --------------- |
| 6     | Hybrid Search — BM25 + Qdrant Ensemble              | ✅ Hoàn thành |
| 7     | HyDE + Compression — Query expansion + Regex filter | ✅ Hoàn thành |
| 8     | LCEL Chain + Memory — Redis Cache + Chat History    | ✅ Hoàn thành |

### Giai đoạn 3: Multi-Agent LangGraph (Tuần 3-4)

| Ngày | Chủ đề                                          | Trạng thái    |
| ----- | -------------------------------------------------- | --------------- |
| 9-10  | Router Agent + Researcher Agent                    | ✅ Hoàn thành |
| 11-12 | Generator + Critic Agent (Anti-Hallucination Loop) | ✅ Hoàn thành |
| 13-14 | Formatter + Fallback Agent (Graceful Degradation)  | ✅ Hoàn thành |

### Giai đoạn 4: LLMOps & Production (Tuần 5-6)

| Ngày | Chủ đề                                          | Trạng thái    |
| ----- | -------------------------------------------------- | --------------- |
| 15-16 | LiteLLM Gateway — Model Failover (Gemini → Groq) | ✅ Hoàn thành |
| 17-18 | PromptOps + Feedback UI — Human-in-the-Loop       | ✅ Hoàn thành |
| 19-20 | CI/CD — GitHub Actions + RAGAS Auto-Evaluation    | ✅ Hoàn thành |
| 21+   | Dockerization — Multi-stage build + Compose       | ✅ Hoàn thành |

---

## 📖 Tài liệu

| File                                                              | Mô tả                               |
| ----------------------------------------------------------------- | ------------------------------------- |
| [`docs/AGENT_ARCHITECTURE.md`](docs/AGENT_ARCHITECTURE.md)         | Kiến trúc Multi-Agent FSM chi tiết |
| [`docs/TESTING_GUIDE.md`](docs/TESTING_GUIDE.md)                   | Hướng dẫn chạy test suite         |
| [`docs/OBSERVABILITY_GUIDE.md`](docs/OBSERVABILITY_GUIDE.md)       | LangSmith setup & tracing             |
| [`docs/LLMOPS_MIGRATION_GUIDE.md`](docs/LLMOPS_MIGRATION_GUIDE.md) | LiteLLM integration guide             |
| [`docs/UI_GUIDE.md`](docs/UI_GUIDE.md)                             | Streamlit UI & Feedback system        |
| [`docs/DOCKER_DEPLOYMENT.md`](docs/DOCKER_DEPLOYMENT.md)           | Docker deployment guide               |

---

## ⚙️ Cấu hình

### Biến môi trường (`.env`)

```env
# Bắt buộc
GOOGLE_API_KEY=your_gemini_api_key

# Optional — Observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=AURA-Academic-Bot

# Optional — Fallback models
GOOGLE_API_KEY_2=your_backup_key
GROQ_API_KEY=your_groq_key

# Docker tự override (không cần sửa)
# QDRANT_URL=http://qdrant:6333
# REDIS_URL=redis://redis:6379
```

### App Config (`configs/app.yaml`)

```yaml
retrieval:
  top_k: 8                    # Số documents trả về
  ensemble:
    dense: 0.6                # Trọng số Vector Search
    bm25: 0.4                 # Trọng số Keyword Search
  hyde:
    enable: true              # Bật HyDE cho câu hỏi mập mờ

compression:
  enable: true                # Bật Regex compression
  min_reduction_ratio: 0.30   # Giảm tối thiểu 30% token

reorder:
  enable: true
  strategy: "short_to_long_group_by_heading"
```

---
