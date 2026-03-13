# AURA - Agentic RAGOps (Enterprise-Grade Academic Chatbot) 🎓

**AURA** là một hệ thống Trợ lý ảo Học thuật Tiên tiến (Agentic RAG) được thiết kế đặc biệt để xử lý, truy xuất và tổng hợp thông tin từ các tài liệu học thuật, báo cáo khoa học, và tài liệu phức tạp (có chứa toán học, bảng biểu).

Hệ thống vượt qua chuẩn RAG truyền thống bằng cách áp dụng kiến trúc **Đa tác nhân (Multi-Agent FSM)** với LangGraph, kết hợp cùng các công nghệ tìm kiếm siêu việt (Hybrid Search: Qdrant + BM25, HyDE, Regex Compression).

---

## 🚀 Tính năng nổi bật

1. **Agentic RAG (LangGraph)**
   - **Router Agent**: Tự động phân loại luồng hội thoại (Tán gẫu vs. Tra cứu kiến thức học thuật).
   - **Researcher Agent**: Tổng hợp toàn bộ công nghệ tìm kiếm nâng cao để mang về tài liệu chuẩn xác nhất.
2. **Hybrid Retrieval Pipeline (Tìm kiếm kép)**
   - Kết hợp **Qdrant (Dense Vector - Mức độ ngữ nghĩa)** và **BM25 (Sparse - Khớp từ khóa)** với tỷ trọng tối ưu.
   - **HyDE (Hypothetical Document Embeddings)**: Chống lại các câu hỏi mập mờ bằng cách yêu cầu LLM tự sinh câu trả lời nháp để đi quét vector.
   - **Context Compression**: Lọc bỏ các dòng nhiễu thông qua Regex (giữ lại công thức toán, định nghĩa, từ khóa chính) nhằm nén lượng Token đưa vào LLM.
   - **Reorder**: Thay đổi thứ tự tài liệu để đối phó với hiện tượng "Lost in the middle" của LLM.
3. **Data Ingestion & OCR Thông minh**
   - Hỗ trợ đa dạng format: PDF text, PDF scan, Word.
   - Tích hợp **RapidOCR** để bóc tách chữ từ tài liệu scan/hình ảnh.
   - **Academic Chunking**: Bộ cắt dữ liệu (Splitter) nhận diện Heading (Chương/Mục) và không cắt đứt gãy công thức toán học.
4. **Giám sát & Đánh giá (LLMOps)**
   - Theo dõi từng bước chạy của Agent (Token, Latency) qua **LangSmith**.
   - Đánh giá tự động chất lượng câu trả lời bằng thư viện **Ragas** (Context Precision, Faithfulness).

---

## 🏗 Kiến trúc hệ thống & Cách thức hoạt động

Quá trình hoạt động của AURA chia thành 2 luồng chính:

### 1. Luồng Nập liệu (Ingestion Pipeline)

`File PDF` $\rightarrow$ `Parsing/OCR` $\rightarrow$ `Academic Chunking (cắt theo heading)` $\rightarrow$ `Embedding (Gemini)` $\rightarrow$ Lư đồng thời vào **Qdrant** (Vector) & **BM25** (Từ khóa).

### 2. Luồng Truy vấn (Agentic Query Pipeline)

1. **User** gửi câu hỏi.
2. **Router Agent** kiểm tra:
   - Nếu là tán gẫu $\rightarrow$ Rẽ sang **Chat Node** $\rightarrow$ Trả lời kết thúc.
   - Nếu là hỏi kiến thức $\rightarrow$ Rẽ sang **Researcher Node**.
3. **Researcher Node** thực thi chuỗi lệnh:
   - *Luồng chính*: Truy vấn Qdrant + Truy vấn BM25 $\rightarrow$ Trộn điểm (Ensemble).
   - *Luồng phụ (Trigger)*: Nếu kết quả kém, tự động kích hoạt **HyDE**.
   - *Hậu xử lý*: Nén (Compression) $\rightarrow$ Đảo vị trí (Reorder) $\rightarrow$ Xuất ra danh sách tài liệu cuối cùng.
4. (Sắp ra mắt) **Critic / Generator Node**: Viết câu trả lời và tự động kiểm duyệt sự thật (Faithfulness) trước khi gửi cho User.

---

## 📂 Cấu trúc dự án

```text
academic chatbot/
├── app/                  # FastAPI Application
│   ├── api/              # Định nghĩa các endpoint (upload, search, chat)
│   ├── settings.py       # Load config từ YAML & Biến môi trường
│   └── main.py           # File khởi chạy server FastApi
├── configs/              # Các file cấu hình hệ thống
│   ├── app.yaml          # Tinh chỉnh thông số (chunk size, retriever weights, v.v.)
│   └── ops.yaml
├── core/                 # Thư viện dùng chung (Utilities)
│   ├── chunking/         # Thuật toán cắt văn bản học thuật
│   ├── embedding/        # Gọi API tạo Vector (Gemini)
│   ├── parsing/          # Bóc tách PDF
│   └── ocr.py            # OCR xử lý tài liệu ảnh
├── lc/                   # LangChain & LangGraph Logic
│   ├── agents/           # Node & Graph (Router, Researcher)
│   ├── chains/           # Pipeline tìm kiếm (Ensemble, HyDE, Compression, Reorder)
│   ├── retrievers/       # Khởi tạo thuật toán BM25 và Vector Search
│   └── vectordb/         # Giao tiếp với cơ sở dữ liệu Qdrant
├── eval/                 # Đánh giá tự động hệ thống (Baseline, Ragas)
├── tests/                # Bộ Unit Test (pytest)
├── artifacts/            # Lưu trữ file log, database BM25 cục bộ
├── qdrant_local_db/      # Vector database cục bộ
└── .env                  # Cấu hình Secret Keys (Google API, LangSmith)
```

---

## ⚙️ Cài đặt & Cấu hình

**1. Clone dự án & Cài đặt môi trường**

```bash
git clone <repository-url>
cd "academic chatbot"

# Tạo môi trường ảo
python -m venv .venv

# Activate môi trường
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/Mac:
source .venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

**2. Cấu hình các biến môi trường (.env)**
Tạo một file `.env` ở thư mục gốc chứa các thông tin sau:

```env
GOOGLE_API_KEY=your_gemini_api_key

# Cấu hình LangSmith (Để giám sát hệ thống)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=AURA-Academic-Bot
```

---

## 🏃 Hướng dẫn chạy & Sử dụng

### 1. Khởi động Backend API (FastAPI)

Lõi hệ thống cung cấp API cho việc Upload tài liệu và hỏi đáp.

```bash
uvicorn app.main:app --reload
```

- Mở URL: [http://localhost:8000/docs](http://localhost:8000/docs) để xem giao diện Swagger UI và gọi thử API.

### 2. Chạy thử các Agent (Bản Console)

Bạn có thể theo dõi luồng suy nghĩ của các Node (Router, Researcher) thông qua log console bằng file chạy trực tiếp:

```bash
python lc/agents/graph.py
```

### 3. Chạy bộ Unit Test

Xác nhận rằng toàn bộ module Retriever, Agent, Splitting hoạt động hoàn hảo:

```bash
pytest tests/ -v -s
```
