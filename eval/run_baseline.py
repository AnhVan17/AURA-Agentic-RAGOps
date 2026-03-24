"""
Ngày 5: Baseline Evaluation với RAGAS
======================================
Script này sẽ:
1. Nạp golden_dataset.json (20 câu hỏi + đáp án chuẩn)
2. Với mỗi câu hỏi, gọi API /session/search để lấy context từ Qdrant
3. Gọi API /ask để lấy câu trả lời từ LLM
4. Dùng RAGAS chấm điểm 4 chỉ số: Faithfulness, Answer Relevance, Context Precision, Context Recall
5. Xuất báo cáo ra file eval/baseline_report.txt
"""
import json
import os
import sys
import time

# === Fix Windows console encoding ===
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ====================================================
# CẤU HÌNH
# ====================================================
BASE_URL = "http://localhost:8000"
SESSION_ID = "eval_baseline"  # Session riêng cho evaluation
GOLDEN_DATASET = "eval/golden_dataset.json"
OUTPUT_FILE = "eval/baseline_report.txt"
TOP_K = 5  # Số chunks tìm kiếm

# Papers cần nạp để đánh giá (3 papers chính trong golden dataset)
PAPERS = [
    "data/RAGAS Automated Evaluation of Retrieval Augmented Generation.pdf",
    "data/Attention Is All You Need.pdf",
    "data/ReAct Synergizing Reasoning and Acting in Language Models.pdf",
]


def log(msg: str, f=None):
    """In ra console và ghi vào file."""
    print(msg)
    if f:
        f.write(msg + "\n")


def step1_ingest_papers():
    """Nạp 3 papers vào Qdrant qua API /session/upload."""
    import requests
    print("=" * 60)
    print("BƯỚC 1: NẠP PAPERS VÀO QDRANT")
    print("=" * 60)

    for pdf_path in PAPERS:
        if not os.path.exists(pdf_path):
            print(f"  ⚠️ Không tìm thấy: {pdf_path}, bỏ qua.")
            continue
        fname = os.path.basename(pdf_path)
        print(f"  📄 Đang nạp: {fname}...")
        with open(pdf_path, "rb") as f:
            files = {"file": (fname, f, "application/pdf")}
            resp = requests.post(
                f"{BASE_URL}/session/upload?session_id={SESSION_ID}",
                files=files,
                timeout=120,
            )
        if resp.status_code == 200:
            data = resp.json()
            n_chunks = data.get("counts", {}).get("chunks", "?")
            ms = data.get("stage_ms", {}).get("total_ms", "?")
            print(f"     ✅ OK — {n_chunks} chunks, {ms} ms")
        else:
            print(f"     ❌ Lỗi {resp.status_code}: {resp.text[:200]}")

    print()


def step2_run_rag_pipeline(questions: list) -> list:
    """Chạy pipeline RAG cho mỗi câu hỏi: Search context + Ask LLM."""
    import requests
    print("=" * 60)
    print("BƯỚC 2: CHẠY RAG CHO 20 CÂU HỎI")
    print("=" * 60)

    results = []
    for i, item in enumerate(questions):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i+1:2d}/20] {q[:60]}...")

        # 2a) Search context từ Qdrant
        try:
            resp_search = requests.get(
                f"{BASE_URL}/session/search",
                params={"session_id": SESSION_ID, "q": q, "k": TOP_K},
                timeout=30,
            )
            if resp_search.status_code == 200:
                search_data = resp_search.json()
                contexts = [r["text"] for r in search_data.get("results", [])]
            else:
                contexts = []
                print(f"       ⚠️ Search lỗi: {resp_search.status_code}")
        except Exception as e:
            contexts = []
            print(f"       ⚠️ Search exception: {e}")

        # 2b) Ask LLM
        try:
            resp_ask = requests.post(
                f"{BASE_URL}/ask",
                json={"session_id": SESSION_ID, "question": q, "k": TOP_K},
                timeout=60,
            )
            if resp_ask.status_code == 200:
                answer = resp_ask.json().get("answer", "")
            else:
                answer = ""
                print(f"       ⚠️ Ask lỗi: {resp_ask.status_code}")
        except Exception as e:
            answer = ""
            print(f"       ⚠️ Ask exception: {e}")

        results.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": gt,
        })
        print(f"       ✅ Context: {len(contexts)} chunks | Answer: {len(answer)} chars")

    print()
    return results


def step3_evaluate_ragas(results: list) -> dict:
    """Chấm điểm bằng RAGAS với Gemini làm Judge LLM."""
    print("=" * 60)
    print("BƯỚC 3: CHẤM ĐIỂM RAGAS (Gemini Judge)")
    print("=" * 60)

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas import RunConfig
        from datasets import Dataset

        # === CẤU HÌNH GEMINI LÀM JUDGE ===
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from dotenv import load_dotenv
        load_dotenv()

        judge_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
        )
        judge_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
        )
        print("  Gemini Judge LLM + Embeddings đã sẵn sàng")

        # Chuẩn bị dataset theo format RAGAS yêu cầu
        eval_data = {
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "ground_truth": [r["ground_truth"] for r in results],
        }
        dataset = Dataset.from_dict(eval_data)

        print("   Đang chấm điểm (có thể mất 5-15 phút)...")
        t0 = time.perf_counter()

        # Chạy RAGAS evaluation với Gemini
        score = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=RunConfig(timeout=180, max_workers=2),
        )
        dt = time.perf_counter() - t0
        print(f"   Hoàn tất trong {dt:.1f} giây\n")
        return {"scores": score, "detail_df": score.to_pandas()}

    except ImportError as e:
        print(f"   Thiếu thư viện: {e}")
        print("  → Chuyển sang chấm điểm thủ công đơn giản...")
        return step3_manual_evaluate(results)
    except Exception as e:
        print(f"   RAGAS lỗi: {e}")
        print("  → Chuyển sang chấm điểm thủ công...")
        return step3_manual_evaluate(results)


def step3_manual_evaluate(results: list) -> dict:
    """
    Chấm điểm thủ công (khi chưa cài được RAGAS).
    Đo 2 chỉ số cơ bản:
    - Context Hit Rate: % câu hỏi mà context có chứa từ khóa liên quan  
    - Answer Length Score: Câu trả lời có đủ dài/chi tiết không
    """
    from difflib import SequenceMatcher

    scores = []
    for r in results:
        # Context Hit: kiểm tra ground_truth có trùng khớp với context không
        gt_words = set(r["ground_truth"].lower().split())
        ctx_text = " ".join(r["contexts"]).lower()
        hit_count = sum(1 for w in gt_words if w in ctx_text)
        context_score = hit_count / max(len(gt_words), 1)

        # Answer similarity: so sánh answer với ground_truth
        answer_sim = SequenceMatcher(
            None, r["answer"].lower(), r["ground_truth"].lower()
        ).ratio()

        scores.append({
            "question": r["question"][:50],
            "context_hit_rate": round(context_score, 3),
            "answer_similarity": round(answer_sim, 3),
            "n_contexts": len(r["contexts"]),
            "answer_len": len(r["answer"]),
        })

    # Tính trung bình
    avg_ctx = sum(s["context_hit_rate"] for s in scores) / len(scores)
    avg_ans = sum(s["answer_similarity"] for s in scores) / len(scores)

    return {
        "scores": {"context_hit_rate": round(avg_ctx, 3), "answer_similarity": round(avg_ans, 3)},
        "detail": scores,
    }


def step4_report(results: list, eval_result: dict):
    """Xuất báo cáo ra file."""
    print("=" * 60)
    print("BƯỚC 4: XUẤT BÁO CÁO")
    print("=" * 60)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        log("=" * 60, f)
        log(" BASELINE EVALUATION REPORT - NGÀY 5", f)
        log(f"   Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}", f)
        log(f"   Session: {SESSION_ID}", f)
        log(f"   Số câu hỏi: {len(results)}", f)
        log(f"   Top K: {TOP_K}", f)
        log("=" * 60, f)

        # Điểm tổng
        log("\n ĐIỂM TỔNG:", f)
        scores = eval_result.get("scores", {})
        if hasattr(scores, "items"):
            for k, v in scores.items():
                if isinstance(v, (int, float)):
                    bar = "" * int(v * 20) + "░" * (20 - int(v * 20))
                    log(f"   {k:25s}: {v:.4f}  [{bar}]", f)
        else:
            log(f"   Scores: {scores}", f)

        # Chi tiết từng câu hỏi
        log("\n CHI TIẾT TỪNG CÂU:", f)
        log("-" * 60, f)

        detail = eval_result.get("detail", eval_result.get("detail_df", None))
        if detail is not None and hasattr(detail, "iterrows"):
            # RAGAS DataFrame
            for idx, row in detail.iterrows():
                log(f"\n  Q{idx+1}: {results[idx]['question'][:70]}...", f)
                for col in detail.columns:
                    if col not in ("question", "answer", "contexts", "ground_truth"):
                        log(f"       {col}: {row[col]:.4f}" if isinstance(row[col], float) else f"       {col}: {row[col]}", f)
        elif isinstance(detail, list):
            # Manual evaluation
            for i, s in enumerate(detail):
                log(f"\n  Q{i+1}: {s['question']}...", f)
                log(f"       Context Hit Rate : {s['context_hit_rate']}", f)
                log(f"       Answer Similarity: {s['answer_similarity']}", f)
                log(f"       Contexts Found   : {s['n_contexts']}", f)
                log(f"       Answer Length     : {s['answer_len']} chars", f)

        log("\n" + "=" * 60, f)
        log(" Báo cáo đã lưu vào: " + OUTPUT_FILE, f)

    print(f"\n File báo cáo: {OUTPUT_FILE}")


# ====================================================
# MAIN
# ====================================================
if __name__ == "__main__":
    # Load golden dataset
    with open(GOLDEN_DATASET, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f" Đã load {len(questions)} câu hỏi từ {GOLDEN_DATASET}\n")

    # Bước 1: Nạp papers (nếu chưa nạp)
    step1_ingest_papers()

    # Bước 2: Chạy RAG pipeline cho 20 câu hỏi  
    results = step2_run_rag_pipeline(questions)

    # Bước 3: Chấm điểm
    eval_result = step3_evaluate_ragas(results)

    # Bước 4: Xuất báo cáo
    step4_report(results, eval_result)

    print("\n🎉 HOÀN TẤT ĐÁNH GIÁ BASELINE NGÀY 5!")
