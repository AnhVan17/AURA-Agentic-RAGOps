"""
Ngày 18: Feedback Collector — Human-in-the-Loop (Offline)
==========================================================
Thu thập phản hồi 👍/👎 từ người dùng.

Khi Thumbs Down:
  - Ghi toàn bộ state snapshot ra file JSON.
  - Gửi feedback lên LangSmith (nếu có run_id).
  
Dữ liệu feedback phục vụ:
  1. Phân tích lý do Agent trả lời sai.
  2. Cải thiện Prompt trên LangSmith Hub.
  3. Bổ sung test cases vào bộ đánh giá.
"""

import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Thư mục lưu feedback JSON
FEEDBACK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


def save_feedback(
    question: str,
    answer: str,
    score: str,  # "thumbs_up" hoặc "thumbs_down"
    session_id: str = "",
    state_snapshot: Optional[Dict[str, Any]] = None,
    user_comment: str = "",
    run_id: Optional[str] = None,
) -> str:
    """
    Lưu phản hồi của người dùng ra file JSON.

    Args:
        question: Câu hỏi gốc
        answer: Câu trả lời Agent đưa ra
        score: "thumbs_up" hoặc "thumbs_down"
        session_id: ID phiên làm việc
        state_snapshot: Toàn bộ GraphState tại thời điểm trả lời 
                        (context, critic_score, relevancy_score, search_meta...)
        user_comment: Lý do người dùng bấm 👎 (tuỳ chọn)
        run_id: LangSmith run ID để liên kết feedback

    Returns:
        Đường dẫn file JSON đã lưu
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    feedback_record = {
        "timestamp": timestamp,
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "score": score,
        "user_comment": user_comment,
        "run_id": run_id,
    }

    # Chỉ ghi state snapshot nếu Thumbs Down (vì cần debug)
    if score == "thumbs_down" and state_snapshot:
        # Lọc các field cần thiết để debug
        feedback_record["state_snapshot"] = {
            "context_text": state_snapshot.get("context_text", "")[:2000],  # Giới hạn
            "draft_answer": state_snapshot.get("draft_answer", ""),
            "final_answer": state_snapshot.get("final_answer", ""),
            "critic_score": state_snapshot.get("critic_score", ""),
            "critic_feedback": state_snapshot.get("critic_feedback", ""),
            "relevancy_score": state_snapshot.get("relevancy_score", ""),
            "attempts": state_snapshot.get("attempts", 0),
            "search_meta": state_snapshot.get("search_meta", {}),
            "next_action": state_snapshot.get("next_action", ""),
        }

    # Tạo filename
    safe_score = "up" if score == "thumbs_up" else "down"
    filename = f"fb_{timestamp}_{safe_score}_{session_id or 'nosession'}.json"
    filepath = FEEDBACK_DIR / filename

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(feedback_record, f, ensure_ascii=False, indent=2)
        logger.info(f"[Feedback] Saved: {filepath}")
    except Exception as e:
        logger.error(f"[Feedback] Failed to save: {e}")
        return ""

    # Gửi feedback lên LangSmith nếu có run_id
    if run_id:
        try:
            from ops.observability import log_feedback as ls_log_feedback
            ls_log_feedback(
                run_id=run_id,
                key="user_feedback",
                score=1.0 if score == "thumbs_up" else 0.0,
                comment=user_comment or f"User gave {score}",
            )
            logger.info(f"[Feedback] Sent to LangSmith: run_id={run_id}")
        except Exception as e:
            logger.warning(f"[Feedback] LangSmith log failed: {e}")

    return str(filepath)


def load_feedback_stats() -> Dict[str, Any]:
    """
    Đọc tóm tắt thống kê feedback.
    """
    up_count = 0
    down_count = 0
    total = 0

    try:
        for f in FEEDBACK_DIR.glob("fb_*.json"):
            total += 1
            if "_up_" in f.name:
                up_count += 1
            elif "_down_" in f.name:
                down_count += 1
    except Exception as e:
        logger.error(f"[Feedback] Failed to load stats: {e}")

    return {
        "total": total,
        "thumbs_up": up_count,
        "thumbs_down": down_count,
        "satisfaction_rate": round(up_count / total * 100, 1) if total > 0 else 0.0,
    }


def load_recent_feedback(limit: int = 20) -> list:
    """Đọc N feedback gần nhất (để hiển thị trên UI Admin)."""
    files = sorted(FEEDBACK_DIR.glob("fb_*.json"), reverse=True)[:limit]
    records = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                records.append(json.load(fp))
        except Exception:
            continue
    return records
