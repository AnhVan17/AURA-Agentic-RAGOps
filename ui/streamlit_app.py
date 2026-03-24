import os
import json
import streamlit as st
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
TIMEOUT = 60

st.set_page_config(
    page_title="AURA — Trợ lý Học thuật",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — Premium Dark Theme
# ============================================================
st.markdown("""
<style>
    /* Ẩn footer Streamlit */
    footer {visibility: hidden;}
    
    /* Main container */
    .main > div { padding-top: 1rem; }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 8px !important;
    }
    
    /* Feedback buttons */
    .feedback-btn {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        cursor: pointer;
        font-size: 14px;
        transition: all 0.2s;
        border: 1px solid #ddd;
        background: white;
        margin-right: 8px;
    }
    .feedback-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stat-card h2 { margin: 0; font-size: 2rem; }
    .stat-card p { margin: 4px 0 0; opacity: 0.9; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown { color: #e0e0e0; }
    
    /* Header gradient */
    .header-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS 
# ============================================================
def call_api(path: str, payload: dict, timeout=TIMEOUT) -> dict:
    """Gọi FastAPI backend."""
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Không kết nối được Backend. Hãy chạy: uvicorn app.main:app --reload"}
    except Exception as e:
        return {"error": str(e)}


def upload_file(session_id: str, file) -> dict:
    """Upload file lên backend."""
    try:
        url = f"{API_BASE}/session/upload"
        files = {"file": (file.name, file.getvalue(), "application/octet-stream")}
        params = {"session_id": session_id}
        r = requests.post(url, params=params, files=files, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def save_feedback_local(question, answer, score, session_id, state_snapshot=None, comment=""):
    """Ghi feedback ra JSON qua module ops.feedback."""
    try:
        from ops.feedback.collector import save_feedback
        filepath = save_feedback(
            question=question,
            answer=answer,
            score=score,
            session_id=session_id,
            state_snapshot=state_snapshot,
            user_comment=comment,
        )
        return filepath
    except Exception as e:
        st.error(f"Lỗi lưu feedback: {e}")
        return ""


# ============================================================
# SESSION STATE INIT
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "S1"
if "last_state" not in st.session_state:
    st.session_state.last_state = None
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("##  AURA")
    st.markdown("**Trợ lý Học thuật Thông minh**")
    st.markdown("---")
    
    # Session ID
    st.session_state.session_id = st.text_input(
        "🔑 Session ID", 
        value=st.session_state.session_id,
        help="ID phiên làm việc. Mỗi session có bộ tài liệu riêng."
    )
    
    # File upload
    st.markdown("### Tải tài liệu")
    uploaded_files = st.file_uploader(
        "Chọn file PDF/DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload tài liệu để AURA tra cứu và trả lời."
    )
    
    if uploaded_files:
        if st.button("📤 Ingest tài liệu", type="primary", use_container_width=True):
            for f in uploaded_files:
                with st.spinner(f"Đang xử lý {f.name}..."):
                    result = upload_file(st.session_state.session_id, f)
                    if "error" in result:
                        st.error(f" {f.name}: {result['error']}")
                    else:
                        chunks = result.get("counts", {}).get("chunks", "?")
                        st.success(f" {f.name} → {chunks} chunks")
    
    st.markdown("---")
    
    # Feedback Stats
    st.markdown("###  Thống kê Feedback")
    try:
        from ops.feedback.collector import load_feedback_stats
        stats = load_feedback_stats()
        col1, col2 = st.columns(2)
        col1.metric("👍", stats["thumbs_up"])
        col2.metric("👎", stats["thumbs_down"])
        if stats["total"] > 0:
            st.progress(stats["satisfaction_rate"] / 100)
            st.caption(f"Tỷ lệ hài lòng: {stats['satisfaction_rate']}%")
    except Exception:
        st.caption("Chưa có dữ liệu feedback")
    
    st.markdown("---")
    
    # Clear chat
    if st.button("🗑️ Xoá lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_state = None
        st.session_state.feedback_given = set()
        st.rerun()


# ============================================================
# MAIN CHAT AREA
# ============================================================
st.markdown('<p class="header-gradient"> AURA — Trợ lý Học thuật</p>', unsafe_allow_html=True)
st.caption("Hỏi đáp dựa trên tài liệu của bạn • Có nút 👍👎 để phản hồi chất lượng")

# Hiển thị lịch sử chat
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Hiển thị nút feedback cho tin nhắn của assistant
        if msg["role"] == "assistant" and i not in st.session_state.feedback_given:
            col1, col2, col3 = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key=f"up_{i}", help="Câu trả lời tốt"):
                    question = st.session_state.messages[i - 1]["content"] if i > 0 else ""
                    save_feedback_local(
                        question=question,
                        answer=msg["content"],
                        score="thumbs_up",
                        session_id=st.session_state.session_id,
                    )
                    st.session_state.feedback_given.add(i)
                    st.toast(" Cảm ơn phản hồi!", icon="👍")
                    st.rerun()
            with col2:
                if st.button("👎", key=f"down_{i}", help="Câu trả lời chưa tốt"):
                    question = st.session_state.messages[i - 1]["content"] if i > 0 else ""
                    save_feedback_local(
                        question=question,
                        answer=msg["content"],
                        score="thumbs_down",
                        session_id=st.session_state.session_id,
                        state_snapshot=st.session_state.last_state,
                    )
                    st.session_state.feedback_given.add(i)
                    st.toast(" Đã ghi nhận. Cảm ơn bạn!", icon="👎")
                    st.rerun()

# Input chat
if prompt := st.chat_input("Đặt câu hỏi về tài liệu của bạn..."):
    # Hiển thị tin nhắn user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gọi API
    with st.chat_message("assistant"):
        with st.spinner("AURA đang suy nghĩ..."):
            payload = {
                "session_id": st.session_state.session_id,
                "question": prompt,
                "k": 8,
                "lang": "vi",
            }
            result = call_api("/ask", payload)

            if "error" in result:
                answer = f" Lỗi: {result['error']}"
                st.error(answer)
            else:
                answer = result.get("answer", "Không có câu trả lời.")
                footnotes = result.get("footnotes", [])
                
                # Format footnotes
                if footnotes:
                    answer += "\n\n---\n**Nguồn tham khảo:**\n"
                    for fn in footnotes:
                        heading = fn.get("heading", "(no heading)")
                        page = fn.get("page")
                        page_str = f" — trang {page}" if page else ""
                        answer += f"- [{fn['n']}] {heading}{page_str}\n"

                st.markdown(answer)

                # Lưu state snapshot để debug nếu user bấm 👎
                st.session_state.last_state = result.get("state_snapshot", {})

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
