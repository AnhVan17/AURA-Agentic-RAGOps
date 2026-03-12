import json
import logging
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from app.settings import APPSETTINGS

logger = logging.getLogger(__name__)



# 1. ĐỊNH NGHĨA STATE (Bộ nhớ trung tâm của Graph)

class GraphState(TypedDict):
    """
    Cuốn sổ tay truyền tay giữa các Node trong Graph.

    Fields:
        question     : Câu hỏi gốc của người dùng.
        session_id   : ID phiên làm việc, dùng để truy vấn đúng Collection trên Qdrant / BM25.
        context      : Danh sách tài liệu (dạng dict) mà Researcher tìm được.
        context_text : Chuỗi text đã nối (join) sẵn, dùng trực tiếp cho prompt LLM.
        draft_answer : Câu trả lời nháp (chat hoặc generator ghi vào).
        attempts     : Số lần đã thử tìm kiếm (phòng trường hợp retry).
        next_action  : Quyết định của Router ("search" hoặc "chat").
        search_meta  : Metadata bổ sung từ quá trình tìm kiếm (hyde, compression, reorder info).
    """
    question: str
    session_id: str
    context: List[Dict[str, Any]]
    context_text: str
    draft_answer: str
    attempts: int
    next_action: str
    search_meta: Dict[str, Any]



# 2. KHỞI TẠO LLM DÙNG CHUNG
def _get_llm() -> ChatGoogleGenerativeAI:
    """Tạo LLM instance (lazy, tránh lỗi khi import nếu chưa có API key)."""
    return ChatGoogleGenerativeAI(
        model=APPSETTINGS.app.default_llm,
        google_api_key=APPSETTINGS.google_api_key,
        temperature=0, 
    )


# 3. ROUTER NODE 
ROUTER_PROMPT_TEMPLATE = """Bạn là Router điều hướng trong hệ thống Chatbot Học thuật tiếng Việt.

NHIỆM VỤ: Phân loại câu hỏi của người dùng vào đúng 1 trong 2 loại:
- "chat": Câu chào hỏi, tán gẫu, cảm ơn, lời khen, câu hỏi không cần tra cứu kiến thức (VD: "Xin chào", "Bạn là ai?", "Cảm ơn nhé").
- "search": Câu hỏi yêu cầu kiến thức chuyên môn, định nghĩa, công thức, giải thích khái niệm, tìm thông tin sự thật (VD: "RAG là gì?", "Giải thích thuật toán Gradient Descent", "Cho tôi công thức tính đạo hàm").

QUY TẮC:
1. Nếu không chắc chắn → mặc định chọn "search" (an toàn hơn).
2. Chỉ trả về đúng 1 chuỗi JSON, KHÔNG giải thích thêm.

ĐỊNH DẠNG TRẢ VỀ BẮT BUỘC (chỉ 1 dòng JSON duy nhất):
{{"action": "search"}} hoặc {{"action": "chat"}}

Câu hỏi: {question}"""


def router_node(state: GraphState) -> dict:
    """
    Phân loại câu hỏi → quyết định next_action = "search" hoặc "chat".
    
    Cách hoạt động:
    1. Lấy question từ State.
    2. Gửi prompt cho LLM với temperature=0 (ổn định).
    3. Ép LLM trả về JSON: {"action": "search"} hoặc {"action": "chat"}.
    4. Parse JSON → ghi vào next_action trong State.
    5. Nếu lỗi parse → fallback an toàn sang "search".
    """
    question = state.get("question", "")
    logger.info(f"[Router] Phân tích câu hỏi: '{question[:80]}...'")

    prompt = ROUTER_PROMPT_TEMPLATE.format(question=question)

    try:
        llm = _get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        # Xóa markdown wrapper nếu LLM tự thêm ```json ... ```
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        result = json.loads(content)
        action = result.get("action", "search")

        # Validate: chỉ chấp nhận "search" hoặc "chat"
        if action not in ("search", "chat"):
            logger.warning(f"[Router] Action không hợp lệ: '{action}'. Fallback → 'search'")
            action = "search"

    except json.JSONDecodeError as e:
        logger.error(f"[Router] Không parse được JSON từ LLM: {content!r}. Error: {e}")
        action = "search"
    except Exception as e:
        logger.error(f"[Router] Lỗi không mong muốn: {e}. Fallback → 'search'")
        action = "search"

    logger.info(f"[Router] Quyết định: '{action}'")
    return {"next_action": action}


# 4. RESEARCHER NODE 
def researcher_node(state: GraphState) -> dict:
    """
    Kích hoạt toàn bộ pipeline Retrieval của Tuần 2 để tìm tài liệu.
    
    Cách hoạt động:
    1. Lấy question + session_id từ State.
    2. Gọi advanced_retrieve() — hàm đã gói gọn:
       - Dense search (Qdrant vector similarity)
       - Sparse search (BM25 keyword matching)
       - Ensemble Merge (trộn điểm Min-Max)
       - HyDE (Hypothetical Document Embedding — nếu kết quả ban đầu kém)
       - Compression (Regex lọc bỏ câu thừa, giảm token)
       - Reorder (Sắp xếp ngắn→dài theo heading)
    3. Nhét kết quả vào context (list docs) + context_text (chuỗi join sẵn).
    4. Lưu metadata (HyDE có dùng không, compression ratio...) vào search_meta.
    """
    question = state.get("question", "")
    session_id = state.get("session_id", "")
    attempts = state.get("attempts", 0)

    logger.info(f"[Researcher] Tìm kiếm tài liệu cho session='{session_id}', q='{question[:80]}...'")

    if not session_id:
        logger.error("[Researcher] Thiếu session_id! Không thể truy vấn.")
        return {
            "context": [],
            "context_text": "",
            "attempts": attempts + 1,
            "search_meta": {"error": "missing_session_id"},
        }

    try:
        # Import hàm siêu cấp từ Tuần 2
        from lc.chains.context_build import advanced_retrieve

        result = advanced_retrieve(
            session_id=session_id,
            q=question,
            k=APPSETTINGS.retrieval.top_k,
            use_hyde=APPSETTINGS.retrieval.hyde.enable,
            use_compress=APPSETTINGS.compression.enable,
            use_reorder=APPSETTINGS.reorder.enable,
        )

        # Trích xuất kết quả
        docs = result.get("docs", [])
        context_text = result.get("context_joined", "")

        search_meta = {
            "hyde": result.get("hyde", {}),
            "compression": result.get("compression", {}),
            "reorder": result.get("reorder", {}),
            "total_docs_found": len(docs),
        }

        logger.info(
            f"[Researcher] Tìm được {len(docs)} tài liệu. "
            f"HyDE={'có' if search_meta['hyde'].get('used') else 'không'}, "
            f"Compression ratio={search_meta['compression'].get('ratio', 0)}"
        )

    except Exception as e:
        logger.error(f"[Researcher] Pipeline lỗi: {e}", exc_info=True)
        docs = []
        context_text = ""
        search_meta = {"error": str(e)}

    return {
        "context": docs,
        "context_text": context_text,
        "attempts": attempts + 1,
        "search_meta": search_meta,
    }


# 5. CHAT NODE 

CHAT_PROMPT_TEMPLATE = """Bạn là trợ lý học thuật thân thiện, hỗ trợ sinh viên Việt Nam.
Hãy trả lời câu hỏi xã giao sau một cách ngắn gọn, lịch sự, bằng tiếng Việt.
Giới thiệu bản thân là "AURA - Trợ lý Học thuật Thông minh".

Câu hỏi: {question}"""


def chat_node(state: GraphState) -> dict:
    """
    Xử lý câu hỏi tán gẫu / xã giao.
    Gọi LLM trả lời đơn giản, không cần tìm tài liệu.
    """
    question = state.get("question", "")
    logger.info(f"[Chat] Trả lời câu hỏi xã giao: '{question[:80]}...'")

    try:
        llm = _get_llm()
        prompt = CHAT_PROMPT_TEMPLATE.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()
    except Exception as e:
        logger.error(f"[Chat] Lỗi LLM: {e}")
        answer = "Xin chào! Tôi là AURA — Trợ lý Học thuật. Bạn cần giúp gì?"

    return {"draft_answer": answer}



# 6. CONDITIONAL EDGE — Hàm bẻ ghi đường ray

def route_decision(state: GraphState) -> str:
    """Đọc next_action trong State để quyết định rẽ nhánh."""
    return state.get("next_action", "search")


# 7. BUILD GRAPH — Ráp nối toàn bộ thành FSM

def build_core_graph() -> Any:
    """
    Xây dựng và compile LangGraph.
    
    Sơ đồ:
        START → router
                  ├── "chat"   → chat_node   → END
                  └── "search" → researcher  → END
    
    Returns:
        CompiledGraph — có thể gọi .invoke(state) trực tiếp.
    """
    workflow = StateGraph(GraphState)

    # Đăng ký các Node
    workflow.add_node("router", router_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("chat", chat_node)

    # Thiết lập Entry Point (Điểm bắt đầu luôn là Router)
    workflow.set_entry_point("router")

    # Conditional Edge: Router → (search | chat)
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "search": "researcher",
            "chat": "chat",
        },
    )

    # Terminal Edges: Cả hai nhánh đều kết thúc
    workflow.add_edge("researcher", END)
    workflow.add_edge("chat", END)

    return workflow.compile()



# 8. TEST CHẠY THỬ
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    graph = build_core_graph()

    # --- Test 1: Câu hỏi tán gẫu ---
    print("\n" + "=" * 60)
    print("TEST 1: Câu hỏi tán gẫu")
    print("=" * 60)
    state_chat = {
        "question": "Chào buổi sáng! Bạn khỏe không?",
        "session_id": "",
        "context": [],
        "context_text": "",
        "draft_answer": "",
        "attempts": 0,
        "next_action": "",
        "search_meta": {},
    }
    result_chat = graph.invoke(state_chat)
    print(f"  → Action:  {result_chat['next_action']}")
    print(f"  → Answer:  {result_chat['draft_answer'][:200]}")

    # --- Test 2: Câu hỏi kiến thức (cần session_id thật để chạy Retriever) ---
    print("\n" + "=" * 60)
    print("TEST 2: Câu hỏi kiến thức (không có session → sẽ báo lỗi nhẹ)")
    print("=" * 60)
    state_search = {
        "question": "Giải thích khái niệm Gradient Descent trong Machine Learning?",
        "session_id": "",  # Để trống → Researcher sẽ báo lỗi nhẹ
        "context": [],
        "context_text": "",
        "draft_answer": "",
        "attempts": 0,
        "next_action": "",
        "search_meta": {},
    }
    result_search = graph.invoke(state_search)
    print(f"  → Action:       {result_search['next_action']}")
    print(f"  → Docs found:   {len(result_search['context'])}")
    print(f"  → Search meta:  {result_search['search_meta']}")
