import json
import logging
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from app.settings import APPSETTINGS
from lc.prompts.templates import get_prompt_text  

logger = logging.getLogger(__name__)

# Số lần tối đa cho vòng lặp Critic (chống Infinite Loop)
MAX_CRITIC_ATTEMPTS = 3


# 1. ĐỊNH NGHĨA STATE
class GraphState(TypedDict):
    question: str
    session_id: str
    context: List[Dict[str, Any]]
    context_text: str
    draft_answer: str
    final_answer: str
    attempts: int
    next_action: str
    search_meta: Dict[str, Any]
    critic_score: str
    critic_feedback: str
    relevancy_score: str


# 2. KHỞI TẠO LLM
def _get_llm(temperature: float = 0) -> Any:
    # Nếu dùng LiteLLM Proxy (Chạy docker ở localhost:4000)
    if getattr(APPSETTINGS, "USE_LITELLM_PROXY", False):
        logger.info(f"Using LiteLLM Model Gateway (temp={temperature})")
        return ChatOpenAI(
            model="aura-llm-primary", # Tên model định nghĩa trong litellm_config.yaml
            api_key="sk-1234",        # Master key của proxy
            base_url="http://localhost:4000",
            temperature=temperature
        )
    
    # Mặc định gọi trực tiếp Gemini (Legacy)
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=APPSETTINGS.google_api_key,
        temperature=temperature,
    )


# 3. ROUTER NODE (Ngày 9)
ROUTER_PROMPT_TEMPLATE = get_prompt_text("router")


def router_node(state: GraphState) -> dict:
    """Phân loại câu hỏi → next_action = 'search' hoặc 'chat'."""
    question = state.get("question", "")
    logger.info(f"[Router] Phân tích: '{question[:80]}...'")

    prompt = ROUTER_PROMPT_TEMPLATE.format(question=question)

    try:
        llm = _get_llm(temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        # Xóa markdown wrapper
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        result = json.loads(content)
        action = result.get("action", "search")
        if action not in ("search", "chat"):
            action = "search"

    except Exception as e:
        logger.error(f"[Router] Lỗi: {e}. Fallback → 'search'")
        action = "search"

    logger.info(f"[Router] → '{action}'")
    return {"next_action": action}


# 4. RESEARCHER NODE 
def researcher_node(state: GraphState) -> dict:
    """
    Gọi advanced_retrieve() (BM25 + Qdrant + HyDE + Compression + Reorder).
    Nếu đang bị Critic đánh trượt (attempts > 1), log lại feedback để debug.
    """
    question = state.get("question", "")
    session_id = state.get("session_id", "")
    attempts = state.get("attempts", 0)
    critic_feedback = state.get("critic_feedback", "")

    # Log nếu đang retry do Critic đánh trượt
    if attempts > 0 and critic_feedback:
        logger.warning(
            f"[Researcher] ĐÃ BỊ CRITIC ĐÁNH TRƯỢT (lần {attempts}). "
            f"Feedback: '{critic_feedback[:100]}...'. Đang tìm lại..."
        )

    logger.info(f"[Researcher] Tìm kiếm lần {attempts + 1} cho session='{session_id}'")

    if not session_id:
        logger.error("[Researcher] Thiếu session_id!")
        return {
            "context": [],
            "context_text": "",
            "attempts": attempts + 1,
            "search_meta": {"error": "missing_session_id"},
        }

    try:
        from lc.chains.context_build import advanced_retrieve

        result = advanced_retrieve(
            session_id=session_id,
            q=question,
            k=APPSETTINGS.retrieval.top_k,
            use_hyde=APPSETTINGS.retrieval.hyde.enable,
            use_compress=APPSETTINGS.compression.enable,
            use_reorder=APPSETTINGS.reorder.enable,
        )

        docs = result.get("docs", [])
        context_text = result.get("context_joined", "")
        search_meta = {
            "hyde": result.get("hyde", {}),
            "compression": result.get("compression", {}),
            "reorder": result.get("reorder", {}),
            "total_docs_found": len(docs),
        }

        logger.info(f"[Researcher] Tìm được {len(docs)} tài liệu.")

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
# Ngày 17: Prompt đã được tách sang lc/prompts/templates.py (PromptOps)
CHAT_PROMPT_TEMPLATE = get_prompt_text("chat")


def chat_node(state: GraphState) -> dict:
    """Trả lời câu hỏi tán gẫu / xã giao."""
    question = state.get("question", "")
    logger.info(f"[Chat] Trả lời xã giao: '{question[:80]}...'")

    try:
        llm = _get_llm(temperature=0.7)
        prompt = CHAT_PROMPT_TEMPLATE.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()
    except Exception as e:
        logger.error(f"[Chat] Lỗi: {e}")
        answer = "Xin chào! Tôi là AURA — Trợ lý Học thuật. Bạn cần giúp gì?"

    return {"draft_answer": answer}


# 6. GENERATOR NODE 
# Ngày 17: Prompt đã được tách sang lc/prompts/templates.py (PromptOps)
GENERATOR_PROMPT_TEMPLATE = get_prompt_text("generator")


def generator_node(state: GraphState) -> dict:
    """
    Đọc context_text + question → viết draft_answer.

    Cách hoạt động:
    1. Lấy context_text (tài liệu đã nén từ Researcher) và question từ State.
    2. Xây prompt ép LLM chỉ dùng tài liệu, cấm bịa.
    3. Ghi câu trả lời vào draft_answer.
    """
    question = state.get("question", "")
    context_text = state.get("context_text", "")
    attempts = state.get("attempts", 0)

    logger.info(f"[Generator] Viết nháp (lần {attempts}), context={len(context_text)} chars")

    # Nếu không có context → ghi rõ lý do
    if not context_text.strip():
        logger.warning("[Generator] Context rỗng, không có tài liệu để viết.")
        return {
            "draft_answer": "Xin lỗi, tôi không tìm thấy tài liệu liên quan để trả lời câu hỏi này."
        }

    try:
        llm = _get_llm(temperature=0.3)  # Sáng tạo nhẹ nhưng vẫn bám sát tài liệu
        prompt = GENERATOR_PROMPT_TEMPLATE.format(
            context=context_text,
            question=question,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()
        logger.info(f"[Generator] Viết nháp xong: {len(answer)} chars")
    except Exception as e:
        logger.error(f"[Generator] Lỗi LLM: {e}")
        answer = "Đã có lỗi khi tạo câu trả lời. Vui lòng thử lại."

    return {"draft_answer": answer}


# 7. CRITIC NODE 
# Ngày 17: Prompt đã được tách sang lc/prompts/templates.py (PromptOps)
CRITIC_PROMPT_TEMPLATE = get_prompt_text("critic")


def critic_node(state: GraphState) -> dict:
    """
    Quan Tòa kiểm duyệt: Đối chiếu draft_answer với context_text.

    Cách hoạt động:
    1. Nhận draft_answer và context_text từ State.
    2. Gửi prompt "LLM-as-a-Judge" với temperature=0 (cực kỳ máy móc).
    3. Ép LLM trả về JSON: {"score": "pass/fail", "reason": "..."}.
    4. Ghi kết quả vào critic_score và critic_feedback.
    5. Nếu lỗi parse JSON → mặc định PASS (an toàn, tránh infinite loop).
    """
    draft_answer = state.get("draft_answer", "")
    context_text = state.get("context_text", "")
    attempts = state.get("attempts", 0)

    logger.info(f"[Critic] Kiểm duyệt câu trả lời (attempt={attempts})...")

    # Nếu context rỗng và answer đã từ chối → auto pass
    if not context_text.strip():
        logger.info("[Critic] Context rỗng → auto PASS (không có gì để so sánh)")
        return {"critic_score": "pass", "critic_feedback": "Context rỗng, answer đã từ chối đúng."}

    try:
        llm = _get_llm(temperature=0)  # Temperature = 0: Quan Tòa không được du di
        prompt = CRITIC_PROMPT_TEMPLATE.format(
            context=context_text,
            answer=draft_answer,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        # Xóa markdown wrapper
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        result = json.loads(content)
        score = result.get("score", "pass").lower()
        reason = result.get("reason", "")

        if score not in ("pass", "fail"):
            logger.warning(f"[Critic] Score không hợp lệ: '{score}'. Default → 'pass'")
            score = "pass"

    except json.JSONDecodeError as e:
        logger.error(f"[Critic] Không parse được JSON: {content!r}. Error: {e}. Default → 'pass'")
        score = "pass"
        reason = f"JSON parse error, default pass. Raw: {content[:100]}"
    except Exception as e:
        logger.error(f"[Critic] Lỗi: {e}. Default → 'pass'")
        score = "pass"
        reason = str(e)

    logger.info(f"[Critic] Kết quả: {score.upper()} | Reason: {reason[:100]}")
    return {"critic_score": score, "critic_feedback": reason}


# 8. RELEVANCY GRADER NODE  — Kiểm tra Context có liên quan không
# Ngày 17: Prompt đã được tách sang lc/prompts/templates.py (PromptOps)
RELEVANCY_PROMPT_TEMPLATE = get_prompt_text("relevancy")


def relevancy_grader_node(state: GraphState) -> dict:
    """
    Kiểm tra tài liệu Researcher tìm được có liên quan đến câu hỏi không.

    Ngày 12 — Lớp kiểm duyệt thứ 2:
    - Ngày 11: Critic kiểm tra "câu trả lời có bịa không" (sau Generator).
    - Ngày 12: Relevancy kiểm tra "tài liệu có đúng chủ đề không" (trước Generator).

    Cách hoạt động:
    1. Nhận context_text + question từ State.
    2. Gửi prompt cho LLM (temperature=0) đánh giá độ liên quan.
    3. Nếu "relevant" → tiếp tục sang Generator.
    4. Nếu "not_relevant" → nhảy sang Fallback (tránh viết nhảm).
    """
    question = state.get("question", "")
    context_text = state.get("context_text", "")

    logger.info(f"[Relevancy] Đánh giá độ liên quan context ({len(context_text)} chars) vs câu hỏi...")

    # Context rỗng → chắc chắn không liên quan
    if not context_text.strip():
        logger.warning("[Relevancy] Context rỗng → not_relevant")
        return {"relevancy_score": "not_relevant"}

    try:
        llm = _get_llm(temperature=0)
        prompt = RELEVANCY_PROMPT_TEMPLATE.format(
            context=context_text[:3000],  # Giới hạn context gửi cho Grader để tiết kiệm token
            question=question,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        # Xóa markdown wrapper
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        result = json.loads(content)
        score = result.get("score", "relevant").lower()
        reason = result.get("reason", "")

        if score not in ("relevant", "not_relevant"):
            score = "relevant"  # Default an toàn: cho qua

    except Exception as e:
        logger.error(f"[Relevancy] Lỗi: {e}. Default → 'relevant'")
        score = "relevant"
        reason = str(e)

    logger.info(f"[Relevancy] → {score.upper()} | {reason[:100]}")
    return {"relevancy_score": score}


# 9. FALLBACK NODE — Graceful Degradation 
def fallback_node(state: GraphState) -> dict:
    """
    Xử lý mềm mỏng khi hệ thống thất bại. Phân biệt 3 loại lỗi:
    1. Chưa upload tài liệu (session_id rỗng)
    2. Tài liệu không liên quan (relevancy fail)
    3. Hết lượt retry (critic fail x3)

    Ngày 14: Thêm error_type vào State để hệ thống giám sát (LangSmith)
    có thể phân loại lỗi tự động.
    """
    attempts = state.get("attempts", 0)
    relevancy = state.get("relevancy_score", "")
    session_id = state.get("session_id", "")

    if not session_id:
        error_type = "no_session"
        logger.warning("[Fallback] Chưa có session_id → chưa upload tài liệu.")
        msg = (
            "Bạn chưa tải lên tài liệu nào. Vui lòng upload file PDF/DOCX trước, "
            "sau đó đặt câu hỏi để tôi tra cứu giúp bạn."
        )
    elif relevancy == "not_relevant":
        error_type = "not_relevant"
        logger.warning("[Fallback] Tài liệu không liên quan đến câu hỏi.")
        msg = (
            " Xin lỗi bạn, tài liệu hiện có không chứa thông tin liên quan "
            "đến câu hỏi của bạn. Vui lòng tải lên tài liệu phù hợp hoặc "
            "đặt lại câu hỏi cụ thể hơn."
        )
    else:
        error_type = "max_retries"
        logger.warning(f"[Fallback] Hết {attempts} lần thử. Trả lời xin lỗi.")
        msg = (
            "Xin lỗi bạn, sau nhiều lần tra cứu, tôi không thể tìm được "
            "thông tin đáng tin cậy để trả lời câu hỏi này. Vui lòng thử hỏi "
            "lại với câu hỏi cụ thể hơn, hoặc kiểm tra xem tài liệu đã được "
            "tải lên chưa."
        )

    return {
        "draft_answer": msg,
        "critic_score": error_type,
    }


# 11. FORMATTER NODE — Định dạng và thêm trích dẫn
def formatter_node(state: GraphState) -> dict:
    """Định dạng câu trả lời cuối cùng và tự động thêm trích dẫn từ Metadata."""
    draft = state.get("draft_answer", "")
    critic_score = state.get("critic_score", "")
    context = state.get("context", [])
    next_action = state.get("next_action", "search")
    
    # Bỏ qua trích dẫn nếu đây là câu trả lời chat hoặc lỗi từ Fallback
    if next_action == "chat" or critic_score in ("no_session", "not_relevant", "max_retries"):
        return {"final_answer": draft}
        
    # Lọc trùng lặp (Deduplicate)
    citations = []
    seen = set()
    for doc in context:
        fname = doc.get("file_name")
        page = doc.get("page_idx")
        if fname:
            cite_str = str(fname)
            if page is not None:
                cite_str += f" - Trang {page}"
            if cite_str not in seen:
                seen.add(cite_str)
                citations.append(cite_str)
                
    final_answer = draft
    if citations:
        final_answer += "\n\n**Nguồn tham khảo:**\n"
        for c in citations:
            final_answer += f"- {c}\n"
            
    return {"final_answer": final_answer}


# 12. CONDITIONAL EDGES 
def route_decision(state: GraphState) -> str:
    """Router → 'search' hoặc 'chat'."""
    return state.get("next_action", "search")


def check_relevancy(state: GraphState) -> str:
    """
    Ngày 12: Kiểm tra kết quả Relevancy Grader.
      - "relevant"     : Tài liệu liên quan → cho Generator viết.
      - "not_relevant" : Tài liệu lạc đề → nhảy Fallback.
    """
    score = state.get("relevancy_score", "relevant")
    if score == "relevant":
        logger.info("[Decision] Relevancy PASS → tiến tới Generator.")
        return "relevant"
    else:
        logger.warning("[Decision] Relevancy FAIL → Fallback (tài liệu không liên quan).")
        return "not_relevant"


def check_hallucination(state: GraphState) -> str:
    """
    Ngày 11: Kiểm tra kết quả Critic Node.
      - "useful"      : Đáp án trung thực → END.
      - "not_useful"  : Đáp án bịa đặt, còn lượt → quay lại Researcher.
      - "max_retries" : Hết lượt → Fallback.
    """
    score = state.get("critic_score", "pass")
    attempts = state.get("attempts", 0)

    if score == "pass":
        logger.info("[Decision] Critic PASS → Kết thúc thành công.")
        return "useful"
    elif attempts >= MAX_CRITIC_ATTEMPTS:
        logger.warning(f"[Decision] Critic FAIL & đã hết {MAX_CRITIC_ATTEMPTS} lượt → Fallback.")
        return "max_retries"
    else:
        logger.warning(f"[Decision] Critic FAIL (lần {attempts}) → Quay lại Researcher.")
        return "not_useful"


# 12. BUILD GRAPH — Ráp nối toàn bộ thành FSM
def build_core_graph():
    workflow = StateGraph(GraphState)

    # ── Đăng ký tất cả các Node ──
    workflow.add_node("router", router_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("relevancy_grader", relevancy_grader_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("formatter", formatter_node)

    # ── Entry Point ──
    workflow.set_entry_point("router")

    # ── Router → (search | chat) ──
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {"search": "researcher", "chat": "chat"},
    )

    # ── Chat → Formatter (formatter sẽ bỏ qua trích dẫn nhờ next_action='chat') ──
    workflow.add_edge("chat", "formatter")

    # ── Researcher → Relevancy Grader ──
    workflow.add_edge("researcher", "relevancy_grader")

    # ── Relevancy Grader → (relevant | not_relevant) ──
    workflow.add_conditional_edges(
        "relevancy_grader",
        check_relevancy,
        {
            "relevant": "generator",
            "not_relevant": "fallback",
        },
    )

    # ── Generator → Critic ──
    workflow.add_edge("generator", "critic")

    # ── Critic → (useful | not_useful | max_retries) ──
    workflow.add_conditional_edges(
        "critic",
        check_hallucination,
        {
            "useful": "formatter",          # Pass → định dạng + trích dẫn
            "not_useful": "researcher",     # Fail → vòng lặp
            "max_retries": "fallback",      # Hết lượt → xin lỗi
        },
    )

    # ── Fallback → Formatter (xin lỗi cũng cần format lịch sự) ──
    workflow.add_edge("fallback", "formatter")

    # ── Formatter → END ──
    workflow.add_edge("formatter", END)

    return workflow.compile()


