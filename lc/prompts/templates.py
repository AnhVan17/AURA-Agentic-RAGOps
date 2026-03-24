"""
PromptOps — Prompt Templates tập trung
=================================================
Tách toàn bộ prompt ra khỏi graph.py, lưu tại đây.
Hỗ trợ 2 chế độ:
  1. LangSmith Hub (Remote) — ưu tiên dùng nếu Hub hoạt động.
  2. Local Fallback (Hardcode) — dùng khi Hub lỗi hoặc chưa push.

Cách dùng:
    from lc.prompts.templates import get_prompt
    prompt = get_prompt("router")
    # → trả về ChatPromptTemplate
"""

import logging
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# LOCAL FALLBACK PROMPTS — Bản gốc từ graph.py

ROUTER_TEMPLATE = """Bạn là Router điều hướng trong hệ thống Chatbot Học thuật.

NHIỆM VỤ: Phân loại câu hỏi của người dùng vào đúng 1 trong 2 loại:
- "chat": Câu chào hỏi, tán gẫu, cảm ơn, lời khen, giới thiệu, câu hỏi không cần tra cứu kiến thức.
- "search": Câu hỏi yêu cầu kiến thức chuyên môn, định nghĩa, công thức, giải thích khái niệm.

VÍ DỤ PHÂN LOẠI:
- "chào bạn", "xin chào", "hello", "hi", "hey" → "chat"
- "bạn là ai?", "who are you?", "tên bạn là gì?" → "chat"
- "cảm ơn", "thank you", "thanks", "tạm biệt", "bye" → "chat"
- "bạn khỏe không?", "how are you?", "what's up?" → "chat"
- "haha", "ok", "được rồi", "nice", "good" → "chat"
- "Machine Learning là gì?", "What is RAG?" → "search"
- "Giải thích thuật toán Transformer" → "search"
- "So sánh LSTM và GRU" → "search"
- "Tóm tắt bài báo về Attention" → "search"

QUY TẮC:
1. Nếu câu hỏi KHÔNG yêu cầu kiến thức chuyên môn → chọn "chat".
2. Nếu không chắc chắn → mặc định chọn "search".
3. Chỉ trả về đúng 1 chuỗi JSON, KHÔNG giải thích thêm.

ĐỊNH DẠNG BẮT BUỘC:
{{"action": "search"}} hoặc {{"action": "chat"}}

Câu hỏi: {question}"""


CHAT_TEMPLATE = """Bạn là trợ lý học thuật thân thiện, hỗ trợ sinh viên Việt Nam.
Hãy trả lời câu hỏi xã giao sau một cách ngắn gọn, lịch sự, bằng tiếng Việt.
Giới thiệu bản thân là "AURA - Trợ lý Học thuật Thông minh".

Câu hỏi: {question}"""


GENERATOR_TEMPLATE = """Bạn là trợ lý học thuật AURA. Nhiệm vụ: Viết câu trả lời DỰA HOÀN TOÀN vào tài liệu được cung cấp.

QUY TẮC TUYỆT ĐỐI:
1. CHỈ sử dụng thông tin có trong TÀI LIỆU bên dưới. TUYỆT ĐỐI KHÔNG bịa thêm.
2. Nếu TÀI LIỆU không đủ để trả lời → hãy nói rõ "Tài liệu chưa đề cập đến nội dung này."
3. Trả lời bằng tiếng Việt, rõ ràng, súc tích, phong cách học thuật.
4. Nếu có công thức toán → giữ nguyên ký hiệu gốc từ tài liệu.

TÀI LIỆU:
---
{context}
---

CÂU HỎI: {question}

CÂU TRẢ LỜI:"""


CRITIC_TEMPLATE = """Bạn là GIÁM KHẢO NGHIÊM KHẮC trong hệ thống Chatbot Học thuật.

NHIỆM VỤ: Kiểm tra xem CÂU TRẢ LỜI có TRUNG THÀNH (Faithful) với TÀI LIỆU không.

TIÊU CHÍ ĐÁNH GIÁ:
1. Mỗi khẳng định (claim) trong CÂU TRẢ LỜI phải có cơ sở từ TÀI LIỆU.
2. Nếu CÂU TRẢ LỜI chứa sự kiện, con số, định nghĩa KHÔNG THỂ suy ra từ TÀI LIỆU → FAIL.
3. Nếu CÂU TRẢ LỜI chỉ diễn đạt lại (paraphrase) nội dung TÀI LIỆU → PASS.
4. Nếu CÂU TRẢ LỜI nói "không tìm thấy tài liệu" hoặc từ chối trả lời → PASS (trung thực).

TÀI LIỆU:
---
{context}
---

CÂU TRẢ LỜI CẦN KIỂM TRA:
---
{answer}
---

TRẢ VỀ ĐÚNG 1 CHUỖI JSON DUY NHẤT (KHÔNG giải thích thêm):
{{"score": "pass", "reason": "lý do ngắn gọn"}}
hoặc
{{"score": "fail", "reason": "chỉ rõ khẳng định nào bịa đặt"}}"""


RELEVANCY_TEMPLATE = """Bạn là GIÁM KHẢO ĐÁNH GIÁ ĐỘ LIÊN QUAN trong hệ thống Chatbot Học thuật.

NHIỆM VỤ: Đánh giá xem TÀI LIỆU có chứa thông tin để trả lời CÂU HỎI hay không.

TIÊU CHÍ:
1. Nếu TÀI LIỆU có chứa ít nhất 1 thông tin trực tiếp liên quan đến CÂU HỎI → "relevant".
2. Nếu TÀI LIỆU hoàn toàn không liên quan, lạc đề, hoặc rỗng → "not_relevant".
3. Không cần TÀI LIỆU trả lời hoàn chỉnh, chỉ cần có nội dung liên quan là đủ.

TÀI LIỆU:
---
{context}
---

CÂU HỎI: {question}

TRẢ VỀ ĐÚNG 1 CHUỖI JSON DUY NHẤT:
{{"score": "relevant", "reason": "lý do ngắn gọn"}}
hoặc
{{"score": "not_relevant", "reason": "lý do ngắn gọn"}}"""


# REGISTRY — Ánh xạ tên → local template
_LOCAL_REGISTRY = {
    "router": ROUTER_TEMPLATE,
    "chat": CHAT_TEMPLATE,
    "generator": GENERATOR_TEMPLATE,
    "critic": CRITIC_TEMPLATE,
    "relevancy": RELEVANCY_TEMPLATE,
}

# HUB PULL — Load từ LangSmith Prompt Hub

# Mapping tên prompt → repo trên Hub
_HUB_REGISTRY = {
    "router": "anhvan/router-prompt",
    "chat": "anhvan/chat-prompt",
    "generator": "anhvan/generator-prompt",
    "critic": "anhvan/critic-prompt",
    "relevancy": "anhvan/relevancy-prompt",
}

# Cache để tránh gọi Hub nhiều lần trong cùng session
_hub_cache: dict = {}


def _pull_from_hub(prompt_name: str) -> Optional[ChatPromptTemplate]:
    """
    Thử load prompt từ LangSmith Hub.
    Trả về None nếu Hub lỗi hoặc chưa push prompt.
    """
    if prompt_name in _hub_cache:
        return _hub_cache[prompt_name]

    repo = _HUB_REGISTRY.get(prompt_name)
    if not repo:
        return None

    try:
        from langchain import hub
        prompt = hub.pull(repo)
        _hub_cache[prompt_name] = prompt
        logger.info(f"[PromptOps] Loaded '{prompt_name}' from Hub: {repo}")
        return prompt
    except Exception as e:
        logger.warning(
            f"[PromptOps] Hub pull failed for '{prompt_name}' ({repo}): {e}. "
            f"Falling back to local template."
        )
        return None


def get_prompt(prompt_name: str, use_hub: bool = True) -> ChatPromptTemplate:
    """
    Lấy prompt theo tên. Ưu tiên Hub, fallback về Local.

    Args:
        prompt_name: Tên prompt (router, chat, generator, critic, relevancy)
        use_hub: Có thử load từ Hub không (default: True)

    Returns:
        ChatPromptTemplate sẵn sàng .format()

    Usage:
        prompt = get_prompt("router")
        formatted = prompt.format(question="Machine Learning là gì?")
    """
    # 1. Thử Hub
    if use_hub:
        hub_prompt = _pull_from_hub(prompt_name)
        if hub_prompt is not None:
            return hub_prompt

    # 2. Fallback về Local
    local_template = _LOCAL_REGISTRY.get(prompt_name)
    if local_template is None:
        raise ValueError(
            f"[PromptOps] Unknown prompt: '{prompt_name}'. "
            f"Available: {list(_LOCAL_REGISTRY.keys())}"
        )

    logger.debug(f"[PromptOps] Using local template for '{prompt_name}'")
    return ChatPromptTemplate.from_template(local_template)


def get_prompt_text(prompt_name: str, use_hub: bool = False) -> str:
    """
    Trả về raw text của prompt (không phải ChatPromptTemplate).
    Dùng cho backward-compatible với graph.py hiện tại.

    Args:
        prompt_name: Tên prompt
        use_hub: default False (vì cần raw text, Hub trả ChatPromptTemplate)

    Returns:
        Raw template string
    """
    local_template = _LOCAL_REGISTRY.get(prompt_name)
    if local_template is None:
        raise ValueError(f"[PromptOps] Unknown prompt: '{prompt_name}'")
    return local_template


def clear_hub_cache():
    """Xóa cache Hub — hữu ích khi cần reload prompt mới."""
    global _hub_cache
    _hub_cache.clear()
    logger.info("[PromptOps] Hub cache cleared.")


def list_prompts() -> dict:
    """Liệt kê tất cả prompt có sẵn."""
    return {
        name: {
            "hub_repo": _HUB_REGISTRY.get(name, "N/A"),
            "local_available": name in _LOCAL_REGISTRY,
            "cached_from_hub": name in _hub_cache,
        }
        for name in _LOCAL_REGISTRY
    }
