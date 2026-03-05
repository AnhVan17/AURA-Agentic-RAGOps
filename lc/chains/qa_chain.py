import time
import logging
import os
from typing import Dict, Any

from app.settings import APPSETTINGS
from lc.chains.context_build import advanced_retrieve
from core.citation.citation import build_citation_context
from core.guardrails.guardrails import should_abstain_for_qa
from core.chunking.tokens import count_tokens
from core.chunking.chunk import detect_lang_fast
from lc.cache import cached_advanced_retrieve


from ops.observability import trace_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


logger = logging.getLogger(__name__)


def _call_llm(prompt: str, max_tokens: int = 512) -> str:
    """
    Calls the Gemini LLM using LangChain's abstraction for better tracing.
    """
    try:
        api_key = APPSETTINGS.google_api_key
        if not api_key:
            logger.error("Google API Key is missing in settings.")
            return "[Error] API Key missing."

        model_name = getattr(APPSETTINGS.toy, "model", "gemini-2.5-flash")
        
        # Tạo LLM qua LangChain để tự động trace sang LangSmith
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.4,
            max_output_tokens=max_tokens,
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        if response and response.content:
            full_text = response.content.strip()
            logger.info(f"LLM generated {len(full_text)} chars ({count_tokens(full_text)} tokens).")
            return full_text
        else:
            logger.warning("LLM returned an empty response.")
            return "Hệ thống không nhận được phản hồi trọn vẹn từ AI. Vui lòng thử lại."
            
    except Exception as e:
        logger.error(f"Error calling LLM: {str(e)}", exc_info=True)
        return f"[Error] LLM failed: {str(e)}"

@trace_chain("qa_with_citation")
def answer_with_citation(session_id: str, question: str, k: int = 8) -> Dict[str, Any]:
    """
    Main entry point for QA with citations. Optimized for Bilingual Support.
    """
    t0 = time.perf_counter()
    
    # Phát hiện ngôn ngữ câu hỏi để trả về thông báo lỗi/fallback phù hợp
    lang = detect_lang_fast(question)
    
    # 1. Retrieval
    try:
        # Sử dụng tham số k=12 để lấy nhiều ngữ cảnh hơn cho câu trả lời chi tiết
        effective_k = k if k != 8 else 12 
        ar = advanced_retrieve(session_id, question, k=effective_k, use_hyde=True, use_compress=True, use_reorder=True)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        err_msg = "Lỗi tìm kiếm tài liệu." if lang == "vi" else "Retrieval error."
        return {"answer": err_msg, "stage_ms": {"retrieve_ms": 0, "qa_ms": 0}}
        
    t_retr = (time.perf_counter() - t0) * 1000

    docs = ar["docs"]
    context_marked, footnotes = build_citation_context(docs)
    ctx_tokens = count_tokens(context_marked)

    # 2. Guardrails check
    guard_cfg = getattr(APPSETTINGS, "guardrail", {})
    if hasattr(guard_cfg, "dict"): guard_cfg = guard_cfg.dict()
    
    should_stop, reason = should_abstain_for_qa(len(docs), ctx_tokens, guard_cfg)
    
    if should_stop:
        logger.info(f"Guardrail triggered for session {session_id}: {reason}")
        msg = "Không đủ thông tin để trả lời chính xác từ nguồn đã cung cấp." if lang == "vi" else "I don't have enough information from the provided source to answer this accurately."
        return {
            "answer": msg,
            "citations": [],
            "footnotes": [],
            "stage_ms": {"retrieve_ms": round(t_retr, 2), "qa_ms": 0.0}
        }

    # 3. Prompt Build
    prompt_path = os.path.join("lc", "prompt", "qa_v1.txt")
    if not os.path.exists(prompt_path):
        logger.error(f"Prompt file not found: {prompt_path}")
        return {"answer": "[System Error] Prompt template missing.", "stage_ms": {"retrieve_ms": t_retr, "qa_ms": 0}}

    with open(prompt_path, "r", encoding="utf-8") as f:
        qa_tmpl = f.read()
    
    full_prompt = qa_tmpl.format(question=question, context_with_markers=context_marked)

    # 4. LLM Generation
    t1 = time.perf_counter()
    # Ép buộc sử dụng 2048 tokens để trả lời dài và chi tiết
    max_out = 2048
    answer = _call_llm(full_prompt, max_tokens=max_out)
    t_qa = (time.perf_counter() - t1) * 1000

    return {
        "answer": answer,
        "citations": [{"n": f["n"]} for f in footnotes],
        "footnotes": footnotes,
        "stage_ms": {
            "retrieve_ms": round(t_retr, 2),
            "qa_ms": round(t_qa, 2)
        },
        "lang": lang
    }
