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

from ops.observability import trace_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
import redis
from langchain_community.cache import RedisSemanticCache
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

# ===== BƯỚC 1: SETUP SEMANTIC CACHING & MEMORY STORE =====

_redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
_use_redis = False
_memory_store = {}

def init_globals():
    """Khởi tạo RedisCache cho toàn hệ thống nếu Redis có sẵn. Vô cùng hữu ích cho Semantic Cache."""
    global _use_redis
    try:
        r = redis.Redis.from_url(_redis_url)
        r.ping()
        _use_redis = True
        logger.info("✅ Redis connected! Enabling Redis Semantic Caching & Chat History.")
        
        api_key = APPSETTINGS.google_api_key
        if api_key:
            # Semantic Caching với Google Embeddings & Redis
            # Giúp trả lời siêu tốc nếu câu hỏi ý nghĩa tương tự
            set_llm_cache(RedisSemanticCache(
                redis_url=_redis_url,
                embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key),
                score_threshold=0.1
            ))
    except Exception as e:
        logger.warning(f"⚠️ Redis is not available locally. Defaulting to InMemoryCache. (Error: {e})")
        set_llm_cache(InMemoryCache())

# Kích hoạt cache khi module load
init_globals()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Hỗ trợ RedisChatMessageHistory (memory siêu tốc) hoặc Fallback local memory."""
    global _use_redis
    if _use_redis:
        try:
            from langchain_community.chat_message_histories import RedisChatMessageHistory
            return RedisChatMessageHistory(session_id, url=_redis_url)
        except Exception:
            pass
            
    # Fallback to local dict memory for testing without Redis server
    from langchain_community.chat_message_histories import ChatMessageHistory
    if session_id not in _memory_store:
        _memory_store[session_id] = ChatMessageHistory()
    return _memory_store[session_id]


# ===== BƯỚC 2: XÂY DỰNG LCEL GRAPH (Retrieval -> Compression -> LLM -> History) =====

def _build_qa_chain():
    prompt_path = os.path.join("lc", "prompt", "qa_v1.txt")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            template_str = f.read()
    else:
        template_str = "Answer based on context.\nContext: {context_with_markers}\nQuestion: {question}"

    # ChatPromptTemplate hỗ trợ inject History vào giữa (LCEL logic)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional academic research assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", template_str)
    ])

    api_key = APPSETTINGS.google_api_key
    model_name = getattr(APPSETTINGS.toy, "model", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key if api_key else "dummy_key", 
        temperature=0.4,
        max_output_tokens=2048,
    )

    # Nối LCEL cơ bản (Prompt -> LLM -> Parse String)
    chain = prompt | llm | StrOutputParser()

    # Nối tiếp MessageHistory (Bộ nhớ đệm thông minh)
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    return chain_with_history

# Tạo Singleton Chain
_qa_chain = _build_qa_chain()


@trace_chain("qa_with_citation")
def answer_with_citation(session_id: str, question: str, k: int = 8) -> Dict[str, Any]:
    """
    Main entry point for QA with citations.
    Sử dụng LCEL Graph cho luồng xử lý trơn tru và hỗ trợ Memory.
    """
    t0 = time.perf_counter()
    lang = detect_lang_fast(question)
    
    # 1. Retrieval & Compression (Day 7 logic nằm bên trong)
    try:
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

    # 3 & 4. LLM Generation via LCEL (Includes Memory & Cache)
    t1 = time.perf_counter()
    try:
        # Gọi chain.invoke() thay vì _call_llm thủ công
        answer = _qa_chain.invoke(
            {"question": question, "context_with_markers": context_marked},
            config={"configurable": {"session_id": session_id}}
        )
    except Exception as e:
        logger.error(f"Error calling LCEL QA Chain: {str(e)}", exc_info=True)
        answer = f"[Error] LLM failed: {str(e)}"
        
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
