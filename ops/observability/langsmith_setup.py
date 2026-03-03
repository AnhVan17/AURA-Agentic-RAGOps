from __future__ import annotations
import os
import logging
from functools import wraps
from typing import Optional, Callable

logger = logging.getLogger(__name__)


_langsmith_initialized = False
_tracer = None


def init_langsmith(
    project_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bool:
    """
    Khởi tạo LangSmith tracing.
    
    Gọi hàm này 1 lần khi app khởi động (vd: trong main.py).
    
    Args:
        project_name: Tên project trên LangSmith (mặc định từ env)
        api_key: API key (mặc định từ LANGCHAIN_API_KEY)
    
    Returns:
        True nếu khởi tạo thành công
    """
    global _langsmith_initialized, _tracer
    
    # Đã khởi tạo rồi
    if _langsmith_initialized:
        logger.debug("LangSmith đã được khởi tạo trước đó")
        return True
    
    # Set environment variables
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    
    if api_key:
        os.environ["LANGCHAIN_API_KEY"] = api_key
    
    if project_name:
        os.environ["LANGCHAIN_PROJECT"] = project_name
    else:
        os.environ.setdefault("LANGCHAIN_PROJECT", "academic-rag-chatbot")
    
    # Kiểm tra API key
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    
    if not langchain_api_key or langchain_api_key == "your_langsmith_api_key_here":
        # TẮT tracing để tránh lỗi 403
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.warning(
            "LANGCHAIN_API_KEY chưa được cấu hình. "
            "Truy cập https://smith.langchain.com/ để lấy API key."
        )
        return False
    
    if not tracing_enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.warning("LANGCHAIN_TRACING_V2 không phải 'true'. Tracing bị tắt.")
        return False
    
    # Thử kết nối LangSmith
    try:
        from langsmith import Client
        
        _tracer = Client()
        _langsmith_initialized = True
        
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        logger.info(f"LangSmith đã sẵn sàng! Project: {project}")
        logger.info(f"Dashboard: https://smith.langchain.com/")
        
        return True
        
    except ImportError:
        logger.error("Package langsmith chưa cài. Chạy: pip install langsmith")
        return False
    except Exception as e:
        logger.error(f"Lỗi kết nối LangSmith: {e}")
        return False


def is_tracing_enabled() -> bool:
    """Kiểm tra tracing đã được bật chưa."""
    return _langsmith_initialized


def trace_chain(name: str):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Nếu chưa init, chạy bình thường không trace
            if not _langsmith_initialized:
                return func(*args, **kwargs)
            
            try:
                from langchain_core.tracers.context import tracing_v2_enabled
                with tracing_v2_enabled(project_name=os.getenv("LANGCHAIN_PROJECT")):
                    return func(*args, **kwargs)
            except Exception as e:
                logger.debug(f"Trace error (không ảnh hưởng): {e}")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def log_feedback(
    run_id: str,
    key: str,
    score: float,
    comment: Optional[str] = None,
) -> bool:
    if not _langsmith_initialized or not _tracer:
        logger.warning("LangSmith chưa khởi tạo. Feedback không được log.")
        return False
    
    try:
        _tracer.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            comment=comment,
        )
        logger.info(f"Đã log feedback: {key}={score}")
        return True
    except Exception as e:
        logger.error(f"Lỗi log feedback: {e}")
        return False
