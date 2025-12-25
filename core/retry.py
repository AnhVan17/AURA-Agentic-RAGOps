import time
import random
import logging
import asyncio
from functools import wraps
from typing import Any, Callable, Type, Tuple, Union
from app.settings import APPSETTINGS

logger = logging.getLogger(__name__)

# Các lỗi mặc định nên thử lại (Network, Timeout, Rate Limit)
DEFAULT_RETRYABLE = (TimeoutError, ConnectionError, RuntimeError)

def retryable(
    max_attempts: int = None, 
    base_delay: float = None, 
    max_delay: float = None,
    exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRYABLE
):
    """
    Decorator đa năng hỗ trợ cả hàm đồng bộ (sync) và bất đồng bộ (async).
    Thêm Exponential Backoff, Jitter và Logging để tối ưu cho production.
    """
    # Lấy thông tin từ config nếu không được truyền trực tiếp
    retry_cfg = APPSETTINGS.retry
    attempts = max_attempts or retry_cfg.get("max_attempts", 3)
    base = base_delay or retry_cfg.get("base_delay", 0.5)
    limit = max_delay or retry_cfg.get("max_delay", 5.0)

    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                current_attempt = 0
                while True:
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        current_attempt += 1
                        if current_attempt >= attempts:
                            logger.error(f"[Retry] Async '{func.__name__}' failed after {attempts} attempts. Error: {e}")
                            raise
                        
                        # Tính toán delay với Exponential Backoff + Jitter
                        delay = min(limit, base * (2 ** (current_attempt - 1))) + (random.random() * 0.1)
                        logger.warning(f"[Retry] Async '{func.__name__}' attempt {current_attempt}/{attempts} failed. Retrying in {delay:.2f}s... (Error: {e})")
                        await asyncio.sleep(delay)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                current_attempt = 0
                while True:
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        current_attempt += 1
                        if current_attempt >= attempts:
                            logger.error(f"[Retry] Sync '{func.__name__}' failed after {attempts} attempts. Error: {e}")
                            raise
                        
                        delay = min(limit, base * (2 ** (current_attempt - 1))) + (random.random() * 0.1)
                        logger.warning(f"[Retry] Sync '{func.__name__}' attempt {current_attempt}/{attempts} failed. Retrying in {delay:.2f}s... (Error: {e})")
                        time.sleep(delay)
            return sync_wrapper

    return decorator

