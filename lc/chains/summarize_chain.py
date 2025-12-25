from __future__ import annotations
import logging
import os
import time
from typing import Dict, Any

from app.settings import APPSETTINGS
from lc.chains.context_build import advanced_retrieve

logger = logging.getLogger(__name__)

def _call_llm(prompt: str, max_tokens: int = 512) -> str:
    """
    Internal LLM caller for summarizing.
    """
    try:
        import google.generativeai as genai
        api_key = APPSETTINGS.google_api_key
        if not api_key:
            logger.error("Google API Key is missing in settings.")
            return "[Error] API Key missing."

        genai.configure(api_key=api_key)
        model_name = getattr(APPSETTINGS.toy, "model", "gemini-2.5-flash")
        
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        model = genai.GenerativeModel(model_name)
        
        safety = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        response = model.generate_content(
            prompt, 
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.2},
            safety_settings=safety
        )
        
        if response and response.text:
            return response.text.strip()
        else:
            logger.warning("LLM returned an empty summary response or was blocked.")
            return "Hệ thống không thể tạo bản tóm tắt trọn vẹn tại thời điểm này."
            
    except Exception as e:
        logger.error(f"Error calling LLM for summary: {str(e)}", exc_info=True)
        return f"[Error] Summary failed: {str(e)}"

def summarize_mode(session_id: str, mode: str, question: str | None = None, k: int = 10) -> Dict[str, Any]:
    """
    Summarizes content based on different modes (tldr, executive, qfs).
    """
    t0 = time.perf_counter()
    
    # 1. Retrieval of context
    try:
        # For summarizing, we usually want more context k, and maybe disabling HyDE to get raw facts
        ar = advanced_retrieve(session_id, question or "", k=k, use_hyde=False, use_compress=True, use_reorder=True)
        ctx = ar["context_joined"]
    except Exception as e:
        logger.error(f"Context retrieval for summary failed: {e}")
        return {"mode": mode, "output": "Lỗi khi lấy dữ liệu tóm tắt.", "time_ms": 0}

    # 2. Prompt Preparation
    prompt_path = os.path.join("lc", "prompt", "sum_v1.txt")
    if not os.path.exists(prompt_path):
        logger.error(f"Summary prompt file not found: {prompt_path}")
        return {"mode": mode, "output": "[System Error] Summary template missing.", "time_ms": 0}

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            sum_tmpl = f.read()
        
        # Determine max tokens for output from settings
        limit_tokens = APPSETTINGS.summary.max_output_tokens if hasattr(APPSETTINGS.summary, "max_output_tokens") else 512
        
        full_prompt = sum_tmpl.format(
            mode=mode,
            max_tokens=limit_tokens,
            question=(question or "N/A"),
            context=ctx
        )
    except KeyError as e:
        logger.error(f"Prompt formatting error: Missing key {e}")
        return {"mode": mode, "output": "Lỗi định dạng cấu trúc tóm tắt.", "time_ms": 0}

    # 3. LLM call
    t1 = time.perf_counter()
    summary_out = _call_llm(full_prompt, max_tokens=limit_tokens)
    t_total = (time.perf_counter() - t0) * 1000

    return {
        "mode": mode,
        "output": summary_out,
        "time_ms": round(t_total, 2),
        "source_count": len(ar.get("docs", []))
    }
