from __future__ import annotations
from typing import List, Dict
import logging
import re
import google.generativeai as genai

from app.settings import APPSETTINGS
from core.embedding.embed_gemini import embed_texts

logger = logging.getLogger(__name__)


PROMPT_EXTRACT = """You are an academic content extraction assistant.
TASK: Extract ONLY the essential academic content from the given context.
KEEP (must include):
✓ Definitions, theorems, propositions, laws, axioms
✓ Mathematical formulas and equations
✓ Numbered/labeled statements (e.g., "Definition 1.1", "Theorem 2.3", "Equation (5)")
✓ Specific data, conditions, constraints, assumptions
✓ Section/chapter headings directly relevant to key concepts
REMOVE (must exclude):
✗ Long illustrative examples
✗ Verbose explanations or redundant descriptions
✗ General commentary without technical content
OUTPUT FORMAT:
- Preserve original sentences EXACTLY (no paraphrasing)
- One statement per line
- Maintain all mathematical symbols and numbering
Context to extract:
{ctx}
Extracted content:
"""

def _llm_extract(ctx: str, max_out_tokens: int) -> str:
    """
    Extract academic content from context using Gemini.
    
    Args:
        ctx: Context to extract from
        max_out_tokens: Maximum number of output tokens
        
    Returns:
        Extracted content, or empty string if failed
    """
    try:
        if not APPSETTINGS.google_api_key:
            logger.warning("Google API Key not configured")
            return ""
        
        genai.configure(api_key=APPSETTINGS.google_api_key)
        model = genai.GenerativeModel(APPSETTINGS.app.default_llm)
        
        resp = model.generate_content(
            PROMPT_EXTRACT.format(ctx=ctx),
            generation_config={
                "max_output_tokens": max_out_tokens,
                "temperature": 0.1,  # Reduce creativity for consistent output
            },
            safety_settings={
                "HARASSMENT": "block_none",
                "HATE_SPEECH": "block_none", 
                "SEXUALLY_EXPLICIT": "block_none",
                "DANGEROUS_CONTENT": "block_none",
            }
        )
        
        if not resp.candidates:
            logger.warning("Gemini returned no candidates (possibly blocked)")
            return ""
            
        return (resp.text or "").strip()
        
    except Exception as e:
        logger.error(f"Error calling Gemini extract: {e}")
        return ""
    
def _heuristic_extract_optimized(ctx: str) -> str:
    """
    Trích xuất dựa trên luật (Rule-based) tối ưu cho văn bản học thuật/kỹ thuật.
    Cải tiến: Dùng Regex mạnh hơn, thêm ngữ cảnh (buffer lines).
    """
    if not ctx: return ""
    
    lines = [x.strip() for x in ctx.split("\n") if x.strip()]
    keep_indices = set()

    p_math = re.compile(r"([=≈≠]\s*[\d\w]|[\$\\]|(\b[A-Z]\([a-z0-9,]+\)))")  
    p_def = re.compile(r"(?:là|gọi là|định nghĩa|khái niệm)\s+(?:một|các|những|tập hợp|hệ thống)", re.IGNORECASE)
    p_head = re.compile(r"^(Chương|Mục|Điều|Khoản|Section|Chapter|Part)\b", re.IGNORECASE)

    for i, ln in enumerate(lines):
        is_relevant = False
        
        # Check Heading
        if p_head.match(ln):
            is_relevant = True
        
        # Check Math/Fomula
        elif p_math.search(ln):
            is_relevant = True
            
        # Check Definition
        elif p_def.search(ln):
            is_relevant = True
            
        # Check keywords 
        elif any(k in ln for k in ["công thức", "định lý", "theorem", "lemma", "equation"]):
            is_relevant = True

        if is_relevant:
            keep_indices.add(i)
            if i > 0:
                keep_indices.add(i - 1)

    final_lines = [lines[i] for i in sorted(list(keep_indices))]
    if len(final_lines) < len(lines) * 0.1: 
        fallback = lines[:2] + lines[-1:] if len(lines) > 3 else lines
        return "\n".join(fallback)

    return "\n".join(final_lines)

def count_tokens(text: str, model_name: str | None = None) -> int:
    """
    Count tokens using Gemini's tokenizer.
    """
    if model_name is None:
        model_name = APPSETTINGS.app.default_llm
    try:
        if not APPSETTINGS.google_api_key or not text:
            return len(text) // 4
            
        genai.configure(api_key=APPSETTINGS.google_api_key)
        model = genai.GenerativeModel(model_name)
        return model.count_tokens(text).total_tokens
    except Exception as e:
        logger.warning(f"Failed to count tokens (model={model_name}), using estimation: {e}")
        # Fallback: estimate (1 token ≈ 4 chars for English, 3 for Vietnamese)
        return len(text) // 4

def compress_block(ctx: str, min_ratio: float = 0.30, max_out_tokens: int = 400) -> Dict:
    before_tokens = count_tokens(ctx)  
    
    out = _llm_extract(ctx, max_out_tokens=max_out_tokens)
    
    method = "llm"
    if not out or len(out) < 5:
        logger.info("LLM extraction failed or returned too little content, falling back to heuristic.")
        out = _heuristic_extract_optimized(ctx)
        method = "heuristic"
    
    after_tokens = count_tokens(out) 
    reduction_ratio = 1 - (after_tokens / max(1, before_tokens))
    
    return {
        "original_tokens": before_tokens,  
        "compressed_tokens": after_tokens, 
        "reduction_ratio": round(reduction_ratio, 2), 
        "method": method, 
        "compressed": out
    }