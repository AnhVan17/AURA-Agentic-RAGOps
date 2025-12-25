import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def should_abstain_for_qa(num_docs: int, ctx_tokens: int, cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Determines if the system should abstain from answering based on retrieved context quality.
    
    Returns:
        (True, reason) if it should abstain.
        (False, "") if it's safe to proceed.
    """
    min_docs = cfg.get("min_docs", 2)
    min_tokens = cfg.get("min_ctx_tokens", 150)

    if num_docs < min_docs:
        reason = f"Insufficient documents retrieved ({num_docs} < {min_docs})."
        logger.warning(f"[Guardrail] Abstaining: {reason}")
        return True, reason

    if ctx_tokens < min_tokens:
        reason = f"Context too short ({ctx_tokens} tokens < {min_tokens})."
        logger.warning(f"[Guardrail] Abstaining: {reason}")
        return True, reason

    return False, ""
