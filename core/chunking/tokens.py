from functools import lru_cache

@lru_cache(maxsize=128)
def _enc():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

def count_tokens(text: str) -> int:
    if _enc() is None:
        return max(1, len(text)/4)
    return len(_enc().encode(text))
    
