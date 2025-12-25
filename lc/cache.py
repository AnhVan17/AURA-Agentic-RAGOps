from functools import lru_cache

@lru_cache(maxsize=256)
def cached_advanced_retrieve(session_id: str, q: str, k: int, flags: str):
    from lc.chains.context_build import advanced_retrieve
    return advanced_retrieve(session_id, q, k=k,
                             use_hyde="hyde" in flags,
                             use_compress="compress" in flags,
                             use_reorder="reorder" in flags)
