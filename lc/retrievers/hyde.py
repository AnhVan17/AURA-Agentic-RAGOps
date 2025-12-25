from __future__ import annotations
from typing import List, Tuple, Dict, Any
from app.settings import APPSETTINGS
from core.embedding.embed_gemini import embed_texts
from lc.vectordb.qdrant_store import search_dense
from core.telemetry.telemetry import timeit_stage
import google.generativeai as genai

def gen_hyde_draft_sync(query: str, lang: str = "vi", n_sent: int = 2) -> str:
    """Synchronous version of gen_hyde_draft."""
    if not APPSETTINGS.google_api_key:
        return query
    try:
        genai.configure(api_key=APPSETTINGS.google_api_key)
        model = genai.GenerativeModel(APPSETTINGS.app.default_llm)
        
        prompt = (
            f"Write a short academic passage (exactly {n_sent} sentences) "
            f"that answers or explains: '{query}'.\n\n"
            f"Requirements:\n"
            f"- Use academic/technical language and terminology\n"
            f"- Write as if citing from a research paper\n"
            f"- DO NOT write 'I don't know'\n"
            f"- Only write direct content"
        )
        if lang == "vi":
            prompt = (
                f"Viết một đoạn văn bản học thuật ngắn (chính xác {n_sent} câu) "
                f"giải thích hoặc trả lời câu hỏi: '{query}'.\n\n"
                f"Yêu cầu:\n"
                f"- Sử dụng văn phong chuyên ngành, từ khóa kỹ thuật\n"
                f"- Viết như đang trích dẫn từ một paper/tài liệu\n"
                f"- KHÔNG viết 'tôi không biết'\n"
                f"- Chỉ viết nội dung trả lời trực tiếp"
            )
            
        resp = model.generate_content(prompt)
        if resp and resp.text:
            return resp.text.strip()
        return query
    except Exception as e:
        print(f"HyDE Generation Error: {e}") 
        return query

def hyde_dense_search(client, collection: str, query: str, k_fetch: int = 20, lang: str = "en", draft_len_sentences: int = 2) -> List[Tuple[str, float, Dict]]:
    """
    Performs a dense search using a HyDE generated draft.
    Returns: List of (chunk_id, score, payload)
    """
    draft = gen_hyde_draft_sync(query, lang=lang, n_sent=draft_len_sentences)
    vecs, _ = embed_texts([draft], batch_size=1, task_type="retrieval_query")
    if not vecs:
        return []
    
    hits = search_dense(client, collection, vecs[0], limit=k_fetch)
    results = []
    for h in hits:
        cid = str(h.payload.get("chunk_id", h.id))
        results.append((cid, float(h.score), h.payload))
    return results

async def gen_hyde_draft(query: str, lang: str = "vi", n_sent: int = 2) -> str:
# ... (rest of the file remains same)
    """
    Tạo văn bản giả định (hypothetical document) từ query.
    
    Args:
        query: Câu hỏi gốc
        lang: Ngôn ngữ ('vi' hoặc 'en')
        n_sent: Số câu trong văn bản giả định
    
    Returns:
        Văn bản giả định hoặc query gốc nếu có lỗi
    """
    if not APPSETTINGS.google_api_key:
        return query
    
    try:
        genai.configure(api_key=APPSETTINGS.google_api_key)
        
        model = genai.GenerativeModel(
            model_name=APPSETTINGS.app.default_llm, 
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=200,  
                candidate_count=1
            )
        )
    
        if lang == "vi":
            prompt = (
                f"Viết một đoạn văn bản học thuật ngắn (chính xác {n_sent} câu) "
                f"giải thích hoặc trả lời câu hỏi: '{query}'.\n\n"
                f"Yêu cầu:\n"
                f"- Sử dụng văn phong chuyên ngành, từ khóa kỹ thuật\n"
                f"- Viết như đang trích dẫn từ một paper/tài liệu\n"
                f"- KHÔNG viết 'tôi không biết' hay giải thích cách trả lời\n"
                f"- Chỉ viết nội dung trả lời trực tiếp"
            )
        else:
            prompt = (
                f"Write a short academic passage (exactly {n_sent} sentences) "
                f"that answers or explains: '{query}'.\n\n"
                f"Requirements:\n"
                f"- Use academic/technical language and terminology\n"
                f"- Write as if citing from a research paper\n"
                f"- DO NOT write 'I don't know' or explain how to answer\n"
                f"- Only write direct content"
            )
        
        resp = await model.generate_content_async(prompt)
        
        if resp and resp.text:
            return resp.text.strip()
        return query

    except Exception as e:
        print(f"HyDE Generation Error: {e}") 
        return query


def should_apply_hyde(
    bm25_results: List[Tuple[str, float]], 
    dense_results: List[Any],
    query: str
) -> bool:
    """
    Quyết định có nên áp dụng HyDE không dựa trên chất lượng kết quả ban đầu.
    
    Args:
        bm25_results: Kết quả BM25 [(id, score), ...]
        dense_results: Kết quả dense vector search
        query: Câu hỏi gốc
    
    Returns:
        True nếu nên dùng HyDE
    """
    cfg = APPSETTINGS.retrieval.hyde
    
    # Nếu disable trong config
    if not cfg.get("enable", False):
        return False
    
    # Điều kiện 1: Quá ít kết quả
    if len(bm25_results) < cfg.trigger.get("min_hits", 2):
        return True
    
    # Điều kiện 2: Top BM25 score quá thấp
    if bm25_results and bm25_results[0][1] < cfg.trigger.get("bm25_top1_below", 0.30):
        return True
    
    # Điều kiện 3: Query quá ngắn (< 3 từ) → có thể mơ hồ
    if len(query.split()) < 3:
        return True
    
    return False


async def hybrid_search_with_hyde(
    client,
    collection: str,
    query: str,
    lang: str,
    bm25_index,
    k: int = 8
) -> Tuple[List, bool]:
    """
    Tìm kiếm hybrid với HyDE fallback thông minh.
    
    Returns:
        (results, hyde_applied)
    """
    from lc.retrievers.ensemble import ensemble_merge
    
    vecs, dim = embed_texts([query], batch_size=1)
    qvec = vecs[0]
    
    dense_hits = search_dense(client, collection, qvec, limit=k)
    dense_pairs = [(str(h.payload.get("chunk_id", h.id)), float(h.score)) for h in dense_hits]
    
    bm25_hits = bm25_index.search(query, k=k)
    bm25_pairs = []
    for hit in bm25_hits:
        pl = bm25_index.get_payload(hit.idx)
        bm25_pairs.append((pl.get("chunk_id"), hit.score))
    
    hyde_applied = False
    if should_apply_hyde(bm25_pairs, dense_hits, query):
        print(f"🔄 Applying HyDE for query: {query[:50]}...")
        
        # Generate HyDE draft
        cfg = APPSETTINGS.retrieval.hyde
        n_sent = cfg.get("draft_len_sentences", 2)
        hyde_text = await gen_hyde_draft(query, lang=lang, n_sent=n_sent)
        
        # Re-search với HyDE text
        vecs_hyde, _ = embed_texts([hyde_text], batch_size=1)
        qvec_hyde = vecs_hyde[0]
        
        dense_hits = search_dense(client, collection, qvec_hyde, limit=k)
        dense_pairs = [(str(h.payload.get("chunk_id", h.id)), float(h.score)) for h in dense_hits]
        
        hyde_applied = True
    
    # 3. Merge results
    w_dense = APPSETTINGS.retrieval.ensemble.get("dense", 0.6)
    w_bm25 = APPSETTINGS.retrieval.ensemble.get("bm25", 0.4)
    merged = ensemble_merge(dense_pairs, bm25_pairs, w_dense, w_bm25, k=k)
    
    # 4. Build final results
    payload_by_id = {}
    for h in dense_hits:
        cid = str(h.payload.get("chunk_id", h.id))
        payload_by_id[cid] = {
            "text": h.payload.get("text", ""),
            "heading_path": h.payload.get("heading_path", []),
            "page_idx": h.payload.get("page_idx", None),
            "lang": h.payload.get("lang", "vi")
        }
    for hit in bm25_hits:
        pl = bm25_index.get_payload(hit.idx)
        payload_by_id[pl["chunk_id"]] = {
            "text": pl["text"],
            "heading_path": pl.get("heading_path", []),
            "page_idx": pl.get("page_idx", None),
            "lang": pl.get("lang", "vi")
        }
    
    results = []
    for pid, final, parts in merged:
        payload = payload_by_id.get(pid, {})
        results.append({
            "id": pid, 
            "score": final, 
            "contrib": parts,
            **payload
        })
    
    return results, hyde_applied
