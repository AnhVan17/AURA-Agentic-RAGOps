from __future__ import annotations
from typing import Dict, Any, List
from app.settings import APPSETTINGS
from lc.retrievers.bm25 import BM25Index
from lc.retrievers.ensemble import ensemble_merge
from lc.retrievers.hyde import hyde_dense_search
from lc.vectordb.qdrant_store import get_client, collection_name, search_dense
from core.embedding.embed_gemini import embed_texts
from lc.chains.compress import compress_block
from lc.chains.reorder import reorder_short_to_long_group_by_heading
from ops.observability import trace_chain


@trace_chain("advanced_retrieval")
def advanced_retrieve(session_id: str, q: str, k: int = 8,
                    use_hyde: bool = True, use_compress: bool = True, use_reorder: bool = True) -> Dict[str, Any]:
    client = get_client()
    collection = collection_name(session_id)
    
    # Dense + BM25
    vecs, _ = embed_texts([q], batch_size=1, task_type="retrieval_query")
    dense_hits = search_dense(client, collection, vecs[0], k)
    dense_pairs = [(str(h.id), float(h.score)) for h in dense_hits]

    bm = BM25Index(session_id)
    bm_ready = bm.load()
    bm_hits = []
    bm_pairs = []
    if bm_ready:
        bm_hits = bm.search(q, k)
        for h in bm_hits:
            pl = bm.get_payload(h.idx)
            bm_pairs.append((pl.get("chunk_id"), h.score))
    
    # Merge 
    merged = ensemble_merge(dense_pairs, bm_pairs,
                            APPSETTINGS.retrieval.ensemble.get("dense", 0.6),
                            APPSETTINGS.retrieval.ensemble.get("bm25", 0.4),
                            k=max(k, 20))

    # HyDE Trigger 
    use_hyde = use_hyde and APPSETTINGS.retrieval.hyde.enable
    hyde_cfg = APPSETTINGS.retrieval.hyde
    min_hits = hyde_cfg.trigger.get("min_hits", 3)
    thres = hyde_cfg.trigger.get("bm25_top1_below", 10.0)
    
    needs_hyde = (len([x for x in merged if x[1] > 0]) < min_hits) or \
                 (bm_ready and (len(bm_hits) == 0 or (bm_hits[0].score < thres)))

    hyde_info = {"used": False}
    if use_hyde and needs_hyde:
        # Use simple dense search with HyDE draft
        # Get language (assume 'vi' if not specified, or detect)
        lang = "vi" # default for now or pass as arg
        hyde_results = hyde_dense_search(
            client, collection, q, 
            k_fetch=max(k, 20), 
            lang=lang,
            draft_len_sentences=hyde_cfg.draft_len_sentences
        )
        
        hyde_pairs = [(pid, sc) for pid, sc, _obj in hyde_results]
        merged = ensemble_merge(hyde_pairs, bm_pairs,
                                APPSETTINGS.retrieval.ensemble.get("dense", 0.6),
                                APPSETTINGS.retrieval.ensemble.get("bm25", 0.4),
                                k=max(k, 20))
        hyde_info = {"used": True}

    # Payload Mapping 
    payload_by_id = {}
    for h in dense_hits:
        payload_by_id[str(h.id)] = {
            "chunk_id": str(h.id),
            "text": h.payload.get("text", ""),
            "heading_path": h.payload.get("heading_path", []),
            "page_idx": h.payload.get("page_idx", None),
            "lang": h.payload.get("lang", "vi")
        }
    if bm_ready:
        for h in bm_hits:
            pl = bm.get_payload(h.idx)
            pid = str(pl.get("chunk_id"))
            if pid not in payload_by_id:
                payload_by_id[pid] = {
                    "chunk_id": pid,
                    "text": pl.get("text", ""),
                    "heading_path": pl.get("heading_path", []),
                    "page_idx": pl.get("page_idx", None),
                    "lang": pl.get("lang", "vi")
                }

    # Build top_docs list
    top_docs = []
    for pid, final_score, parts in merged[:k]:
        p = payload_by_id.get(str(pid))
        if p:
            doc_entry = {**p, "score_final": final_score, "contrib": parts}
            top_docs.append(doc_entry)

    # Compression (Sử dụng RegexContextCompressor - Langchain BaseDocumentCompressor)
    comp_info = {"used": False, "ratio": 0.0}
    context_joined = ""
    comp_cfg = APPSETTINGS.compression
    if use_compress and comp_cfg.enable and top_docs:
        from lc.retrievers.compressor import RegexContextCompressor
        from langchain_core.documents import Document
        from lc.chains.compress import count_tokens

        # Chuyển đổi sang list[Document] theo chuẩn Langchain
        docs = [Document(page_content=d["text"], metadata=d) for d in top_docs]
        
        total_before = sum(count_tokens(d.page_content) for d in docs)
        
        # Nén (Lọc bằng Regex)
        compressor = RegexContextCompressor()
        compressed_docs = compressor.compress_documents(docs, query=q)
        
        total_after = sum(count_tokens(d.page_content) for d in compressed_docs)
        
        if compressed_docs:
            comp_info = {
                "used": True, 
                "ratio": round(1 - (total_after / max(1, total_before)), 2)
            }
            # Thay thế text đã nén vào top_docs để reorder hoạt động chính xác
            for cd in compressed_docs:
                for td in top_docs:
                    if td["chunk_id"] == cd.metadata.get("chunk_id"):
                        td["text"] = cd.page_content
            
            context_joined = "\n\n".join(d.page_content for d in compressed_docs)
        else:
            context_joined = "\n\n".join(d["text"] for d in top_docs)
    else:
        context_joined = "\n\n".join(d["text"] for d in top_docs)

    # Reorder
    reorder_info = {"used": False}
    if use_reorder and APPSETTINGS.reorder.enable:
        top_docs = reorder_short_to_long_group_by_heading(top_docs)
        reorder_info = {"used": True}

    return {
        "hyde": hyde_info,
        "compression": comp_info,
        "reorder": reorder_info,
        "docs": top_docs,
        "context_joined": context_joined
    }
    


