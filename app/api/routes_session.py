from app.settings import APPSETTINGS
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, Request
from core.parsing.parsing import parse_file, parse_pdf_ocr as parse_file_with_ocr_if_needed
from core.ocr import ocr_pdf
from core.chunking.chunk import chunk_heading_aware, detect_lang_fast
import shutil
import tempfile
import time
import logging
from typing import Any, Dict
from pathlib import Path
from core.embedding.embed_gemini import embed_texts
from lc.vectordb.qdrant_store import get_client, collection_name, ensure_collection, upsert_points, search_dense
from lc.retrievers.bm25 import BM25Index
from lc.retrievers.ensemble import ensemble_merge
from lc.retrievers.tokenizer import simple_vi_en_tokens
from lc.chains.context_build import advanced_retrieve

logger = logging.getLogger(__name__)


router = APIRouter()


@router.post("/session/preview")
async def preview_file(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    parsed_content = parse_file(tmp_path)
    snippet = parsed_content.text_norm[:2000]  
    return {
        "filename": file.filename,
        "note": parsed_content.note,
        "stats": {
            "n_char_raw": parsed_content.n_char_raw,
            "n_char_normalized": parsed_content.n_char_normalized,
            "parse_ms": parsed_content.parse_ms,
            "norm_ms": parsed_content.norm_ms,
        },
        "preview": snippet
    }

@router.post("/session/preview_ocr")
async def preview_ocr(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        from core.parsing.parsing import detect_file_type, is_low_text_pdf
        ftype = detect_file_type(tmp_path)
        if ftype != "pdf":
            return {"error":"not_pdf","note":"Chỉ OCR cho PDF."}
    
        if not is_low_text_pdf(tmp_path):
            return {"status":"skip","note":"PDF có text, không cần OCR."}
            
        ocr_res = ocr_pdf(tmp_path)
        return {
            "status":"ok",
            "pages_ocr": len(ocr_res.pages),
            "render_ms": ocr_res.total_render_ms,
            "ocr_ms": ocr_res.total_ocr_ms,
            "preview": (ocr_res.text_joined[:2000] if ocr_res.text_joined else "")
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

@router.post("/session/preview_chunk")
async def preview_chunk(
    file: UploadFile = File(...),
    target_tokens: int = Form(700),
    overlap_sentences: int = Form(2)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    parsed = parse_file_with_ocr_if_needed(tmp_path)  
    lang = detect_lang_fast(parsed.text_norm)
    chunks = chunk_heading_aware(
        text=parsed.text_norm,
        target_tokens=target_tokens,
        overlap_sentences=overlap_sentences,
        lang_hint=lang,
        page_idx=None,
        id_prefix="C"
    )
    preview = [{
        "id": c.id,
        "n_tokens": c.n_tokens,
        "heading_path": c.heading_path,
        "snippet": (c.text[:300] + ("..." if len(c.text) > 300 else "")),
    } for c in chunks[:3]]

    stats = {
        "total_chunks": len(chunks),
        "avg_tokens": round(sum(c.n_tokens for c in chunks)/max(1,len(chunks)),1),
        "min_tokens": min(c.n_tokens for c in chunks) if chunks else 0,
        "max_tokens": max(c.n_tokens for c in chunks) if chunks else 0,
    }
    return {"lang": lang, "stats": stats, "preview": preview}


@router.post("/session/upload")
async def session_upload(request: Request, session_id: str = Query(..., min_length=1), file: UploadFile = File(...)):
    maxb = APPSETTINGS.api["max_upload_mb"] * 1024 * 1024
    cl = request.headers.get("content-length")
    if cl and int(cl) > maxb:
        raise HTTPException(status_code=413, detail="Upload too large")
    t0 = time.perf_counter()
    
    # Lưu file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    #Pipeline: Loader → Splitter  → Embed → Qdrant  ===
    from ops.loaders import AcademicDocumentLoader
    from ops.splitters import AcademicTextSplitter

    # 1) Load — Tự động phát hiện PDF text/scan, chọn PyMuPDF hoặc OCR
    loader = AcademicDocumentLoader(tmp_path)
    raw_docs = list(loader.lazy_load())
    load_method = raw_docs[0].metadata.get("method", "unknown") if raw_docs else "unknown"

    # 2) Chunk — Heading-aware splitting với metadata đầy đủ
    ch_cfg = APPSETTINGS.ingest.chunk
    splitter = AcademicTextSplitter(
        target_tokens=ch_cfg.get("target_tokens", 700),
        overlap_sentences=ch_cfg.get("overlap_sentences", 2),
    )
    chunks = splitter.split_documents(raw_docs)

    # 3) Embed (batch)
    texts = [c.page_content for c in chunks]
    vecs, dim = embed_texts(texts, batch_size=APPSETTINGS.ingest.batch_size)

    # 4) Qdrant upsert
    client = get_client()
    coll = collection_name(session_id)
    ensure_collection(client, coll, dim)
    payloads = [{
        "chunk_id": c.metadata.get("chunk_id", f"C{i:05d}"),
        "text": c.page_content,
        "heading_path": c.metadata.get("heading_path", []),
        "section": c.metadata.get("section", ""),
        "page_idx": c.metadata.get("page"),
        "lang": c.metadata.get("lang", "vi"),
        "file_name": file.filename,
        "source_path": tmp_path,
        "method": c.metadata.get("method", load_method),
    } for i, c in enumerate(chunks)]
    ids = [f"{session_id}_{p['chunk_id']}" for p in payloads]
    upsert_points(client, coll, vecs, payloads, ids)

    # 5) BM25 Update
    bm = BM25Index(session_id)
    if not bm.load():
        bm.fit(texts, payloads)
        bm.save()
    else:
        bm.docs_tokens += [simple_vi_en_tokens(t) for t in texts]
        bm.payloads += payloads
        bm._bm25 = None
        bm.save()
        bm.load()

    dt = (time.perf_counter() - t0) * 1000
    return {
        "session_id": session_id,
        "collection": coll,
        "filename": file.filename,
        "method": load_method,
        "counts": {"pages": len(raw_docs), "chunks": len(chunks), "vectors": len(vecs)},
        "stage_ms": {"total_ms": round(dt, 1)},
    }


@router.get("/session/search")
def session_search(session_id: str, q: str, k: int = 8):
    """Hybrid Search: Dense (Qdrant) + Sparse (BM25) → Ensemble Merge."""
    if not q or not session_id:
        raise HTTPException(status_code=400, detail="Missing q or session_id")
    
    # 1. Dense Search (Vector)
    vecs, dim = embed_texts([q], batch_size=1)
    qvec = vecs[0]
    client = get_client()
    coll = collection_name(session_id)
    try:
        dense_hits = search_dense(client, coll, qvec, limit=k)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found or search error: {e}")
    dense_pairs = [(str(h.payload.get("chunk_id", h.id)), float(h.score)) for h in dense_hits]

    # 2. Sparse Search (BM25)
    bm = BM25Index(session_id)
    bm25_pairs = []
    bm25_hits = []
    if bm.load():
        bm25_hits = bm.search(q, k=k)
        for hit in bm25_hits:
            pl = bm.get_payload(hit.idx)
            bm25_pairs.append((pl.get("chunk_id"), hit.score))

    # 3. Ensemble Merge (nếu có BM25)
    if bm25_pairs:
        w_dense = APPSETTINGS.retrieval.ensemble.get("dense", 0.6)
        w_bm25 = APPSETTINGS.retrieval.ensemble.get("bm25", 0.4)
        merged = ensemble_merge(dense_pairs, bm25_pairs, w_dense, w_bm25, k=k)

        # Build payload lookup
        payload_by_id = {}
        for h in dense_hits:
            cid = str(h.payload.get("chunk_id", h.id))
            payload_by_id[cid] = h.payload if h.payload else {}
        for hit in bm25_hits:
            pl = bm.get_payload(hit.idx)
            payload_by_id[pl.get("chunk_id", "")] = pl

        out = []
        for pid, final, parts in merged:
            payload = payload_by_id.get(pid, {})
            out.append({
                "id": pid,
                "score": final,
                "contrib": parts,
                "text": payload.get("text", ""),
                "heading_path": payload.get("heading_path", []),
                "section": payload.get("section", ""),
                "page_idx": payload.get("page_idx", None),
                "lang": payload.get("lang", "vi"),
                "file_name": payload.get("file_name", ""),
            })
    else:
        # Fallback: Dense-only (nếu chưa có BM25 index)
        out = []
        for h in dense_hits:
            out.append({
                "id": h.payload.get("chunk_id", h.id) if h.payload else h.id,
                "score": h.score,
                "text": h.payload.get("text","") if h.payload else "",
                "heading_path": h.payload.get("heading_path", []) if h.payload else [],
                "section": h.payload.get("section", "") if h.payload else "",
                "page_idx": h.payload.get("page_idx", None) if h.payload else None,
                "lang": h.payload.get("lang", "vi") if h.payload else "vi",
                "file_name": h.payload.get("file_name", "") if h.payload else "",
            })
    return {"k": k, "mode": "hybrid" if bm25_pairs else "dense_only", "results": out}


@router.get("/session/search_hybrid")
def session_search_hybrid(session_id: str, q: str, k: int = 8):
    #Dense
    vecs, dim = embed_texts([q], batch_size=1)
    qvec = vecs[0]
    client = get_client()
    coll = collection_name(session_id)
    dense_hits = search_dense(client, coll, qvec, limit=k)
    dense_pairs = [(str(h.payload.get("chunk_id", h.id)), float(h.score)) for h in dense_hits]
    
    #BM25
    bm = BM25Index(session_id)
    if not bm.load():
        raise HTTPException(status_code=404, detail="BM25 index not found")
    bm25_hits = bm.search(q, k=k)
    bm25_pairs  = []
    for hit in bm25_hits:
        pl = bm.get_payload(hit.idx)
        pid = pl.get("chunk_id")
        bm25_pairs.append((pid, hit.score))

    # merge
    w_dense = APPSETTINGS.retrieval.ensemble.get("dense", 0.6)
    w_bm25 = APPSETTINGS.retrieval.ensemble.get("bm25", 0.4)
    merged = ensemble_merge(dense_pairs, bm25_pairs, w_dense, w_bm25, k=k)

    payload_by_id = {}
    # dense payload
    for h in dense_hits:
        cid = str(h.payload.get("chunk_id", h.id))
        payload_by_id[cid] = {
            "text": h.payload.get("text",""),
            "heading_path": h.payload.get("heading_path", []),
            "page_idx": h.payload.get("page_idx", None),
            "lang": h.payload.get("lang", "vi")
        }
    # bm25 payload 
    for hit in bm25_hits:
        pl = bm.get_payload(hit.idx)
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
            "id": pid, "score": final, "contrib": parts,
            **payload
        })
    return {"k": k, "alpha_dense": w_dense, "results": results}



@router.get("/session/search_advanced")
def session_search_advanced(
    session_id: str,
    q: str,
    k: int = 8,
    hyde: bool = True,
    compress: bool = True,
    reorder: bool = True
):
    if not session_id or not q:
        raise HTTPException(status_code=400, detail="Missing session_id or q")
    
    try:
        res = advanced_retrieve(
            session_id, q, k=k, 
            use_hyde=hyde, 
            use_compress=compress, 
            use_reorder=reorder
        )
        preview = []
        for d in res["docs"][:5]: 
            text = d.get("text", "")
            preview.append({
                "chunk_id": d["chunk_id"],
                "heading": " / ".join(d.get("heading_path", [])),
                "score": round(d.get("score_final", 0), 4),
                "snippet": text[:200] + ("..." if len(text) > 200 else "")
            })
            
        return {
            "success": True,
            "flags": {
                "hyde": res["hyde"], 
                "compression": res["compression"], 
                "reorder": res["reorder"]
            },
            "stats": {
                "total_docs": len(res["docs"]),
                "context_chars": len(res["context_joined"]),
                "approx_tokens": len(res["context_joined"]) // 4
            },
            "preview_docs": preview,
            "context_preview": res["context_joined"][:1500] # Trả về đoạn đầu của context đã xử lý
        }
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Advanced search failed: {str(e)}")