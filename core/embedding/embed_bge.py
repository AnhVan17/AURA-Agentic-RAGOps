"""
BGE-M3 Embedding Module
========================
Sử dụng BAAI/bge-m3 qua thư viện FlagEmbedding chính chủ.
Tạo cả Dense và Sparse vectors học hỏi (learned lexical weights) trong 1 lần gọi.
- Dense: 1024-dim vectors (semantic search)
- Sparse: Learned token weights (keyword-aware search, tốt hơn BM25 rất nhiều)

Qdrant sẽ lưu cả 2 loại vectors và thực hiện Hybrid Search (RRF) natively.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import logging
import time

logger = logging.getLogger(__name__)

# === Singleton Model ===
_model = None
_model_loading = False

DENSE_DIM = 1024  # BGE-M3 dense dimension


def get_model():
    """Lazy load BGE-M3 model (singleton, chỉ tải 1 lần)."""
    global _model, _model_loading
    if _model is not None:
        return _model
    if _model_loading:
        # Tránh load song song
        while _model_loading:
            time.sleep(0.5)
        return _model

    _model_loading = True
    try:
        logger.info("Loading BGE-M3 model via FlagEmbedding (may take 30-60s)...")
        from FlagEmbedding import BGEM3FlagModel
        _model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=False,  # CPU mode (set True nếu có GPU)
        )
        logger.info("BGE-M3 model loaded successfully!")
        return _model
    except Exception as e:
        logger.error(f"Failed to load BGE-M3: {e}")
        _model_loading = False
        raise
    finally:
        _model_loading = False


def embed_dense_sparse(
    texts: List[str],
    batch_size: int = 4, # Smaller batch size to prevent OOM
) -> Dict[str, Any]:
    """
    Tạo cả Dense và Sparse embeddings cho danh sách văn bản với BGE-M3.

    Returns:
        {
            "dense": List[List[float]],           # N x 1024
            "sparse": List[Dict[int, float]],     # N x {token_id: weight}
        }
    """
    model = get_model()

    all_dense = []
    all_sparse = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"  Embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} ({len(batch)} texts)")

        output = model.encode(
            batch,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=batch_size
        )

        # Dense vectors: numpy array -> list
        for vec in output["dense_vecs"]:
            all_dense.append(vec.tolist())

        # Sparse vectors: list of dicts {token_id: weight}
        for weights in output["lexical_weights"]:
            # Optionally filter out very low weights to save space (e.g. < 0.1)
            filtered_weights = {int(k): round(float(v), 4) for k, v in weights.items() if float(v) > 0.1}
            all_sparse.append(filtered_weights)

    return {
        "dense": all_dense,
        "sparse": all_sparse,
    }


def embed_query(query: str) -> Dict[str, Any]:
    """
    Embed 1 câu hỏi (cho search).

    Returns:
        {
            "dense": List[float],           # 1024-dim
            "sparse": Dict[int, float],     # {token_id: weight}
        }
    """
    result = embed_dense_sparse([query], batch_size=1)
    return {
        "dense": result["dense"][0],
        "sparse": result["sparse"][0],
    }
