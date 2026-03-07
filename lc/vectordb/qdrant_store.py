"""
Qdrant Vector Store — Hybrid Search (Dense + Sparse)
=====================================================
Sử dụng Named Vectors của Qdrant:
  - "dense": BGE-M3 dense vectors (1024-dim, Cosine)
  - "sparse": BGE-M3 learned sparse vectors (token weights)

Hybrid Search = prefetch dense + sparse → RRF fusion
"""
from __future__ import annotations
from typing import List, Dict, Any
from app.settings import APPSETTINGS
from qdrant_client import QdrantClient
from qdrant_client.http import models
from core.embedding.embed_bge import DENSE_DIM
import uuid
import logging

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Singleton: chỉ tạo 1 QdrantClient duy nhất."""
    global _client
    if _client is None:
        if APPSETTINGS.qdrant.url == "local":
            _client = QdrantClient(path="qdrant_local_db")
        else:
            _client = QdrantClient(
                url=APPSETTINGS.qdrant.url,
                api_key=APPSETTINGS.qdrant.api_key,
                prefer_grpc=APPSETTINGS.qdrant.prefer_grpc,
            )
    return _client


def collection_name(session_id: str) -> str:
    return f"{APPSETTINGS.qdrant_collection_prefix}{session_id}"


def ensure_collection(client: QdrantClient, coll: str, vector_size: int = DENSE_DIM):
    """Tạo collection với Named Vectors: dense (VectorParams) + sparse (SparseVectorParams)."""
    if client.collection_exists(collection_name=coll):
        logger.info(f"Collection '{coll}' already exists.")
        return

    logger.info(f"Creating collection '{coll}' with hybrid vectors...")
    client.create_collection(
        collection_name=coll,
        vectors_config={
            "dense": models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,  # IDF weighting cho sparse
            ),
        },
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=128,
        ),
    )
    logger.info(f"Collection '{coll}' created with dense({vector_size}) + sparse vectors.")


def _is_valid_uuid(val: Any) -> bool:
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False


def upsert_points(
    client: QdrantClient,
    coll: str,
    dense_vectors: List[List[float]],
    sparse_vectors: List[Dict[int, float]],
    payloads: List[Dict[str, Any]],
    ids: List[str],
):
    """Upsert points với cả dense và sparse vectors."""
    points = []
    for i, doc_id in enumerate(ids):
        # Generate Qdrant-compatible ID
        if isinstance(doc_id, int) or _is_valid_uuid(doc_id):
            qid = doc_id
        else:
            qid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(doc_id)))

        # Convert sparse dict {token_id: weight} → SparseVector
        sp = sparse_vectors[i]
        sparse_indices = list(sp.keys())
        sparse_values = list(sp.values())

        points.append(models.PointStruct(
            id=qid,
            vector={
                "dense": dense_vectors[i],
                "sparse": models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
            },
            payload=payloads[i],
        ))

    # Upsert in batches of 64
    batch_size = 64
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        client.upsert(collection_name=coll, points=batch)
    logger.info(f"Upserted {len(points)} points to '{coll}'")


def search_hybrid(
    client: QdrantClient,
    coll: str,
    dense_vector: List[float],
    sparse_vector: Dict[int, float],
    limit: int = 10,
) -> list:
    """
    Qdrant Native Hybrid Search:
      1. Prefetch top-K from Dense
      2. Prefetch top-K from Sparse
      3. RRF (Reciprocal Rank Fusion) để merge kết quả
    """
    sparse_indices = list(sparse_vector.keys())
    sparse_values = list(sparse_vector.values())

    results = client.query_points(
        collection_name=coll,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=limit * 2,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
                using="sparse",
                limit=limit * 2,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    ).points

    return results


def search_dense(
    client: QdrantClient,
    coll: str,
    query_vector: List[float],
    limit: int = 10,
) -> list:
    """Dense-only search (backward compatible)."""
    return client.query_points(
        collection_name=coll,
        query=query_vector,
        using="dense",
        limit=limit,
        with_payload=True,
    ).points
