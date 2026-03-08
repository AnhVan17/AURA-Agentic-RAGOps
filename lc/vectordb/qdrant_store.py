from __future__ import annotations
from typing import List, Tuple, Dict, Any
from app.settings import APPSETTINGS
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid

_client: QdrantClient | None = None

def get_client() -> QdrantClient:
    """Singleton: chỉ tạo 1 QdrantClient duy nhất, tránh file-lock conflict ở chế độ local."""
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

def ensure_collection(client: QdrantClient, coll: str, vector_size: int):
    # 1. Check if the Collection already exists
    if client.collection_exists(collection_name=coll):
        # 2. If it exists, CHECK COMPATIBILITY (Important)
        info = client.get_collection(coll)
        
        # Note: vectors_config can be a dict or an object depending on the version.
        # The code below handles the most common case (single vector param).
        try:
            existing_size = info.config.params.vectors.size
        except AttributeError:
            # Handle named vectors case (advanced), accessing 'default' if applicable
            existing_size = info.config.params.vectors['default'].size

        if existing_size != vector_size:
            raise ValueError(
                f"❌ Vector Size Mismatch! Collection '{coll}' has size={existing_size}, "
                f"but the current model requires size={vector_size}. "
                "Please rename the collection or delete the existing one."
            )
        
        print(f"✅ Collection '{coll}' already exists and matches the configuration.")
        return

    # 3. If not exists -> Create new
    print(f"⚠️ Collection '{coll}' not found. Creating a new one...")
    try:
        client.create_collection(
            collection_name=coll,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            ),
            # Optimize HNSW for speed/RAM balance
            hnsw_config=models.HnswConfigDiff(
                m=16,           
                ef_construct=100 
            )
        )
        print(f"🎉 Collection '{coll}' created successfully.")
    except Exception as e:
        print(f"❌ Failed to create Collection: {e}")
        raise e

def _is_valid_uuid(val: Any) -> bool:
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False

def upsert_points(client: QdrantClient, coll : str, vectors: List[List[float]], payloads: List[Dict[str,Any]], ids: List[str]):
    qdrant_ids = []
    for doc_id in ids:
        # Nếu là số nguyên hoặc UUID hợp chuẩn thì giữ nguyên
        if isinstance(doc_id, int) or _is_valid_uuid(doc_id):
            qdrant_ids.append(doc_id)
        else:
            # Nếu là chuỗi bất kỳ, băm thành UUID chuẩn
            qdrant_ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, str(doc_id))))

    client.upsert(
        collection_name =coll,
        points = models.Batch(ids = qdrant_ids, vectors = vectors, payloads = payloads)
    )

def search_dense (client: QdrantClient, coll : str, query_vector: List[float], limit: int = 10):
    return client.query_points(
        collection_name = coll,
        query = query_vector,
        limit = limit,
        with_payload = True,
        search_params=models.SearchParams(hnsw_ef=APPSETTINGS.qdrant.hnsw.get("ef_search", 128))
    ).points
