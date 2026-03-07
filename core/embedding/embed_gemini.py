from __future__ import annotations
from typing import List, Tuple
import time, math, random
from core.telemetry.telemetry import timeit_stage
from app.settings import APPSETTINGS

EMBEDDING_MODEL = "gemini-embedding-001"

# === Khởi tạo Embedding Client (google-genai mới) ===
_genai_client = None
_client_ok = False

try:
    from google import genai
    api_key = APPSETTINGS.google_api_key
    if api_key and api_key.strip():
        _genai_client = genai.Client(api_key=api_key)
        _client_ok = True
    else:
        print("Warning: Google API Key is missing or empty. Embedding will use Mock mode.")
except Exception as e:
    print(f"Warning: Failed to initialize google.genai Client: {e}")


def _embed_batch(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    if not _client_ok or _genai_client is None:
        print("  Warning: Mock Embedding (API Key not configured)")
        return [[(hash(t) % 1000) / 1000.0 for _ in range(768)] for t in texts]

    try:
        result = _genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
        )
        # result.embeddings is a list of ContentEmbedding objects
        vectors = [emb.values for emb in result.embeddings]
        return vectors
    except Exception as e:
        print(f"Lỗi khi gọi Embedding API: {str(e)}")
        return []


@timeit_stage("embed_chunks")
def embed_texts(
    texts: List[str],
    batch_size: int = 64,
    task_type: str = "retrieval_document"
) -> Tuple[List[List[float]], int]:
    """
    Tạo embedding cho danh sách văn bản lớn.
    Args:
        task_type: 'retrieval_document' (lưu DB) hoặc 'retrieval_query' (search).
    Returns:
        (vectors, dimension)
    """
    vecs: List[List[float]] = []

    bs = max(1, min(batch_size, 100))

    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        for attempt in range(4):
            try:
                v = _embed_batch(batch, task_type=task_type)

                if len(v) != len(batch):
                    raise ValueError(f"Mismatch: Sent {len(batch)}, got {len(v)}")

                vecs.extend(v)
                break
            except Exception as e:
                print(f" Batch {i // bs} failed (Attempt {attempt + 1}): {e}")
                if attempt < 3:
                    time.sleep(1 * (2 ** attempt) + random.random())
                else:
                    current_dim = len(vecs[0]) if vecs else 768
                    print(f" Filling zeros for batch starting at {i}")
                    vecs.extend([[0.0] * current_dim for _ in batch])

    final_dim = len(vecs[0]) if vecs else 0
    return vecs, final_dim