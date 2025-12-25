from __future__ import annotations
from typing import List, Tuple
import time, math, random
from core.telemetry.telemetry import timeit_stage
from app.settings import APPSETTINGS
import signal

EMBEDDING_MODEL = "models/text-embedding-004"

class TimeoutLLM(Exception): pass
def _alarm_handler(signum, frame): raise TimeoutLLM()

def with_timeout(sec):
    def deco(fn):
        def wrap(*a, **k):
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(sec)
            try:
                return fn(*a, **k)
            finally:
                signal.alarm(0)
        return wrap
    return deco
    
try: 
    import google.generativeai as genai
    if APPSETTINGS.google_api_key and APPSETTINGS.google_api_key.strip():
        genai.configure(api_key=APPSETTINGS.google_api_key)
        _client_ok = True
    else:
        _client_ok = False
        print("Warning: Google API Key is missing or empty. Embedding will use Mock mode.")
except Exception as e:
    _client_ok = False
    print(f"Warning: Failed to initialize Google GenAI: {e}")

def _embed_batch(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    if not _client_ok:
        print(" Warning: Sử dụng Mock Embedding (do chưa config API Key)")
        return [[(hash(t) % 1000)/1000.0 for _ in range(768)] for t in texts]

    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=texts,
            task_type=task_type,
            title="Embedded Chunk" if task_type == "retrieval_document" else None
        )
        
        if 'embedding' in result:
            return result['embedding']
        else:
            raise ValueError("API không trả về key 'embedding'")

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
        batch = texts[i:i+bs]
        for attempt in range(4):
            try:
                v = _embed_batch(batch, task_type=task_type)
                
                if len(v) != len(batch):
                    raise ValueError(f"Mismatch: Sent {len(batch)}, got {len(v)}")
                
                vecs.extend(v)
                break 
            except Exception as e:
                print(f" Batch {i//bs} failed (Attempt {attempt+1}): {e}")
                if attempt < 3:
                    time.sleep(1 * (2 ** attempt) + random.random())
                else:
                    current_dim = len(vecs[0]) if vecs else 768
                    print(f" Filling zeros for batch starting at {i}")
                    vecs.extend([[0.0]*current_dim for _ in batch])
    
    final_dim = len(vecs[0]) if vecs else 0
    return vecs, final_dim