from __future__ import annotations
from typing import List, Tuple, Any, Dict
from dataclasses import dataclass
from pathlib import Path
import json, os
from rank_bm25 import BM25Okapi
from lc.retrievers.tokenizer import simple_vi_en_tokens

ART_ROOT = Path("artifacts/bm25")
ART_ROOT.mkdir(parents=True, exist_ok=True)

@dataclass
class BM25Hit:
    idx : int
    score : float


class BM25Index:
    def __init__(self,session_id : str, k1: float = 1.5, b: float = 0.75):
        self.session_id = session_id
        self.payloads : List[Dict[str,Any]] = []
        self.docs_tokens : List[List[str]] = []
        self.k1 = k1
        self.b = b
        self._bm25: BM25Okapi | None = None
    
    @property
    def art_dir(self) -> Path:
        return ART_ROOT / self.session_id

    def fit(self, text: List[str], payloads: List[Dict[str,Any]]):
        self.payloads = payloads
        self.docs_tokens = [simple_vi_en_tokens(doc) for doc in text]
        self._bm25 = BM25Okapi(self.docs_tokens, k1=self.k1, b=self.b)

    def save(self):
        d = self.art_dir
        d.mkdir(parents=True, exist_ok=True)
        with open(d/"docs.jsonl", "w", encoding="utf-8") as f:
            for i, (tok, pl) in enumerate(zip(self.docs_tokens, self.payloads)):
                f.write(json.dumps({"tokens": tok, "payload": pl}, ensure_ascii=False)+"\n")

    def load(self) -> bool:
        d = self.art_dir/"docs.jsonl"
        if not d.exists(): return False
        docs = []
        payloads = []
        with open(d, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                docs.append(obj["tokens"]); payloads.append(obj["payload"])
        if not docs: return False
        self.docs_tokens = docs; 
        self.payloads = payloads
        self._bm25 = BM25Okapi(self.docs_tokens, k1=self.k1, b=self.b)
        return True

    def search(self, query: str, k: int = 10) -> List[BM25Hit]:
        assert self._bm25 is not None, "BM25 not built"
        q_tokens = simple_vi_en_tokens(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        hits = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [BM25Hit(idx=hit[0], score=hit[1]) for hit in hits[:k]]

    def get_payload(self, idx: int) -> Dict[str, Any]:
        return self.payloads[idx]
