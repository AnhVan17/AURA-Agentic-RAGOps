from __future__ import annotations
from typing import Dict, List, Tuple

def _minmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores: return {}
    vals = list(scores.values()); lo=min(vals); hi=max(vals)
    if hi==lo:
        return {k: 0.0 for k in scores} 
    return {k: (v-lo)/(hi-lo) for k,v in scores.items()}



def ensemble_merge(
    dense: List[Tuple[str, float]],   
    bm25:  List[Tuple[str, float]],
    w_dense: float = 0.6,
    w_bm25: float = 0.4,
    k: int = 8
) -> List[Tuple[str, float, Dict[str,float]]]:
    d = {i:s for i,s in dense}
    b = {i:s for i,s in bm25}
    all_ids = set(d) | set(b)
    d_scaled = _minmax(d)
    b_scaled = _minmax(b)
    out = []
    for i in all_ids:
        sd = d_scaled.get(i, 0.0)
        sb = b_scaled.get(i, 0.0)
        final = w_dense*sd + w_bm25*sb
        out.append((i, final, {"dense": sd, "bm25": sb}))
    out.sort(key=lambda z: (z[1], (z[0] in d and z[0] in b), z[2]["dense"]), reverse=True)
    return out[:k]