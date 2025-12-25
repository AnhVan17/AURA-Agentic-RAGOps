from __future__ import annotations
from typing import List, Dict, Any

def group_by_heading(docs: List[str]) -> Dict[str, List[str]]:
    groups = {}
    for doc in docs:
        hp = " / ".join(doc.get("heading_path") or [])
        groups.setdefault(hp, []).append(doc)
    return groups

def reorder_short_to_long_group_by_heading(docs: List[Dict]) -> List[Dict]:
    groups = group_by_heading(docs)
    ordered = []

    keys = sorted(groups.keys(), key=lambda k: (k == "", k))
    for k in keys:
        arr = groups[k]
        arr.sort(key=lambda d: len(d.get("text","")))
        ordered.extend(arr)
    return ordered