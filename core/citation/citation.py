from __future__ import annotations
from typing import List, Dict, Tuple

def build_citation_context(docs: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    docs: [{'text', 'heading_path', 'page_idx', ...}]
    Trả về: (context_with_markers, footnotes)
    footnotes: [{'n':1,'heading':'Chương 1 / ...','page':5}]
    """
    blocks = []
    footnotes = []
    for i, d in enumerate(docs, start=1):
        hp = " / ".join(d.get("heading_path") or [])
        page = d.get("page_idx")
        txt = d.get("text","").strip()
        if not txt:
            continue
        blocks.append(f"[{i}] {txt}")
        footnotes.append({"n": i, "heading": hp, "page": page})
    return "\n\n".join(blocks), footnotes

def render_footnotes(footnotes: List[Dict]) -> str:
    items = []
    for f in footnotes:
        h = f["heading"]
        p = f["page"]
        tail = f" — trang {p}" if p else ""
        items.append(f"[{f['n']}] {h}{tail}".strip())
    return "; ".join([x for x in items if x])