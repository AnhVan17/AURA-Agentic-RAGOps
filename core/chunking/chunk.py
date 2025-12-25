from __future__ import annotations
from typing import List, Tuple, Optional
from dataclasses import dataclass
from core.chunking.tokens import count_tokens
import re

VI_DIACRITICS = r"[àáạảãăắằặẳẵâấầậẩẫđèéẹẻẽêếềệểễìíịỉĩòóọỏõôốồộổỗơớờợởỡùúụủũưứừựửữỳýỵỷỹ]"
HEADING_VI = re.compile(r"^\s*(Chương|Mục|Điều|Khoản|Phần)\b", re.IGNORECASE)
HEADING_EN = re.compile(r"^\s*(Chapter|Section|Part|Appendix)\b", re.IGNORECASE)
HEADING_NUM = re.compile(r"^\s*((\d+(\.\d+){0,3})|([IVXLCDM]+\.?)|([A-Z]\.))\s+\S+")
SENT_END = re.compile(r"([\.!\?…]+)([\)\]”»']*)\s+")
ALLCAPS = re.compile(r"^[A-Z0-9\s\-:]+$")

@dataclass
class DocumentChunk:
    id: str
    text: str
    lang: str
    heading_path: List[str]
    page_idx: Optional[int]
    start_char: int
    end_char: int
    n_tokens: int

def detect_lang_fast(s: str) -> str:
    return "vi" if re.search(VI_DIACRITICS, s.lower()) else "en"

def looks_like_heading(line: str) -> bool:
    line_stripped = line.strip()
    if not line_stripped: 
        return False

    # Giới hạn độ dài tiêu đề để tránh nhận nhầm một đoạn văn là tiêu đề
    if len(line_stripped.split()) > 15:
        return False

    if HEADING_VI.match(line_stripped): 
        return True
    if HEADING_EN.match(line_stripped): 
        return True
    if HEADING_NUM.match(line_stripped): 
        return True
    if ALLCAPS.match(line_stripped) and not line_stripped.endswith("."):
        return True
    return False

def split_sentences(text: str) -> List[str]:

    out, last = [], 0

    for m in SENT_END.finditer(text):
        end = m.end(1) + (m.end(2) - m.start(2))
        sent = text[last:end].strip()
        if sent:
            out.append(sent) 
        last = m.end()
    tail = text[last:].strip()
    if tail:
        out.append(tail)

    cleaned = []
    for sent in out:
        sent = sent.strip()
        if len(sent) < 2:
            continue
        cleaned.append(sent)
    return cleaned     
    

def heading_stack_from_lines(lines: List[str]) -> List[List[str]]:
    stack = []
    paths = []
    for ln in lines:
        if looks_like_heading(ln):
            title = ln.strip()
            depth = 1
            if re.match(r"^\s*\d+\.\d+\b", title): depth = 2
            if re.match(r"^\s*\d+\.\d+\.\d+\b", title): depth = 3
            if re.match(r"^\s*[A-Z]\.\b", title): depth = 2
            stack = stack[:depth-1] if depth > 0 else []
            stack.append(title)
        paths.append(stack.copy())
    return paths 


def chunk_heading_aware(
    text: str,
    target_tokens: int = 700,
    overlap_sentences: int = 2,
    lang_hint: Optional[str] = None,
    page_idx: Optional[int] = None,
    id_prefix: str = "C"
) -> List[DocumentChunk]:
    lang = lang_hint or detect_lang_fast(text)
    lines = text.split("\n")
    paths_per_line = heading_stack_from_lines(lines)
    sentence_spans: List[tuple[str,int,int,List[str]]] = []
    cursor = 0
    for i, ln in enumerate(lines):
        start_line = cursor
        cursor += len(ln) + 1 
        if not ln.strip():
            continue
        sents = split_sentences(ln)
        if sents:
            total_len = sum(len(s) for s in sents)
            offset = start_line
            for s in sents:
                w = len(s) / max(1, total_len)
                span_len = int(round(w * len(ln)))
                start = offset
                end = start + max(len(s), span_len)
                sentence_spans.append((s, start, end, paths_per_line[i]))

    chunks: List[DocumentChunk] = []
    buf: List[tuple[str,int,int,List[str]]] = []
    buf_tokens = 0
    cid = 0

    def flush(with_overlap: bool):
        nonlocal buf, buf_tokens, cid, chunks
        if not buf: return
        text_join = " ".join(x[0] for x in buf).strip()
        start_char = buf[0][1]; end_char = buf[-1][2]
        hp = []
        for _s, _a, _b, path in reversed(buf):
            if path: hp = path; break
        cid += 1
        chunks.append(DocumentChunk(
            id=f"{id_prefix}{cid:05d}",
            text=text_join,
            heading_path=hp,
            page_idx=page_idx,
            lang=lang,
            start_char=start_char,
            end_char=end_char,
            n_tokens=count_tokens(text_join),
        ))
        if with_overlap and overlap_sentences > 0 and len(buf) > 0:
            buf = buf[-overlap_sentences:]
            buf_tokens = count_tokens(" ".join(x[0] for x in buf))
        else:
            buf, buf_tokens = [], 0

    for s, a, b, path in sentence_spans:
        t = count_tokens(s)
        if buf_tokens + t <= target_tokens + 150:  
            buf.append((s,a,b,path))
            buf_tokens += t
        else:
            flush(with_overlap=True)
            buf.append((s,a,b,path))
            buf_tokens = count_tokens(s)
    flush(with_overlap=False)
    return chunks