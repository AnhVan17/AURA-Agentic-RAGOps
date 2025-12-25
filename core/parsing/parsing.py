from __future__ import annotations
from typing import Optional, Literal
from dataclasses import dataclass
from pathlib import Path
import mimetypes
import time
from core.normalize import normalize_text
from core.telemetry.telemetry import timeit_stage
from core.ocr import ocr_pdf


@dataclass
class ParserResult:
    text_raw: str
    text_norm: str
    source_path: str
    parse_ms: float
    norm_ms: float
    file_type: Literal["txt", "docx", "pdf"]
    n_char_raw: int
    n_char_normalized: int
    note: Optional[str] = None


def _ext(path_str: str) -> str:
    return Path(path_str).suffix.lower()


@timeit_stage("parse_txt")
def parse_txt(filepath: str) -> ParserResult:
    return Path(filepath).read_text(encoding="utf-8", errors="ignore")


@timeit_stage("parse_docx")
def parse_docx(filepath: str) -> ParserResult:
    import docx
    doc = docx.Document(filepath)
    line = []
    for para in doc.paragraphs:
        line.append(para.text)
    return "\n".join(line)


@timeit_stage("parse_pdf_text")
def parse_pdf_text(path: str) -> str:
    from pdfminer.high_level import extract_text
    try:
        return extract_text(path) or ""
    except Exception:
        return ""


def is_low_text_pdf(path: str, threshold_chars: int = 80, max_pages: int = 3) -> bool:
    from pdfminer.high_level import extract_text
    try:
        text = extract_text(path, maxpages=max_pages) or ""
        return len(text.strip()) < threshold_chars
    except Exception:
        return True


def detect_file_type(filepath: str) -> Optional[Literal["txt", "docx", "pdf"]]:
    ext = _ext(filepath)
    if ext == ".txt":
        return "txt"
    elif ext == ".docx":
        return "docx"
    elif ext == ".pdf":
        return "pdf"

    mt, _ = mimetypes.guess_type(filepath)
    if mt == "application/pdf":
        return "pdf"
    elif mt == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return "docx"
    elif mt == "text/plain":
        return "txt"
    raise ValueError(f"Unsupported file type: {filepath}")


def parse_file(filepath: str) -> ParserResult:
    t0 = time.perf_counter()
    file_type = detect_file_type(filepath)
    if file_type == "txt":
        text_raw = parse_txt(filepath)
    elif file_type == "docx":
        text_raw = parse_docx(filepath)
    else:
        text_raw = parse_pdf_text(filepath)
        if is_low_text_pdf(filepath):
            note = "low_text_pdf"
    t1 = time.perf_counter()
    norm_txt = normalize_text(text_raw)
    t2 = time.perf_counter()

    return ParserResult(
        text_raw=text_raw,
        text_norm=norm_txt,
        source_path=filepath,
        parse_ms=(t1 - t0) * 1000,
        norm_ms=(t2 - t1) * 1000,
        file_type=file_type,
        n_char_raw=len(text_raw),
        n_char_normalized=len(norm_txt),
        note=note if 'note' in locals() else None
    )


def parse_pdf_ocr(filepath: str):
    res = parse_file(filepath)
    if res.file_type == "pdf" and res.note == "low_text_pdf":
        ocr_res = ocr_pdf(filepath)
        
        if len(ocr_res.text_joined.strip()) > 30:
            from core.normalize import normalize_text
            norm = normalize_text(ocr_res.text_joined)
            res.text_raw = ocr_res.text_joined
            res.text_norm = norm
            res.note = f"ocr_applied:{len(ocr_res.pages)}pages"
    return res


    
