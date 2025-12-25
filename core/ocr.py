from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import time
import numpy as np
from rapidocr_onnxruntime import RapidOCR 

from app.settings import APPSETTINGS
from core.telemetry.telemetry import timeit_stage

@dataclass
class OCRPage:
    page_idx: int
    render_ms: float
    ocr_ms: float
    text: str
    engine: str

@dataclass
class OCRResult:
    pages: List[OCRPage]
    text_joined: str
    total_render_ms: float
    total_ocr_ms: float

def _pdf_to_images(pdf_path: str, dpi: int, fmt: str ,  max_pages: int | None = None ) :
    from pdf2image import convert_from_path
    t0 = time.perf_counter()
    images = convert_from_path(pdf_path, dpi=dpi, fmt=fmt, first_page=1,
                            last_page=max_pages, use_cropbox = True)
    rendes_ms = (time.perf_counter() -t0) *1000 
    return images, rendes_ms

def _rapid_ocr(engine : RapidOCR ,pil_img)->str:
    try:
        img = np.array(pil_img)
        result,_ = engine(img)
        if not result:
            return ""
        return "\n".join([line[1] for line in result])
    except Exception:
        return ""

@timeit_stage("ocr_pdf")
def ocr_pdf(pdf_path: str) -> OCRResult:
    
    ocr_cfg = APPSETTINGS.ingest.ocr if hasattr(APPSETTINGS, "ingest") else{
        "dpi" : 300, 
        "fmt" : "PNG", 
        "max_pages" : 200
    } 
    dpi = ocr_cfg.get("dpi", 300)
    fmt = ocr_cfg.get("fmt", "PNG")
    max_pages = ocr_cfg.get("max_pages", None)
    images, render_ms = _pdf_to_images(pdf_path,dpi, fmt,max_pages)
    
    pages: List[OCRPage] = []
    total_ocr_ms = 0.0
    ocr_engine = RapidOCR()

    for idx,img in enumerate(images):
        t0 = time.perf_counter()
        text = _rapid_ocr(ocr_engine, img)
        engine_name = "rapidocr" if text else "none"
        ocr_ms = (time.perf_counter() - t0) * 1000
        total_ocr_ms += ocr_ms  

        pages.append(OCRPage(
            page_idx=idx, 
            render_ms=render_ms, 
            ocr_ms=ocr_ms, 
            text=text, 
            engine=engine_name))
        
    
    return OCRResult(
        pages=pages,
        text_joined="\n".join([page.text for page in pages]),
        total_render_ms=render_ms,
        total_ocr_ms=total_ocr_ms
    )
    