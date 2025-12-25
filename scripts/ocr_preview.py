#!/usr/bin/env python
import sys, json
from core.ocr import ocr_pdf

if __name__ == "__main__":
    pdf = sys.argv[1]
    res = ocr_pdf(pdf)
    print(json.dumps({
        "pages_ocr": len(res.pages),
        "render_ms": res.total_render_ms,
        "ocr_ms": res.total_ocr_ms,
        "preview": res.text_joined[:800]
    }, ensure_ascii=False, indent=2))
