import sys, json, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.parsing.parsing import parse_pdf_ocr
from core.chunking.chunk import chunk_heading_aware, detect_lang_fast

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chunk_preview.py <path_to_file>")
        sys.exit(1)

    path = sys.argv[1]
    parsed = parse_pdf_ocr(path)
    lang = detect_lang_fast(parsed.text_norm)
    chunks = chunk_heading_aware(parsed.text_norm, target_tokens=700, overlap_sentences=2, lang_hint=lang)
    print(json.dumps({
        "lang": lang,
        "total_chunks": len(chunks),
        "avg_tokens": sum(c.n_tokens for c in chunks)//max(1,len(chunks)),
        "first_ids": [c.id for c in chunks[:5]],
        "first_headings": [c.heading_path for c in chunks[:5]]
    }, ensure_ascii=False, indent=2))
