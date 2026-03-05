from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator, List
from pathlib import Path

class AcademicDocumentLoader(BaseLoader):
    """
    Polymorphism in Data Engineering: 
    Tự động chọn loader phù hợp dựa trên định dạng và trạng thái thực tế của file (Văn bản vs Scan).
    Giao diện chung (BaseLoader) giúp dễ dàng tích hợp vào pipeline LangChain.
    """
    
    def __init__(self, file_path: str):
        self.file_path = str(file_path)
        self.file_name = Path(file_path).name
    
    def lazy_load(self) -> Iterator[Document]:
        ext = Path(self.file_path).suffix.lower()
        
        # 1. Các định dạng văn bản (Docx, Markdown, Text)
        if ext in [".docx", ".doc", ".txt", ".md"]:
            yield from self._load_with_llamaindex()
        
        # 2. PDF: Áp dụng Hybrid Strategy 
        elif ext == ".pdf":
            yield from self._load_pdf_hybrid_strategy()
        
        # 3. Ảnh (Image): OCR trực tiếp
        elif ext in [".png", ".jpg", ".jpeg"]:
            yield from self._load_with_ocr()
            
        else:
            yield from self._load_with_ocr()

    def _load_with_llamaindex(self) -> Iterator[Document]:
        """Sử dụng LlamaIndex Reader cho các file Docx/Text."""
        from llama_index.core import SimpleDirectoryReader
        
        try:
            reader = SimpleDirectoryReader(input_files=[self.file_path])
            documents = reader.load_data()
            for doc in documents:
                # Merge metadata từ LlamaIndex sang LangChain schema
                yield Document(
                    page_content=doc.text,
                    metadata={
                        "source": self.file_path,
                        "file_name": self.file_name,
                        "method": "llamaindex"
                    }
                )
        except Exception as e:
            print(f"[SmartLoader] Lỗi LlamaIndex: {e}")
            with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
                yield Document(page_content=f.read(), metadata={"source": self.file_path, "method": "raw_text"})

    def _load_pdf_hybrid_strategy(self) -> Iterator[Document]:
        """
        Chiến lược lai (Hybrid):
        - Kiểm tra mật độ văn bản trên trang đầu tiên.
        - Nếu mật độ thấp (< 100 ký tự) -> Coi là file scan -> Dùng OCR.
        - Nếu mật độ cao -> Dùng trích xuất text trực tiếp cho nhanh.
        """
        try:
            import fitz # PyMuPDF
            doc = fitz.open(self.file_path)
            
            # Kiểm tra nhanh trang đầu
            sample_text = doc[0].get_text() if len(doc) > 0 else ""
            is_scanned = len(sample_text.strip()) < 100 
            
            if is_scanned:
                print(f"[SmartLoader] {self.file_name} có vẻ là file scan. Đang chạy OCR...")
                doc.close()
                yield from self._load_with_ocr()
                return

            # PDF văn bản: Trích xuất từng trang
            for page_idx, page in enumerate(doc):
                yield Document(
                    page_content=page.get_text(),
                    metadata={
                        "source": self.file_path,
                        "file_name": self.file_name,
                        "page": page_idx + 1,
                        "method": "pymupdf"
                    }
                )
            doc.close()

        except Exception as e:
            print(f"[SmartLoader] Lỗi PDF: {e}. Fallback sang OCR.")
            yield from self._load_with_ocr()

    def _load_with_ocr(self) -> Iterator[Document]:
        """Sử dụng LegacyOCRLoader"""
        from ops.loaders.legacy_ocr_loader import LegacyOCRLoader
        loader = LegacyOCRLoader(self.file_path)
        yield from loader.lazy_load()