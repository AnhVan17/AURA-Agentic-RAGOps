from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator
from pathlib import Path

class AcademicDocumentLoader(BaseLoader):
    """Tự động chọn loader phù hợp: Text -> Fast PDF -> Fallback OCR."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def lazy_load(self) -> Iterator[Document]:
        ext = Path(self.file_path).suffix.lower()
        
        # 1. File văn bản đơn giản
        if ext in [".docx", ".doc", ".txt", ".md"]:
            yield from self._load_with_llamaindex()
        
        # 2. PDF: Thử cách nhanh trước, nếu thất bại mới dùng OCR
        elif ext == ".pdf":
            # Dùng yield from để trả về generator
            yield from self._load_pdf_hybrid_strategy()
        
        # 3. Các loại ảnh hoặc định dạng lạ
        else:
            yield from self._load_with_ocr()

    def _load_with_llamaindex(self) -> Iterator[Document]:
        from llama_index.core import SimpleDirectoryReader
        
        reader = SimpleDirectoryReader(input_files=[self.file_path])
        for doc in reader.load_data():
            yield Document(
                page_content=doc.text,
                metadata={"source": self.file_path}
            )

    def _load_pdf_hybrid_strategy(self) -> Iterator[Document]:
        """
        Logic cải tiến: Kiểm tra xem PDF có phải là scan không.
        """
        try:
            import fitz # PyMuPDF
            doc = fitz.open(self.file_path)
            
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            
            doc.close()

            # --- CHECK QUAN TRỌNG: Text Density ---
            # Nếu file 10 trang mà đọc được dưới 50 ký tự -> Chắc chắn là file scan
            if len(full_text.strip()) < 50: 
                print(f"[SmartLoader] File {self.file_path} có vẻ là file scan/ảnh. Chuyển sang OCR...")
                yield from self._load_with_ocr()
                return # Kết thúc hàm để không yield dữ liệu rác bên dưới

            # Nếu text ổn, trả về kết quả từ PyMuPDF
            # Lưu ý: PyMuPDF đọc nhanh nhưng mất công thức toán đẹp
            yield Document(
                page_content=full_text,
                metadata={"source": self.file_path, "method": "pymupdf"}
            )

        except Exception as e:
            print(f"[SmartLoader] Lỗi khi đọc nhanh PDF: {e}. Fallback sang OCR.")
            yield from self._load_with_ocr()

    def _load_with_ocr(self) -> Iterator[Document]:
        from ops.loaders.legacy_ocr_loader import LegacyOCRLoader
        print(f"[SmartLoader] Đang chạy OCR cho {self.file_path}...")
        loader = LegacyOCRLoader(self.file_path)
        yield from loader.lazy_load()