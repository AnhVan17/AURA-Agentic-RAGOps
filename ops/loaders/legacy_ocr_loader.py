from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator

from core.ocr import ocr_pdf  # Import code cũ

class LegacyOCRLoader(BaseLoader):
    """Wrap code OCR cũ để có thể swap sang AWS/Google sau này."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def lazy_load(self) -> Iterator[Document]:
        result = ocr_pdf(self.file_path)
        
        for page in result.pages:
            if not page.text.strip():
                continue
            yield Document(
                page_content=page.text,
                metadata={
                    "source": self.file_path,
                    "page": page.page_idx,
                    "ocr_engine": page.engine,
                    "method": "ocr",
                }
            )