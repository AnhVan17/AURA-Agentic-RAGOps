from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator
from pathlib import Path

from core.ocr import ocr_pdf 

class LegacyOCRLoader(BaseLoader):
    """
    Kế thừa BaseLoader chuẩn LangChain. 
    Wrap code OCR cũ để tách biệt tầng logic trích xuất và tầng logic ứng dụng.
    """
    
    def __init__(self, file_path: str):
        self.file_path = str(file_path)
        self.file_name = Path(file_path).name
    
    def lazy_load(self) -> Iterator[Document]:
        result = ocr_pdf(self.file_path)
        
        for page in result.pages:
            if not page.text.strip():
                continue
            yield Document(
                page_content=page.text,
                metadata={
                    "source": self.file_path,
                    "file_name": self.file_name,
                    "page": page.page_idx + 1, 
                    "ocr_engine": page.engine,
                    "method": "ocr",
                }
            )