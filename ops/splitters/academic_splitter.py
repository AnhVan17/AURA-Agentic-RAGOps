from __future__ import annotations
from typing import Any, List, Optional
from langchain_text_splitters import TextSplitter
from langchain_core.documents import Document

from core.chunking.chunk import (
    chunk_heading_aware,
    DocumentChunk,
    detect_lang_fast,
)


class AcademicTextSplitter(TextSplitter):
    """
    Adapter Pattern: Bọc logic chunk_heading_aware() trong interface LangChain.
    
    Sử dụng:
        splitter = AcademicTextSplitter(target_tokens=700, overlap_sentences=2)
        chunks = splitter.split_documents(documents)
    """

    def __init__(
        self,
        target_tokens: int = 700,
        overlap_sentences: int = 2,
        lang_hint: Optional[str] = None,
        id_prefix: str = "C",
        **kwargs: Any,
    ):
        # TextSplitter yêu cầu chunk_size và chunk_overlap
        super().__init__(chunk_size=target_tokens, chunk_overlap=0, **kwargs)
        self.target_tokens = target_tokens
        self.overlap_sentences = overlap_sentences
        self.lang_hint = lang_hint
        self.id_prefix = id_prefix

    def split_text(self, text: str) -> List[str]:
        """
        Interface chuẩn của TextSplitter - bắt buộc phải override.
        Trả về list chuỗi text đã chặt (không kèm metadata).
        """
        chunks: List[DocumentChunk] = chunk_heading_aware(
            text=text,
            target_tokens=self.target_tokens,
            overlap_sentences=self.overlap_sentences,
            lang_hint=self.lang_hint,
            id_prefix=self.id_prefix,
        )
        return [c.text for c in chunks]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Nhận list Document từ Loader -> Chặt nhỏ -> Gắn metadata đầy đủ.
        
        Metadata output cho mỗi chunk:
        - source, file_name, method  (giữ từ Loader)
        - chunk_id     : ID duy nhất (VD: C00001)
        - section      : Heading path dạng chuỗi (VD: "Chapter 2 / Method")
        - heading_path : list heading gốc
        - page         : Số trang (giữ từ Loader)
        - lang         : Ngôn ngữ (vi/en)
        - chunk_index  : Thứ tự chunk trong document
        - n_tokens     : Số token của chunk
        """
        result: List[Document] = []

        for doc in documents:
            text = doc.page_content
            base_meta = doc.metadata.copy() if doc.metadata else {}

            # Lấy page từ metadata gốc 
            page = base_meta.get("page", None)
            lang = self.lang_hint or detect_lang_fast(text)

            chunks: List[DocumentChunk] = chunk_heading_aware(
                text=text,
                target_tokens=self.target_tokens,
                overlap_sentences=self.overlap_sentences,
                lang_hint=lang,
                page_idx=page,
                id_prefix=self.id_prefix,
            )

            for idx, chunk in enumerate(chunks):
                # Merge: metadata gốc + metadata chunking
                chunk_meta = {
                    **base_meta,
                    "chunk_id": chunk.id,
                    "section": " / ".join(chunk.heading_path) if chunk.heading_path else "",
                    "heading_path": chunk.heading_path,
                    "page": chunk.page_idx,
                    "lang": chunk.lang,
                    "chunk_index": idx,
                    "n_tokens": chunk.n_tokens,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }

                result.append(
                    Document(page_content=chunk.text, metadata=chunk_meta)
                )

        return result
