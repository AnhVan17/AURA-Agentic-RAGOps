import re
from typing import Sequence, Optional
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor

class RegexContextCompressor(BaseDocumentCompressor):
    """
    Trình nén tài liệu (Document Compressor) dùng Regex.
    Lọc bỏ các câu không chứa thông tin học thuật/kỹ thuật quan trọng,
    giúp giảm dung lượng token (VD: 2000 token -> 500 token) 
    trước khi đưa vào LLM để tạo câu trả lời.
    """
    
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Nén danh sách các Document.
        """
        compressed_docs = []

        # Các pattern kiểm tra thông tin quan trọng
        p_math = re.compile(r"([=≈≠]\s*[\d\w]|[\$\\]|(\b[A-Z]\([a-z0-9,]+\)))")  
        p_def = re.compile(r"(?:là|gọi là|định nghĩa|khái niệm|definition|defined as)\s", re.IGNORECASE)
        p_head = re.compile(r"^(Chương|Mục|Điều|Khoản|Section|Chapter|Part)\b", re.IGNORECASE)
        keywords = ["công thức", "định lý", "theorem", "lemma", "equation", "proof", "chứng minh", "ví dụ", "example", query.lower()]

        for doc in documents:
            text = doc.page_content
            if not text:
                continue
                
            lines = [x.strip() for x in text.split("\n") if x.strip()]
            keep_indices = set()

            for i, ln in enumerate(lines):
                is_relevant = False
                ln_lower = ln.lower()
                
                # Check Header
                if p_head.match(ln):
                    is_relevant = True
                # Check Math
                elif p_math.search(ln):
                    is_relevant = True
                # Check Definition
                elif p_def.search(ln):
                    is_relevant = True
                # Check Keywords & Query matching
                elif any(k in ln_lower for k in keywords):
                    is_relevant = True
                
                # Cố gắng giữ lại câu chứa từ khóa của query
                if not is_relevant:
                    query_terms = [t for t in query.lower().split() if len(t) > 3]
                    if any(qt in ln_lower for qt in query_terms):
                        is_relevant = True

                if is_relevant:
                    keep_indices.add(i)
                    # Giữ ngữ cảnh (câu trước/sau)
                    if i > 0:
                        keep_indices.add(i - 1)
                    if i < len(lines) - 1:
                        keep_indices.add(i + 1)

            # Reconstruct content
            final_lines = [lines[i] for i in sorted(list(keep_indices))]
            
            # Nếu lọc ra quá ít (<10%), fallback lấy vài đoạn đầu/cuối
            if len(final_lines) < len(lines) * 0.1:
                fallback = lines[:2] + lines[-1:] if len(lines) > 3 else lines
                compressed_text = "\n".join(fallback)
            else:
                compressed_text = "\n".join(final_lines)

            # Only retain if text exists
            if compressed_text.strip():
                # Giữ nguyên toàn bộ metadata
                new_doc = Document(
                    page_content=compressed_text, 
                    metadata=doc.metadata.copy()
                )
                if hasattr(doc, 'id'):
                    new_doc.id = doc.id
                compressed_docs.append(new_doc)

        return compressed_docs
