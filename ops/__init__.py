"""
LLMOps Package
==============

Package chứa các component cho LLMOps:
- observability: LangSmith tracing (Ngày 1)
- loaders: Document loaders (Ngày 2)
- splitters: Text splitters (Ngày 3)
- ... (sẽ thêm dần)
"""

# Ngày 1: Observability
from ops.observability import init_langsmith, trace_chain

# Ngày 2: Loaders
from ops.loaders import LegacyOCRLoader, AcademicDocumentLoader
